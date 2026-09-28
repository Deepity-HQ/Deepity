/**
 * @file LayerBindings.cpp
 * @brief nanobind bindings for Deep::Layer and every concrete PC layer type.
 * Split out of the former monolithic pybinding.cpp, see
 * NetworkBindings.cpp / UtilityBindings.cpp for the rest.
 */
#include "BindingHelpers.h"
#include "LayerBindings.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <random>
#include <string>
#include <vector>

#include <deepity/utils/Activations.h>
#include <deepity/utils/Optimize.h>

#include <deepity/layers/ConvPCLayer.h>
#include <deepity/layers/DirectKPPCLayer.h>
#include <deepity/layers/DiscriminativePCLayer.h>
#include <deepity/layers/FullPCLayer.h>
#include <deepity/layers/GaussSeidelPCLayer.h>
#include <deepity/layers/Layer.h>
#include <deepity/layers/SimpleConvPCLayer.h>
#include <deepity/layers/SimplePCLayer.h>

// ============================================================================
// Internal Helpers
// ============================================================================
namespace
{

auto resolveAct(const std::string& act) -> void (*)(float*, size_t)
{
  if (act == "tanh")
    return Deep::tanh;
  if (act == "sigmoid")
    return Deep::sigmoid;
  if (act == "esigmoid")
    return Deep::e_sigmoid;
  if (act == "relu")
    return Deep::relu;
  if (act == "gelu")
    return Deep::gelu;
  if (act == "linear")
    return Deep::linear;
  return Deep::relu;
}

auto resolveDAct(const std::string& act) -> void (*)(float*, size_t, bool)
{
  if (act == "dtanh")
    return Deep::dTanh;
  if (act == "dsigmoid")
    return Deep::dSigmoid;
  if (act == "d_esigmoid")
    return Deep::d_eSigmoid;
  if (act == "drelu")
    return Deep::dRelu;
  if (act == "dgelu")
    return Deep::dGelu;
  if (act == "dLinear")
    return Deep::dLinear;
  return Deep::dRelu;
}

template <typename LayerT> void RandomizeWeightsHelper(LayerT& self)
{
  std::random_device rd;
  std::mt19937 rng(rd());
  self.RandomizeWeights(rng);
}

template <typename LayerT> void ClampStateHelper(LayerT& self, FloatArray input)
{
  std::vector<float> values(input.data(), input.data() + input.size());
  self.ClampState(values);
}

// Wrap a pointer that is owned by (and must outlive as long as) `self`
// into a zero-copy numpy view, the nanobind equivalent of pybind11's
// `py::array_t<float>(shape, ptr, py::cast(&self))`.
template <typename T, typename Owner>
nb::ndarray<nb::numpy, T> ViewOwnedBy(T* data, std::initializer_list<size_t> shape, Owner& owner)
{
  return nb::ndarray<nb::numpy, T>(data, shape, nb::find(&owner));
}

template <typename LayerT>
void BindCommonPCLayer(nb::class_<LayerT, Deep::Layer>& cls, const char* className)
{
  cls.def("calculate_state", static_cast<float (LayerT::*)() noexcept>(&LayerT::CalculateState))
      .def("update_state", &LayerT::UpdateState)
      .def("update_weights", &LayerT::UpdateWeights)
      .def("flush", &LayerT::Flush)
      .def("clamp_state", &ClampStateHelper<LayerT>, nb::arg("input"))
      .def("unclamp_state", &LayerT::UnclampState)
      .def("randomize_weights", &RandomizeWeightsHelper<LayerT>)
      .def("set_layer_above", &LayerT::SetLayerAbove, nb::rv_policy::reference)
      .def("set_layer_below", &LayerT::SetLayerBelow, nb::rv_policy::reference)
      .def("set_learning_rate", &LayerT::SetLearningRate, nb::arg("lr"))
      .def("set_inference_rate", &LayerT::SetInferenceRate, nb::arg("ir"))
      .def("set_lambda", &LayerT::SetLambda, nb::arg("l"))
      .def_prop_ro("beliefs",
                   [](LayerT& self)
                   {
                     return ViewOwnedBy(self.GetBeliefs(),
                                        {(size_t)self.GetBatchSize(), (size_t)self.GetInputSize()},
                                        self);
                   })
      .def_prop_ro("errors",
                   [](LayerT& self)
                   {
                     return ViewOwnedBy(self.GetErrors(),
                                        {(size_t)self.GetBatchSize(), (size_t)self.GetInputSize()},
                                        self);
                   })
      .def_prop_ro("weights",
                   [](LayerT& self)
                   {
                     return ViewOwnedBy(self.GetWeights(),
                                        {(size_t)self.GetOutputSize(), (size_t)self.GetInputSize()},
                                        self);
                   })
      .def_prop_ro("batch_size", &LayerT::GetBatchSize)
      .def_prop_ro("input_size", &LayerT::GetInputSize)
      .def_prop_ro("output_size", &LayerT::GetOutputSize)
      .def("__repr__",
           [className](const LayerT& self)
           {
             return "<" + std::string(className) + " in=" + std::to_string(self.GetInputSize()) +
                    ", out=" + std::to_string(self.GetOutputSize()) +
                    ", batch=" + std::to_string(self.GetBatchSize()) + ">";
           });
}

} // namespace

// ============================================================================
// Layer Bindings
// ============================================================================
void bind_layers(nb::module_& m)
{
  nb::class_<Deep::Layer>(m, "Layer", "Abstract base class for all Predictive Coding layers.");

  auto discLayerCls =
      nb::class_<Deep::DiscriminativePCLayer, Deep::Layer>(
          m, "DiscriminativePCLayer", "Predictive Coding layer.")
          .def(
              "__init__",
              [](Deep::DiscriminativePCLayer* self,
                 int size,
                 int next_size,
                 int batch_size,
                 float learning_rate,
                 float inference_rate,
                 float precision_rate,
                 float lmbda,
                 const std::string& activation,
                 const std::string& activation_deriv)
              {
                new (self) Deep::DiscriminativePCLayer(size,
                                                       next_size,
                                                       batch_size,
                                                       learning_rate,
                                                       inference_rate,
                                                       precision_rate,
                                                       lmbda,
                                                       resolveActEnum(activation),
                                                       resolveActEnum(activation_deriv));
              },
              nb::arg("size"),
              nb::arg("next_size"),
              nb::arg("batch_size") = 1,
              nb::arg("learning_rate") = 1e-6f,
              nb::arg("inference_rate") = 0.01f,
              nb::arg("precision_rate") = 0.01f,
              nb::arg("lmbda") = 1e-2f,
              nb::arg("activation") = "relu",
              nb::arg("activation_deriv") = "drelu");
  BindCommonPCLayer<Deep::DiscriminativePCLayer>(discLayerCls, "DiscriminativePCLayer");
  discLayerCls.def("update_precision", &Deep::DiscriminativePCLayer::UpdatePrecision)
      .def("set_precision_rate", &Deep::DiscriminativePCLayer::SetPrecisionRate, nb::arg("pr"));

  auto simpleLayerCls =
      nb::class_<Deep::SimplePCLayer, Deep::Layer>(
          m, "SimplePCLayer", "Predictive Coding layer without precision weighting.")
          .def(
              "__init__",
              [](Deep::SimplePCLayer* self,
                 int size,
                 int next_size,
                 int batch_size,
                 float learning_rate,
                 float inference_rate,
                 float lmbda,
                 const std::string& activation,
                 const std::string& activation_deriv)
              {
                new (self) Deep::SimplePCLayer(size,
                                               next_size,
                                               batch_size,
                                               learning_rate,
                                               inference_rate,
                                               lmbda,
                                               resolveAct(activation),
                                               resolveDAct(activation_deriv));
              },
              nb::arg("size"),
              nb::arg("next_size"),
              nb::arg("batch_size") = 1,
              nb::arg("learning_rate") = 1e-6f,
              nb::arg("inference_rate") = 0.01f,
              nb::arg("lmbda") = 1e-2f,
              nb::arg("activation") = "relu",
              nb::arg("activation_deriv") = "drelu");
  BindCommonPCLayer<Deep::SimplePCLayer>(simpleLayerCls, "SimplePCLayer");
  simpleLayerCls.def_prop_ro(
      "biases",
      [](Deep::SimplePCLayer& self)
      { return ViewOwnedBy(self.GetBiases(), {(size_t)self.GetOutputSize()}, self); });

  simpleLayerCls
      .def_prop_ro("biases",
                   [](Deep::SimplePCLayer& self)
                   { return ViewOwnedBy(self.GetBiases(), {(size_t)self.GetOutputSize()}, self); })
      .def("set_mu_cache_threshold",
           &Deep::SimplePCLayer::SetMuCacheThreshold,
           nb::arg("threshold"),
           "Sets the mu-cache staleness threshold: -1 disables caching (default), "
           "0 reproduces the exact clamped-only behavior, >0 extends caching to "
           "unclamped layers as a genuine approximation.")
      .def("get_mu_cache_threshold", &Deep::SimplePCLayer::GetMuCacheThreshold);

  auto gsLayerCls = nb::class_<Deep::GaussSeidelPCLayer, Deep::Layer>(
                        m,
                        "GaussSeidelPCLayer",
                        "PC layer with Gauss-Seidel (sequential-sweep) settling dynamics.")
                        .def(
                            "__init__",
                            [](Deep::GaussSeidelPCLayer* self,
                               int size,
                               int next_size,
                               int batch_size,
                               float learning_rate,
                               float inference_rate,
                               float lmbda,
                               const std::string& activation,
                               const std::string& activation_deriv)
                            {
                              new (self) Deep::GaussSeidelPCLayer(size,
                                                                  next_size,
                                                                  batch_size,
                                                                  learning_rate,
                                                                  inference_rate,
                                                                  lmbda,
                                                                  resolveActEnum(activation),
                                                                  resolveActEnum(activation_deriv));
                            },
                            nb::arg("size"),
                            nb::arg("next_size"),
                            nb::arg("batch_size") = 1,
                            nb::arg("learning_rate") = 1e-6f,
                            nb::arg("inference_rate") = 0.01f,
                            nb::arg("lmbda") = 1e-2f,
                            nb::arg("activation") = "relu",
                            nb::arg("activation_deriv") = "drelu");
  gsLayerCls.def("update_state", &Deep::GaussSeidelPCLayer::UpdateState)
      .def("compute_prediction", &Deep::GaussSeidelPCLayer::ComputePrediction)
      .def("compute_error", &Deep::GaussSeidelPCLayer::ComputeError)
      .def_prop_ro("mu",
                   [](Deep::GaussSeidelPCLayer& self)
                   {
                     return ViewOwnedBy(self.GetMu(),
                                        {(size_t)(self.GetBatchSize() * self.GetOutputSize())},
                                        self);
                   });
  BindCommonPCLayer<Deep::GaussSeidelPCLayer>(gsLayerCls, "GaussSeidelPCLayer");

  auto dkpLayerCls = nb::class_<Deep::DirectKPPCLayer, Deep::Layer>(
                         m, "DirectKPPCLayer", "Direct Kolen-Pollack Predictive Coding layer.")
                         .def(
                             "__init__",
                             [](Deep::DirectKPPCLayer* self,
                                size_t size,
                                size_t next_size,
                                size_t terminal_size,
                                size_t batch_size,
                                float learning_rate,
                                float inference_rate,
                                float feedback_rate,
                                float lmbda,
                                const std::string& activation,
                                const std::string& activation_deriv)
                             {
                               new (self) Deep::DirectKPPCLayer(size,
                                                                next_size,
                                                                terminal_size,
                                                                batch_size,
                                                                learning_rate,
                                                                inference_rate,
                                                                feedback_rate,
                                                                lmbda,
                                                                resolveActEnum(activation),
                                                                resolveActEnum(activation_deriv));
                             },
                             nb::arg("size"),
                             nb::arg("next_size"),
                             nb::arg("terminal_size"),
                             nb::arg("batch_size") = 1,
                             nb::arg("learning_rate") = 1e-6f,
                             nb::arg("inference_rate") = 0.01f,
                             nb::arg("feedback_rate") = 1e-4f,
                             nb::arg("lmbda") = 1e-2f,
                             nb::arg("activation") = "relu",
                             nb::arg("activation_deriv") = "drelu");
  BindCommonPCLayer<Deep::DirectKPPCLayer>(dkpLayerCls, "DirectKPPCLayer");
  dkpLayerCls
      .def("direct_feedback_update",
           &Deep::DirectKPPCLayer::DirectFeedbackUpdate,
           "Perturbs W using the layer above's Psi and the terminal layer's error, "
           "the DFA phase, run once per batch before settling begins.")
      .def("set_terminal_layer", &Deep::DirectKPPCLayer::SetTerminalLayer, nb::arg("layer"))
      .def(
          "set_psi_optimizer",
          [](Deep::DirectKPPCLayer& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetPsiOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"))
      .def(
          "set_optimizer",
          [](Deep::DirectKPPCLayer& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"))
      .def("set_feedback_rate", &Deep::DirectKPPCLayer::SetFeedbackRate, nb::arg("fl"))
      .def_prop_ro("terminal_size", &Deep::DirectKPPCLayer::GetTerminalSize)
      .def_prop_ro("psi",
                   [](Deep::DirectKPPCLayer& self)
                   {
                     return ViewOwnedBy(
                         self.GetDirectFeedbackWeights(),
                         {(size_t)self.GetInputSize(), (size_t)self.GetTerminalSize()},
                         self);
                   })
      .def_prop_ro("biases",
                   [](Deep::DirectKPPCLayer& self)
                   { return ViewOwnedBy(self.GetBiases(), {(size_t)self.GetOutputSize()}, self); });

  auto fullLayerCls = nb::class_<Deep::FullPCLayer, Deep::Layer>(
                          m,
                          "FullPCLayer",
                          "PC layer combining muPC scaling, optional residual connections, "
                          "and DKP direct feedback, every extra OFF by default, so plain "
                          "defaults reproduce DirectKPPCLayer exactly.")
                          .def(
                              "__init__",
                              [](Deep::FullPCLayer* self,
                                 size_t size,
                                 size_t next_size,
                                 size_t terminal_size,
                                 size_t batch_size,
                                 float learning_rate,
                                 float inference_rate,
                                 float feedback_rate,
                                 float lmbda,
                                 const std::string& activation,
                                 const std::string& activation_deriv)
                              {
                                new (self) Deep::FullPCLayer(size,
                                                             next_size,
                                                             terminal_size,
                                                             batch_size,
                                                             learning_rate,
                                                             inference_rate,
                                                             feedback_rate,
                                                             lmbda,
                                                             resolveActEnum(activation),
                                                             resolveActEnum(activation_deriv));
                              },
                              nb::arg("size"),
                              nb::arg("next_size"),
                              nb::arg("terminal_size"),
                              nb::arg("batch_size") = 1,
                              nb::arg("learning_rate") = 1e-6f,
                              nb::arg("inference_rate") = 0.01f,
                              nb::arg("feedback_rate") = 1e-4f,
                              nb::arg("lmbda") = 1e-2f,
                              nb::arg("activation") = "relu",
                              nb::arg("activation_deriv") = "drelu");
  BindCommonPCLayer<Deep::FullPCLayer>(fullLayerCls, "FullPCLayer");
  fullLayerCls
      .def("direct_feedback_update",
           &Deep::FullPCLayer::DirectFeedbackUpdate,
           "Perturbs W using the layer above's Psi and the terminal layer's error, "
           "the DFA phase, run once per batch before settling begins.")
      .def("set_terminal_layer", &Deep::FullPCLayer::SetTerminalLayer, nb::arg("layer"))
      .def(
          "set_psi_optimizer",
          [](Deep::FullPCLayer& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetPsiOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"))
      .def(
          "set_optimizer",
          [](Deep::FullPCLayer& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"))
      .def("set_feedback_rate", &Deep::FullPCLayer::SetFeedbackRate, nb::arg("fl"))
      .def("set_mu_pc_scale",
           &Deep::FullPCLayer::SetMuPCScale,
           nb::arg("a"),
           "Set this layer's muPC forward-scaling factor directly. Usually "
           "set by FullPCNetwork::Compile() instead, once the full "
           "architecture is known, call this directly only for manual, "
           "per-layer control outside that mechanism.")
      .def("set_residual",
           &Deep::FullPCLayer::SetResidual,
           nb::arg("enabled"),
           "Enable/disable this layer's residual connection directly. "
           "Same caveat as set_mu_pc_scale, usually set by "
           "FullPCNetwork::Compile() instead.")
      .def_prop_ro("mu_pc_scale", &Deep::FullPCLayer::GetMuPCScale)
      .def_prop_ro("residual", &Deep::FullPCLayer::GetResidual)
      .def_prop_ro("terminal_size", &Deep::FullPCLayer::GetTerminalSize)
      .def_prop_ro("psi",
                   [](Deep::FullPCLayer& self)
                   {
                     return ViewOwnedBy(
                         self.GetDirectFeedbackWeights(),
                         {(size_t)self.GetInputSize(), (size_t)self.GetTerminalSize()},
                         self);
                   })
      .def_prop_ro("biases",
                   [](Deep::FullPCLayer& self)
                   { return ViewOwnedBy(self.GetBiases(), {(size_t)self.GetOutputSize()}, self); });

  nb::class_<Deep::ConvPCLayer, Deep::Layer>(
      m, "ConvPCLayer", "Convolutional Predictive Coding layer.")
      .def(
          "__init__",
          [](Deep::ConvPCLayer* self,
             int in_channels,
             int out_channels,
             int in_height,
             int in_width,
             int kernel_h,
             int kernel_w,
             int stride_h,
             int stride_w,
             int pad_h,
             int pad_w,
             int batch_size,
             float learning_rate,
             float inference_rate,
             float precision_rate,
             float lmbda,
             const std::string& activation,
             const std::string& activation_deriv)
          {
            new (self) Deep::ConvPCLayer(in_channels,
                                         out_channels,
                                         in_height,
                                         in_width,
                                         kernel_h,
                                         kernel_w,
                                         stride_h,
                                         stride_w,
                                         pad_h,
                                         pad_w,
                                         batch_size,
                                         learning_rate,
                                         inference_rate,
                                         precision_rate,
                                         lmbda,
                                         resolveActEnum(activation),
                                         resolveActEnum(activation_deriv));
          },
          nb::arg("in_channels"),
          nb::arg("out_channels"),
          nb::arg("in_height"),
          nb::arg("in_width"),
          nb::arg("kernel_h"),
          nb::arg("kernel_w"),
          nb::arg("stride_h") = 1,
          nb::arg("stride_w") = 1,
          nb::arg("pad_h") = 0,
          nb::arg("pad_w") = 0,
          nb::arg("batch_size") = 1,
          nb::arg("learning_rate") = 1e-6f,
          nb::arg("inference_rate") = 0.1f,
          nb::arg("precision_rate") = 0.0f,
          nb::arg("lmbda") = 1e-2f,
          nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu")
      .def("calculate_state", &Deep::ConvPCLayer::CalculateState)
      .def("update_state", &Deep::ConvPCLayer::UpdateState)
      .def("update_weights", &Deep::ConvPCLayer::UpdateWeights)
      .def("update_precision", &Deep::ConvPCLayer::UpdatePrecision)
      .def("flush", &Deep::ConvPCLayer::Flush)
      .def("reset_state", &Deep::ConvPCLayer::ResetState)
      .def("resync_log_precision", &Deep::ConvPCLayer::ResyncLogPrecision)
      .def(
          "clamp_state",
          [](Deep::ConvPCLayer& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.ClampState(values);
          },
          nb::arg("input"))
      .def("unclamp_state", &Deep::ConvPCLayer::UnclampState)
      .def("randomize_weights",
           [](Deep::ConvPCLayer& self)
           {
             std::random_device rd;
             std::mt19937 rng(rd());
             self.RandomizeWeights(rng);
           })
      .def("set_layer_above", &Deep::ConvPCLayer::SetLayerAbove, nb::rv_policy::reference)
      .def("set_layer_below", &Deep::ConvPCLayer::SetLayerBelow, nb::rv_policy::reference)
      .def("set_learning_rate", &Deep::ConvPCLayer::SetLearningRate, nb::arg("lr"))
      .def("set_inference_rate", &Deep::ConvPCLayer::SetInferenceRate, nb::arg("ir"))
      .def("set_precision_rate", &Deep::ConvPCLayer::SetPrecisionRate, nb::arg("pr"))
      .def("set_lambda", &Deep::ConvPCLayer::SetLambda, nb::arg("l"))
      .def_prop_ro("beliefs",
                   [](Deep::ConvPCLayer& self)
                   {
                     return ViewOwnedBy(self.GetBeliefs(),
                                        {(size_t)self.GetBatchSize(),
                                         (size_t)self.GetInChannels(),
                                         (size_t)self.GetInHeight(),
                                         (size_t)self.GetInWidth()},
                                        self);
                   })
      .def_prop_ro("errors",
                   [](Deep::ConvPCLayer& self)
                   {
                     return ViewOwnedBy(self.GetErrors(),
                                        {(size_t)self.GetBatchSize(),
                                         (size_t)self.GetInChannels(),
                                         (size_t)self.GetInHeight(),
                                         (size_t)self.GetInWidth()},
                                        self);
                   })
      .def_prop_ro("weights",
                   [](Deep::ConvPCLayer& self)
                   {
                     return ViewOwnedBy(self.GetWeights(),
                                        {(size_t)self.GetOutChannels(),
                                         (size_t)self.GetInChannels(),
                                         (size_t)self.GetKernelH(),
                                         (size_t)self.GetKernelW()},
                                        self);
                   })
      .def_prop_ro("biases",
                   [](Deep::ConvPCLayer& self)
                   { return ViewOwnedBy(self.GetBiases(), {(size_t)self.GetOutChannels()}, self); })
      .def_prop_ro("batch_size", &Deep::ConvPCLayer::GetBatchSize)
      .def_prop_ro("in_channels", &Deep::ConvPCLayer::GetInChannels)
      .def_prop_ro("out_channels", &Deep::ConvPCLayer::GetOutChannels)
      .def_prop_ro("in_height", &Deep::ConvPCLayer::GetInHeight)
      .def_prop_ro("in_width", &Deep::ConvPCLayer::GetInWidth)
      .def_prop_ro("out_height", &Deep::ConvPCLayer::GetOutHeight)
      .def_prop_ro("out_width", &Deep::ConvPCLayer::GetOutWidth)
      .def_prop_ro("kernel_h", &Deep::ConvPCLayer::GetKernelH)
      .def_prop_ro("kernel_w", &Deep::ConvPCLayer::GetKernelW)
      .def_prop_ro("input_size", &Deep::ConvPCLayer::GetInputSize)
      .def_prop_ro("output_size", &Deep::ConvPCLayer::GetOutputSize)
      .def("__repr__",
           [](const Deep::ConvPCLayer& self)
           {
             return "<ConvPCLayer in=(" + std::to_string(self.GetInChannels()) + "," +
                    std::to_string(self.GetInHeight()) + "," + std::to_string(self.GetInWidth()) +
                    ") out_channels=" + std::to_string(self.GetOutChannels()) + " kernel=(" +
                    std::to_string(self.GetKernelH()) + "," + std::to_string(self.GetKernelW()) +
                    ") batch=" + std::to_string(self.GetBatchSize()) + ">";
           });

  nb::class_<Deep::SimpleConvPCLayer>(
      m, "SimpleConvPCLayer", "Convolutional PC layer, precision-free, AdamW-capable.")
      .def_prop_ro("beliefs",
                   [](Deep::SimpleConvPCLayer& self)
                   {
                     size_t n = self.GetBatchSize() * self.GetInputSize();
                     return CopyToNewArray(self.GetBeliefs(), {n});
                   })
      .def_prop_ro("errors",
                   [](Deep::SimpleConvPCLayer& self)
                   {
                     size_t n = self.GetBatchSize() * self.GetInputSize();
                     return CopyToNewArray(self.GetErrors(), {n});
                   })
      .def_prop_ro(
          "weights",
          [](Deep::SimpleConvPCLayer& self)
          {
            if (self.GetOutChannels() == 0)
              return nb::ndarray<nb::numpy, float>();
            size_t colRows = (size_t)self.GetInChannels() * self.GetKernelH() * self.GetKernelW();
            return ViewOwnedBy(self.GetWeights(), {(size_t)self.GetOutChannels(), colRows}, self);
          })
      .def_prop_ro("biases",
                   [](Deep::SimpleConvPCLayer& self)
                   {
                     if (self.GetOutChannels() == 0)
                       return nb::ndarray<nb::numpy, float>();
                     return ViewOwnedBy(self.GetBiases(), {(size_t)self.GetOutChannels()}, self);
                   })
      .def_prop_ro("in_channels", &Deep::SimpleConvPCLayer::GetInChannels)
      .def_prop_ro("out_channels", &Deep::SimpleConvPCLayer::GetOutChannels)
      .def_prop_ro("in_height", &Deep::SimpleConvPCLayer::GetInHeight)
      .def_prop_ro("in_width", &Deep::SimpleConvPCLayer::GetInWidth)
      .def_prop_ro("out_height", &Deep::SimpleConvPCLayer::GetOutHeight)
      .def_prop_ro("out_width", &Deep::SimpleConvPCLayer::GetOutWidth)
      .def_prop_ro("kernel_h", &Deep::SimpleConvPCLayer::GetKernelH)
      .def_prop_ro("kernel_w", &Deep::SimpleConvPCLayer::GetKernelW)
      .def_prop_ro("batch_size", &Deep::SimpleConvPCLayer::GetBatchSize)
      .def("set_learning_rate", &Deep::SimpleConvPCLayer::SetLearningRate, nb::arg("lr"))
      .def("set_inference_rate", &Deep::SimpleConvPCLayer::SetInferenceRate, nb::arg("ir"))
      .def("set_lambda", &Deep::SimpleConvPCLayer::SetLambda, nb::arg("l"))
      .def(
          "set_optimizer",
          [](Deep::SimpleConvPCLayer& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"))
      .def(
          "clamp_state",
          [](Deep::SimpleConvPCLayer& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.ClampState(values);
          },
          nb::arg("input"))
      .def("unclamp_state", &Deep::SimpleConvPCLayer::UnclampState)
      .def("__repr__",
           [](Deep::SimpleConvPCLayer& self)
           {
             return "<SimpleConvPCLayer in_channels=" + std::to_string(self.GetInChannels()) +
                    " out_channels=" + std::to_string(self.GetOutChannels()) +
                    " in=" + std::to_string(self.GetInHeight()) + "x" +
                    std::to_string(self.GetInWidth()) + ">";
           });
}

