/**
 * @file LayerBindings.cpp
 * @brief nanobind bindings for Deep::Layer and every concrete PC layer type.
 * Split out of the former monolithic pybinding.cpp, see
 * NetworkBindings.cpp / UtilityBindings.cpp for the rest.
 */
#include "LayerBindings.h"
#include "BindingHelpers.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <deepity/utils/QuantFormat.h>

#include <deepity/utils/Activations.h>
#include <deepity/utils/Optimize.h>

#include <deepity/layers/ConvPCLayer.h>
#include <deepity/layers/DirectKPPCLayer.h>
#include <deepity/layers/DiscriminativePCLayer.h>
#include <deepity/layers/FullConvPCLayer.h>
#include <deepity/layers/FullPCLayer.h>
#include <deepity/layers/FullPCLayerT.h>
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

template <typename LayerT>
void RandomizeWeightsHelper(LayerT& self, std::optional<uint32_t> seed,
                            const std::string& distribution)
{
  std::mt19937 rng =
      seed.has_value() ? std::mt19937(seed.value()) : std::mt19937(std::random_device{}());
  self.RandomizeWeights(rng, distribution.c_str());
}

template <typename LayerT> void ClampStateHelper(LayerT& self, FloatArray input)
{
  std::vector<float> values(input.data(), input.data() + input.size());
  self.ClampState(values);
}

template <typename T, typename Owner>
nb::ndarray<nb::numpy, T> ViewOwnedBy(T* data, std::initializer_list<size_t> shape, Owner& owner)
{
  return nb::ndarray<nb::numpy, T>(data, shape, nb::find(&owner));
}

template <typename T, typename LayerT>
nb::ndarray<nb::numpy, const T> CopyFromBackend(LayerT& self, const T* data,
                                                std::initializer_list<size_t> shape)
{
  size_t count = 1;
  for (size_t d : shape)
    count *= d;

  T* hostBuf = new T[count];
  self.GetBackend()->CopyToHost(hostBuf, data, count);

  nb::capsule owner(hostBuf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::ndarray<nb::numpy, const T>(hostBuf, shape, owner);
}

template <typename LayerT>
void SetWeightsHelper(LayerT& self, FloatArray values, size_t expectedCount)
{
  if ((size_t)values.size() != expectedCount)
    throw std::invalid_argument("set_weights: expected " + std::to_string(expectedCount) +
                                " values, got " + std::to_string(values.size()));
  self.GetBackend()->CopyFromHost(self.GetWeights(), values.data(), expectedCount);
}

// Same as SetWeightsHelper(), for a layer's bias buffer.
template <typename LayerT>
void SetBiasesHelper(LayerT& self, FloatArray values, size_t expectedCount)
{
  if ((size_t)values.size() != expectedCount)
    throw std::invalid_argument("set_biases: expected " + std::to_string(expectedCount) +
                                " values, got " + std::to_string(values.size()));
  self.GetBackend()->CopyFromHost(self.GetBiases(), values.data(), expectedCount);
}

template <typename LayerT>
void BindCommonPCLayer(nb::class_<LayerT, Deep::Layer>& cls, const char* className)
{
  cls.def("calculate_state", static_cast<float (LayerT::*)() noexcept>(&LayerT::CalculateState),
          "Calculates this layer's internal state (prediction error against "
          "the layer above's mu). @return Settling energy for this step.")
      .def("update_state", &LayerT::UpdateState,
           "Updates this layer's beliefs from the error computed by calculate_state().")
      .def("update_weights", &LayerT::UpdateWeights,
           "Updates this layer's weights from the state computed during settling.")
      .def("flush", &LayerT::Flush, "Flushes remaining batches.")
      .def("clamp_state", &ClampStateHelper<LayerT>, nb::arg("input"),
           "Clamps this layer's beliefs to externally-provided data, "
           "preventing them from being updated by update_state().")
      .def("unclamp_state", &LayerT::UnclampState,
           "Releases a previous clamp_state() call, allowing this layer's "
           "beliefs to update normally again.")
      .def("randomize_weights",
           &RandomizeWeightsHelper<LayerT>,
           nb::arg("seed") = nb::none(),
           nb::arg("distribution") = "normal",
           "Randomizes this layer's weights (and biases) in place. `seed` gives "
           "a reproducible run (None uses a nondeterministic seed); `distribution` "
           "is \"normal\" or \"uniform\".")
      .def("set_layer_above", &LayerT::SetLayerAbove, nb::rv_policy::reference,
           "Sets the layer immediately above this one in the network; may be "
           "None for a terminal layer.")
      .def("set_layer_below", &LayerT::SetLayerBelow, nb::rv_policy::reference,
           "Sets the layer immediately below this one in the network; may be "
           "None for the input layer.")
      .def("set_learning_rate", &LayerT::SetLearningRate, nb::arg("lr"),
           "Sets the learning rate used for weight updates.")
      .def("set_inference_rate", &LayerT::SetInferenceRate, nb::arg("ir"),
           "Sets the inference rate (Euler integration step size).")
      .def("set_lambda", &LayerT::SetLambda, nb::arg("l"),
           "Sets the weight-decay (L2 regularization) coefficient.")
      .def(
          "set_weights",
          [](LayerT& self, FloatArray values)
          {
            SetWeightsHelper(
                self, values, (size_t)self.GetOutputSize() * (size_t)self.GetInputSize());
          },
          nb::arg("values"),
          "Device-safe replacement for `layer.weights[:] = ...`; works on both CPU and GPU.")
      .def_prop_ro("beliefs",
                   [](LayerT& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetBatchSize(), (size_t)self.GetInputSize()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetBeliefs(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetBeliefs(), shape));
                   },
                   "This layer's belief state z, shape (batch_size, input_size).")
      .def_prop_ro("errors",
                   [](LayerT& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetBatchSize(), (size_t)self.GetInputSize()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetErrors(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetErrors(), shape));
                   },
                   "This layer's prediction error e, shape (batch_size, input_size).")
      .def_prop_ro("weights",
                   [](LayerT& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetOutputSize(), (size_t)self.GetInputSize()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetWeights(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetWeights(), shape));
                   },
                   "This layer's forward weights W, shape (output_size, input_size).")
      .def_prop_ro("batch_size", &LayerT::GetBatchSize, "Number of samples processed per call.")
      .def_prop_ro("input_size", &LayerT::GetInputSize, "This layer's input width.")
      .def_prop_ro("output_size", &LayerT::GetOutputSize, "This layer's output width.")
      .def("__repr__",
           [className](const LayerT& self)
           {
             return "<" + std::string(className) + " in=" + std::to_string(self.GetInputSize()) +
                    ", out=" + std::to_string(self.GetOutputSize()) +
                    ", batch=" + std::to_string(self.GetBatchSize()) + ">";
           });
}

// Generic (T-templated) counterparts to ClampStateHelper/SetWeightsHelper/
// SetBiasesHelper above, for BindFullPCLayerT<T, AccumT> -- those three
// are hardcoded to FloatArray since every OTHER bound layer is float-only;
// FullPCLayerT<T, AccumT> is the one type binding once per T.
//
// The Python-facing wire type here is AccumT, not T: for float/float and
// double/double this is T itself (no behavior change), and for every
// quantized T (Fp4E2M1Value, the posits, hwy::float16_t) Python always
// speaks AccumT (float) -- quantization happens transparently at this
// boundary via T's own explicit converting constructor, which
// std::vector's range constructor below invokes per element
// (direct-initialization, so it's allowed even though that constructor
// is `explicit`).
template <typename LayerT, typename T, typename AccumT>
void ClampStateHelperT(LayerT& self, TArray<AccumT> input)
{
  std::vector<T> values(input.data(), input.data() + input.size());
  self.ClampState(values);
}

template <typename LayerT, typename T, typename AccumT>
void SetWeightsHelperT(LayerT& self, TArray<AccumT> values, size_t expectedCount)
{
  if ((size_t)values.size() != expectedCount)
    throw std::invalid_argument("set_weights: expected " + std::to_string(expectedCount) +
                                " values, got " + std::to_string(values.size()));
  std::vector<T> encoded(values.data(), values.data() + values.size());
  self.GetBackend()->CopyFromHost(self.GetWeights(), encoded.data(), expectedCount);
}

template <typename LayerT, typename T, typename AccumT>
void SetBiasesHelperT(LayerT& self, TArray<AccumT> values, size_t expectedCount)
{
  if ((size_t)values.size() != expectedCount)
    throw std::invalid_argument("set_biases: expected " + std::to_string(expectedCount) +
                                " values, got " + std::to_string(values.size()));
  std::vector<T> encoded(values.data(), values.data() + values.size());
  self.GetBackend()->CopyFromHost(self.GetBiases(), encoded.data(), expectedCount);
}

/// @brief Templated counterpart to BindCommonPCLayer()+FullPCLayer's own
/// extra .def() calls combined, for FullPCLayerT<T, AccumT>. Not built
/// on top of BindCommonPCLayer() itself: that helper's signature requires
/// `nb::class_<LayerT, Deep::Layer>&`, and FullPCLayerT<T, AccumT>
/// deliberately doesn't derive from Deep::Layer (see FullPCLayerT's own
/// class doc comment) -- so this takes a plain `nb::class_<LayerT>&`
/// instead. Bound once per supported (T, AccumT) pair from bind_layers()
/// below, each producing its own distinct Python class
/// ("FullPCLayer_f32", "FullPCLayer_f64", ...).
template <typename T, typename AccumT>
void BindFullPCLayerT(nb::module_& m, const char* className)
{
  using LayerT = Deep::FullPCLayerT<T, AccumT>;

  nb::class_<LayerT>(
      m,
      className,
      "PC layer combining muPC scaling, optional residual connections, and DKP direct "
      "feedback, templated over its storage type -- see FullPCLayer for the float32 default.")
      .def(
          "__init__",
          [](LayerT* self, size_t size, size_t next_size, size_t terminal_size, size_t batch_size,
             AccumT learning_rate, AccumT inference_rate, AccumT feedback_rate, AccumT lmbda,
             const std::string& activation, const std::string& activation_deriv)
          {
            new (self) LayerT(size, next_size, terminal_size, batch_size, learning_rate,
                             inference_rate, feedback_rate, lmbda, resolveActEnum(activation),
                             resolveActEnum(activation_deriv));
          },
          nb::arg("size"), nb::arg("next_size"), nb::arg("terminal_size"),
          nb::arg("batch_size") = 1, nb::arg("learning_rate") = AccumT(1e-6),
          nb::arg("inference_rate") = AccumT(0.01), nb::arg("feedback_rate") = AccumT(1e-4),
          nb::arg("lmbda") = AccumT(1e-2), nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu")
      .def("calculate_state", static_cast<AccumT (LayerT::*)() noexcept>(&LayerT::CalculateState),
           "Calculates this layer's internal state (prediction error against "
           "the layer above's mu). @return Settling energy for this step.")
      .def("update_state", &LayerT::UpdateState,
           "Updates this layer's beliefs from the error computed by calculate_state().")
      .def("update_weights", &LayerT::UpdateWeights,
           "Updates this layer's weights from the state computed during settling.")
      .def("clamp_state", &ClampStateHelperT<LayerT, T, AccumT>, nb::arg("input"),
           "Clamps this layer's beliefs to externally-provided data, "
           "preventing them from being updated by update_state().")
      .def("unclamp_state", &LayerT::UnclampState,
           "Releases a previous clamp_state() call, allowing this layer's "
           "beliefs to update normally again.")
      .def("randomize_weights", &RandomizeWeightsHelper<LayerT>, nb::arg("seed") = nb::none(),
           nb::arg("distribution") = "normal",
           "Randomizes this layer's weights (and biases) in place. `seed` gives "
           "a reproducible run (None uses a nondeterministic seed); `distribution` "
           "is \"normal\" or \"uniform\".")
      .def("set_layer_above", &LayerT::SetLayerAbove, nb::rv_policy::reference,
           "Sets the layer immediately above this one in the network; may be "
           "None for a terminal layer.")
      .def("set_layer_below", &LayerT::SetLayerBelow, nb::rv_policy::reference,
           "Sets the layer immediately below this one in the network; may be "
           "None for the input layer.")
      .def("set_terminal_layer", &LayerT::SetTerminalLayer, nb::arg("layer"),
           "Sets the terminal (output) layer this layer's DKP feedback reads "
           "its error from.")
      .def("set_learning_rate", &LayerT::SetLearningRate, nb::arg("lr"),
           "Sets the learning rate used for weight updates.")
      .def("set_inference_rate", &LayerT::SetInferenceRate, nb::arg("ir"),
           "Sets the inference rate (Euler integration step size).")
      .def("set_feedback_rate", &LayerT::SetFeedbackRate, nb::arg("fl"),
           "Sets the learning rate for the direct feedback weights Psi.")
      .def("set_lambda", &LayerT::SetLambda, nb::arg("l"),
           "Sets the weight-decay (L2 regularization) coefficient.")
      .def("direct_feedback_update", &LayerT::DirectFeedbackUpdate,
           "Perturbs W using the layer above's Psi and the terminal layer's error, "
           "the DFA phase, run once per batch before settling begins.")
      .def(
          "set_optimizer",
          [](LayerT& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"),
          "Sets the optimizer used for the forward weights W: \"SGD\", \"ADAM\", or \"ADAMW\".")
      .def(
          "set_psi_optimizer",
          [](LayerT& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetPsiOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"),
          "Sets the optimizer used for the feedback weights Psi: \"SGD\", \"ADAM\", or \"ADAMW\".")
      .def("set_mu_pc_scale", &LayerT::SetMuPCScale, nb::arg("a"),
           "Set this layer's muPC forward-scaling factor directly. Usually set by "
           "FullPCNetworkT::Compile() instead.")
      .def("set_residual", &LayerT::SetResidual, nb::arg("enabled"),
           "Enable/disable this layer's residual connection directly. Same caveat as "
           "set_mu_pc_scale.")
      .def(
          "set_weights",
          [](LayerT& self, TArray<AccumT> values)
          {
            SetWeightsHelperT<LayerT, T, AccumT>(
                self, values, (size_t)self.GetOutputSize() * (size_t)self.GetInputSize());
          },
          nb::arg("values"),
          "Device-safe replacement for `layer.weights[:] = ...`; works on both CPU and GPU.")
      .def(
          "set_biases",
          [](LayerT& self, TArray<AccumT> values)
          { SetBiasesHelperT<LayerT, T, AccumT>(self, values, (size_t)self.GetOutputSize()); },
          nb::arg("values"),
          "Device-safe replacement for `layer.biases[:] = ...`; works on both CPU and GPU.")
      .def_prop_ro("beliefs",
                   [](LayerT& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetBatchSize(), (size_t)self.GetInputSize()};
                     if constexpr (!std::is_same_v<T, AccumT>)
                     {
                       return nb::cast(DecodeCopyToNewArray<T, AccumT>(self.GetBeliefs(), shape));
                     }
                     else
                     {
                       if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                         return nb::cast(ViewOwnedBy(self.GetBeliefs(), shape, self));
                       return nb::cast(CopyFromBackend(self, self.GetBeliefs(), shape));
                     }
                   },
                   "This layer's belief state z, shape (batch_size, input_size).")
      .def_prop_ro("errors",
                   [](LayerT& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetBatchSize(), (size_t)self.GetInputSize()};
                     if constexpr (!std::is_same_v<T, AccumT>)
                     {
                       return nb::cast(DecodeCopyToNewArray<T, AccumT>(self.GetErrors(), shape));
                     }
                     else
                     {
                       if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                         return nb::cast(ViewOwnedBy(self.GetErrors(), shape, self));
                       return nb::cast(CopyFromBackend(self, self.GetErrors(), shape));
                     }
                   },
                   "This layer's prediction error e, shape (batch_size, input_size).")
      .def_prop_ro("weights",
                   [](LayerT& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetOutputSize(), (size_t)self.GetInputSize()};
                     if constexpr (!std::is_same_v<T, AccumT>)
                     {
                       return nb::cast(DecodeCopyToNewArray<T, AccumT>(self.GetWeights(), shape));
                     }
                     else
                     {
                       if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                         return nb::cast(ViewOwnedBy(self.GetWeights(), shape, self));
                       return nb::cast(CopyFromBackend(self, self.GetWeights(), shape));
                     }
                   },
                   "This layer's forward weights W, shape (output_size, input_size).")
      .def_prop_ro("biases",
                   [](LayerT& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetOutputSize()};
                     if constexpr (!std::is_same_v<T, AccumT>)
                     {
                       return nb::cast(DecodeCopyToNewArray<T, AccumT>(self.GetBiases(), shape));
                     }
                     else
                     {
                       if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                         return nb::cast(ViewOwnedBy(self.GetBiases(), shape, self));
                       return nb::cast(CopyFromBackend(self, self.GetBiases(), shape));
                     }
                   },
                   "This layer's biases, shape (output_size,).")
      .def_prop_ro("psi",
                   [](LayerT& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetInputSize(), (size_t)self.GetTerminalSize()};
                     if constexpr (!std::is_same_v<T, AccumT>)
                     {
                       return nb::cast(
                           DecodeCopyToNewArray<T, AccumT>(self.GetDirectFeedbackWeights(), shape));
                     }
                     else
                     {
                       if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                         return nb::cast(ViewOwnedBy(self.GetDirectFeedbackWeights(), shape, self));
                       return nb::cast(CopyFromBackend(self, self.GetDirectFeedbackWeights(), shape));
                     }
                   },
                   "This layer's direct feedback weights Psi, shape (input_size, terminal_size).")
      .def_prop_ro("mu_pc_scale", &LayerT::GetMuPCScale,
                   "This layer's muPC forward-scaling factor `a`. See set_mu_pc_scale().")
      .def_prop_ro("residual", &LayerT::GetResidual,
                   "Whether this layer's residual connection is enabled.")
      .def_prop_ro("terminal_size", &LayerT::GetTerminalSize,
                   "Size of the terminal (output) layer this layer's DKP feedback targets.")
      .def_prop_ro("batch_size", &LayerT::GetBatchSize, "Number of samples processed per call.")
      .def_prop_ro("input_size", &LayerT::GetInputSize, "This layer's input width.")
      .def_prop_ro("output_size", &LayerT::GetOutputSize, "This layer's output width.")
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
  discLayerCls
      .def("update_precision", &Deep::DiscriminativePCLayer::UpdatePrecision,
           "Updates this layer's precision weighting from the error computed "
           "during settling.")
      .def("set_precision_rate", &Deep::DiscriminativePCLayer::SetPrecisionRate, nb::arg("pr"),
           "Sets the learning rate for precision weighting.");

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
  simpleLayerCls
      .def_prop_ro("biases",
                   [](Deep::SimplePCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetOutputSize()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetBiases(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetBiases(), shape));
                   },
                   "This layer's biases, shape (output_size,).")
      .def(
          "set_biases",
          [](Deep::SimplePCLayer& self, FloatArray values)
          { SetBiasesHelper(self, values, (size_t)self.GetOutputSize()); },
          nb::arg("values"),
          "Device-safe replacement for `layer.biases[:] = ...`; works on both CPU and GPU.")
      .def("set_mu_cache_threshold",
           &Deep::SimplePCLayer::SetMuCacheThreshold,
           nb::arg("threshold"),
           "Sets the mu-cache staleness threshold: -1 disables caching (default), "
           "0 reproduces the exact clamped-only behavior, >0 extends caching to "
           "unclamped layers as a genuine approximation.")
      .def("get_mu_cache_threshold", &Deep::SimplePCLayer::GetMuCacheThreshold,
           "Returns the current mu-cache staleness threshold. See set_mu_cache_threshold().")
      .def("set_cross_entropy",
           &Deep::SimplePCLayer::SetCrossEntropy,
           nb::arg("enabled"),
           "Enables softmax cross-entropy energy for this layer's error against "
           "layerBelow's mu (only makes sense on the terminal layer). OFF by "
           "default (plain Gaussian energy).")
      .def("get_cross_entropy", &Deep::SimplePCLayer::GetCrossEntropy,
           "Returns whether softmax cross-entropy energy is enabled. See set_cross_entropy().");

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
  gsLayerCls
      .def("update_state", &Deep::GaussSeidelPCLayer::UpdateState,
           "Sweep 1: updates this layer's beliefs z. See the class docs for "
           "the required three-sweep calling order.")
      .def("compute_prediction", &Deep::GaussSeidelPCLayer::ComputePrediction,
           "Sweep 2: recomputes this layer's outgoing prediction (mu) using "
           "the z just updated by update_state() in this same timestep. Call "
           "on every layer before any layer's compute_error().")
      .def("compute_error",
           static_cast<float (Deep::GaussSeidelPCLayer::*)(bool) noexcept>(
               &Deep::GaussSeidelPCLayer::ComputeError),
           nb::arg("need_energy") = true,
           "Sweep 3: recomputes this layer's prediction error e using the "
           "fresh predictions from compute_prediction().\n"
           "@return Settling energy for this step if need_energy, else 0.")
      .def_prop_ro("mu",
                   [](Deep::GaussSeidelPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)(self.GetBatchSize() * self.GetOutputSize())};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetMu(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetMu(), shape));
                   },
                   "This layer's outgoing prediction buffer.");
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
      .def("set_terminal_layer", &Deep::DirectKPPCLayer::SetTerminalLayer, nb::arg("layer"),
           "Sets the terminal (output) layer this layer's DKP feedback reads "
           "its error from.")
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
          nb::arg("optimizer"),
          "Sets the optimizer used for the feedback weights Psi: \"SGD\", \"ADAM\", or \"ADAMW\".")
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
          nb::arg("optimizer"),
          "Sets the optimizer used for the forward weights W: \"SGD\", \"ADAM\", or \"ADAMW\".")
      .def("set_feedback_rate", &Deep::DirectKPPCLayer::SetFeedbackRate, nb::arg("fl"),
           "Sets the learning rate for the direct feedback weights Psi.")
      .def_prop_ro("terminal_size", &Deep::DirectKPPCLayer::GetTerminalSize,
                   "Size of the terminal (output) layer this layer's DKP feedback targets.")
      .def_prop_ro("psi",
                   [](Deep::DirectKPPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetInputSize(), (size_t)self.GetTerminalSize()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetDirectFeedbackWeights(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetDirectFeedbackWeights(), shape));
                   },
                   "This layer's direct feedback weights Psi, shape (input_size, terminal_size).")
      .def_prop_ro("biases",
                   [](Deep::DirectKPPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetOutputSize()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetBiases(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetBiases(), shape));
                   },
                   "This layer's biases, shape (output_size,).")
      .def(
          "set_biases",
          [](Deep::DirectKPPCLayer& self, FloatArray values)
          { SetBiasesHelper(self, values, (size_t)self.GetOutputSize()); },
          nb::arg("values"),
          "Device-safe replacement for `layer.biases[:] = ...`; works on both CPU and GPU.");

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
      .def("set_terminal_layer", &Deep::FullPCLayer::SetTerminalLayer, nb::arg("layer"),
           "Sets the terminal (output) layer this layer's DKP feedback reads "
           "its error from.")
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
          nb::arg("optimizer"),
          "Sets the optimizer used for the feedback weights Psi: \"SGD\", \"ADAM\", or \"ADAMW\".")
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
          nb::arg("optimizer"),
          "Sets the optimizer used for the forward weights W: \"SGD\", \"ADAM\", or \"ADAMW\".")
      .def("set_feedback_rate", &Deep::FullPCLayer::SetFeedbackRate, nb::arg("fl"),
           "Sets the learning rate for the direct feedback weights Psi.")
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
      .def_prop_ro("mu_pc_scale", &Deep::FullPCLayer::GetMuPCScale,
                   "This layer's muPC forward-scaling factor `a`. See set_mu_pc_scale().")
      .def_prop_ro("residual", &Deep::FullPCLayer::GetResidual,
                   "Whether this layer's residual connection is enabled.")
      .def_prop_ro("terminal_size", &Deep::FullPCLayer::GetTerminalSize,
                   "Size of the terminal (output) layer this layer's DKP feedback targets.")
      .def_prop_ro("psi",
                   [](Deep::FullPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetInputSize(), (size_t)self.GetTerminalSize()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetDirectFeedbackWeights(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetDirectFeedbackWeights(), shape));
                   },
                   "This layer's direct feedback weights Psi, shape (input_size, terminal_size).")
      .def_prop_ro("biases",
                   [](Deep::FullPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetOutputSize()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetBiases(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetBiases(), shape));
                   },
                   "This layer's biases, shape (output_size,).")
      .def(
          "set_biases",
          [](Deep::FullPCLayer& self, FloatArray values)
          { SetBiasesHelper(self, values, (size_t)self.GetOutputSize()); },
          nb::arg("values"),
          "Device-safe replacement for `layer.biases[:] = ...`; works on both CPU and GPU.");

  nb::class_<Deep::FullConvPCLayer, Deep::Layer>(
      m,
      "FullConvPCLayer",
      "Convolutional analog of FullPCLayer: every PC variant, each independently "
      "toggleable, off by default so plain defaults reproduce SimpleConvPCLayer exactly.")
      .def(
          "__init__",
          [](Deep::FullConvPCLayer* self,
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
             int terminal_size,
             int batch_size,
             float learning_rate,
             float inference_rate,
             float feedback_rate,
             float lmbda,
             const std::string& activation,
             const std::string& activation_deriv)
          {
            new (self) Deep::FullConvPCLayer(in_channels,
                                             out_channels,
                                             in_height,
                                             in_width,
                                             kernel_h,
                                             kernel_w,
                                             stride_h,
                                             stride_w,
                                             pad_h,
                                             pad_w,
                                             terminal_size,
                                             batch_size,
                                             learning_rate,
                                             inference_rate,
                                             feedback_rate,
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
          nb::arg("terminal_size") = 0,
          nb::arg("batch_size") = 1,
          nb::arg("learning_rate") = 1e-6f,
          nb::arg("inference_rate") = 0.1f,
          nb::arg("feedback_rate") = 1e-4f,
          nb::arg("lmbda") = 1e-2f,
          nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu")
      .def("calculate_state",
           static_cast<float (Deep::FullConvPCLayer::*)()>(&Deep::FullConvPCLayer::CalculateState),
           "Calculates this layer's internal state (prediction error against "
           "the layer above's mu). @return Settling energy for this step.")
      .def("update_state", &Deep::FullConvPCLayer::UpdateState,
           "Updates this layer's beliefs from the error computed by calculate_state().")
      .def("update_weights", &Deep::FullConvPCLayer::UpdateWeights,
           "Updates this layer's weights from the state computed during settling.")
      .def("direct_feedback_update", &Deep::FullConvPCLayer::DirectFeedbackUpdate,
           "Perturbs W using the layer above's Psi and the terminal layer's error, "
           "the DFA phase, run once per batch before settling begins.")
      .def("reset_state", &Deep::FullConvPCLayer::ResetState,
           "Resets this layer's beliefs to zero, discarding any clamp.")
      .def(
          "clamp_state",
          [](Deep::FullConvPCLayer& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.ClampState(values);
          },
          nb::arg("input"),
          "Clamps this layer's beliefs to externally-provided data, "
          "preventing them from being updated by update_state().")
      .def("unclamp_state", &Deep::FullConvPCLayer::UnclampState,
           "Releases a previous clamp_state() call, allowing this layer's "
           "beliefs to update normally again.")
      .def(
          "randomize_weights",
          [](Deep::FullConvPCLayer& self,
             std::optional<uint32_t>
                 seed,
             const std::string& distribution)
          {
            std::mt19937 rng = seed.has_value() ? std::mt19937(seed.value())
                                                : std::mt19937(std::random_device{}());
            self.RandomizeWeights(rng, distribution.c_str());
          },
          nb::arg("seed") = nb::none(),
          nb::arg("distribution") = "normal",
          "Randomizes this layer's weights (and biases) in place. `seed` gives "
          "a reproducible run (None uses a nondeterministic seed); `distribution` "
          "is \"normal\" or \"uniform\".")
      .def("set_layer_above", &Deep::FullConvPCLayer::SetLayerAbove, nb::rv_policy::reference,
           "Sets the layer immediately above this one in the network; may be "
           "None for a terminal layer.")
      .def("set_layer_below", &Deep::FullConvPCLayer::SetLayerBelow, nb::rv_policy::reference,
           "Sets the layer immediately below this one in the network; may be "
           "None for the input layer.")
      .def("set_terminal_layer", &Deep::FullConvPCLayer::SetTerminalLayer, nb::arg("layer"),
           "Sets the terminal (output) layer this layer's DKP feedback reads "
           "its error from.")
      .def("set_learning_rate", &Deep::FullConvPCLayer::SetLearningRate, nb::arg("lr"),
           "Sets the learning rate used for weight updates.")
      .def("set_inference_rate", &Deep::FullConvPCLayer::SetInferenceRate, nb::arg("ir"),
           "Sets the inference rate (Euler integration step size).")
      .def("set_feedback_rate", &Deep::FullConvPCLayer::SetFeedbackRate, nb::arg("fl"),
           "Sets the learning rate for the direct feedback weights Psi.")
      .def("set_lambda", &Deep::FullConvPCLayer::SetLambda, nb::arg("l"),
           "Sets the weight-decay (L2 regularization) coefficient.")
      .def("set_mu_pc_scale", &Deep::FullConvPCLayer::SetMuPCScale, nb::arg("a"),
           "Set this layer's muPC forward-scaling factor directly. Usually set "
           "by FullConvPCNetwork::Compile() instead.")
      .def("set_residual", &Deep::FullConvPCLayer::SetResidual, nb::arg("enabled"),
           "Enable/disable this layer's residual connection directly. Same "
           "caveat as set_mu_pc_scale.")
      .def(
          "set_weights",
          [](Deep::FullConvPCLayer& self, FloatArray values)
          {
            SetWeightsHelper(self,
                             values,
                             (size_t)self.GetOutChannels() * (size_t)self.GetInChannels() *
                                 (size_t)self.GetKernelH() * (size_t)self.GetKernelW());
          },
          nb::arg("values"),
          "Device-safe replacement for `layer.weights[:] = ...`; works on both CPU and GPU.")
      .def(
          "set_biases",
          [](Deep::FullConvPCLayer& self, FloatArray values)
          { SetBiasesHelper(self, values, (size_t)self.GetOutChannels()); },
          nb::arg("values"),
          "Device-safe replacement for `layer.biases[:] = ...`; works on both CPU and GPU.")
      .def_prop_ro("beliefs",
                   [](Deep::FullConvPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetBatchSize(),
                                   (size_t)self.GetInChannels(),
                                   (size_t)self.GetInHeight(),
                                   (size_t)self.GetInWidth()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetBeliefs(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetBeliefs(), shape));
                   },
                   "This layer's belief state z, shape (batch_size, in_channels, in_height, in_width).")
      .def_prop_ro("errors",
                   [](Deep::FullConvPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetBatchSize(),
                                   (size_t)self.GetInChannels(),
                                   (size_t)self.GetInHeight(),
                                   (size_t)self.GetInWidth()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetErrors(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetErrors(), shape));
                   },
                   "This layer's prediction error e, shape (batch_size, in_channels, in_height, in_width).")
      .def_prop_ro("weights",
                   [](Deep::FullConvPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetOutChannels(),
                                   (size_t)self.GetInChannels(),
                                   (size_t)self.GetKernelH(),
                                   (size_t)self.GetKernelW()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetWeights(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetWeights(), shape));
                   },
                   "This layer's convolution kernels, shape (out_channels, in_channels, kernel_h, kernel_w).")
      .def_prop_ro("biases",
                   [](Deep::FullConvPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetOutChannels()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetBiases(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetBiases(), shape));
                   },
                   "This layer's biases, shape (out_channels,).")
      .def_prop_ro("batch_size", &Deep::FullConvPCLayer::GetBatchSize, "Number of samples processed per call.")
      .def_prop_ro("in_channels", &Deep::FullConvPCLayer::GetInChannels, "Number of input channels.")
      .def_prop_ro("out_channels", &Deep::FullConvPCLayer::GetOutChannels, "Number of output channels.")
      .def_prop_ro("in_height", &Deep::FullConvPCLayer::GetInHeight, "Input image height.")
      .def_prop_ro("in_width", &Deep::FullConvPCLayer::GetInWidth, "Input image width.")
      .def_prop_ro("out_height", &Deep::FullConvPCLayer::GetOutHeight, "Output image height.")
      .def_prop_ro("out_width", &Deep::FullConvPCLayer::GetOutWidth, "Output image width.")
      .def_prop_ro("kernel_h", &Deep::FullConvPCLayer::GetKernelH, "Convolution kernel height.")
      .def_prop_ro("kernel_w", &Deep::FullConvPCLayer::GetKernelW, "Convolution kernel width.")
      .def_prop_ro("input_size", &Deep::FullConvPCLayer::GetInputSize, "Flattened input size.")
      .def_prop_ro("output_size", &Deep::FullConvPCLayer::GetOutputSize, "Flattened output size.")
      .def_prop_ro("terminal_size", &Deep::FullConvPCLayer::GetTerminalSize,
                   "Size of the terminal (output) layer this layer's DKP feedback targets.")
      .def_prop_ro("mu_pc_scale", &Deep::FullConvPCLayer::GetMuPCScale,
                   "This layer's muPC forward-scaling factor `a`. See set_mu_pc_scale().")
      .def_prop_ro("residual", &Deep::FullConvPCLayer::GetResidual,
                   "Whether this layer's residual connection is enabled.")
      .def("__repr__",
           [](const Deep::FullConvPCLayer& self)
           {
             return "<FullConvPCLayer in=(" + std::to_string(self.GetInChannels()) + "," +
                    std::to_string(self.GetInHeight()) + "," + std::to_string(self.GetInWidth()) +
                    ") out_channels=" + std::to_string(self.GetOutChannels()) +
                    " batch=" + std::to_string(self.GetBatchSize()) + ">";
           });

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
      .def("calculate_state",
           static_cast<float (Deep::ConvPCLayer::*)() noexcept>(&Deep::ConvPCLayer::CalculateState),
           "Calculates this layer's internal state (prediction error against "
           "the layer above's mu). @return Settling energy for this step.")
      .def("update_state", &Deep::ConvPCLayer::UpdateState,
           "Updates this layer's beliefs from the error computed by calculate_state().")
      .def("update_weights", &Deep::ConvPCLayer::UpdateWeights,
           "Updates this layer's weights from the state computed during settling.")
      .def("update_precision", &Deep::ConvPCLayer::UpdatePrecision,
           "Updates this layer's precision estimate from current prediction "
           "errors, using the configured precision rate.")
      .def("flush", &Deep::ConvPCLayer::Flush, "Flushes remaining batches.")
      .def("reset_state", &Deep::ConvPCLayer::ResetState,
           "Resets this layer's beliefs/errors back to their initial values, "
           "without touching learned weights.")
      .def("resync_log_precision", &Deep::ConvPCLayer::ResyncLogPrecision,
           "Resynchronizes the cached log-precision from the current precision "
           "value; needed after directly overwriting precision outside the "
           "normal update_precision() path.")
      .def(
          "clamp_state",
          [](Deep::ConvPCLayer& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.ClampState(values);
          },
          nb::arg("input"),
          "Clamps this layer's beliefs to externally-provided data, "
          "preventing them from being updated by update_state().")
      .def("unclamp_state", &Deep::ConvPCLayer::UnclampState,
           "Releases a previous clamp_state() call, allowing this layer's "
           "beliefs to update normally again.")
      .def(
          "randomize_weights",
          [](Deep::ConvPCLayer& self, std::optional<uint32_t> seed, const std::string& distribution)
          {
            std::mt19937 rng = seed.has_value() ? std::mt19937(seed.value())
                                                : std::mt19937(std::random_device{}());
            self.RandomizeWeights(rng, distribution.c_str());
          },
          nb::arg("seed") = nb::none(),
          nb::arg("distribution") = "normal",
          "Randomizes this layer's weights (and biases) in place. `seed` gives "
          "a reproducible run (None uses a nondeterministic seed); `distribution` "
          "is \"normal\" or \"uniform\".")
      .def("set_layer_above", &Deep::ConvPCLayer::SetLayerAbove, nb::rv_policy::reference,
           "Sets the layer immediately above this one in the network; may be "
           "None for a terminal layer.")
      .def("set_layer_below", &Deep::ConvPCLayer::SetLayerBelow, nb::rv_policy::reference,
           "Sets the layer immediately below this one in the network; may be "
           "None for the input layer.")
      .def("set_learning_rate", &Deep::ConvPCLayer::SetLearningRate, nb::arg("lr"),
           "Sets the learning rate used for weight updates.")
      .def("set_inference_rate", &Deep::ConvPCLayer::SetInferenceRate, nb::arg("ir"),
           "Sets the inference rate (Euler integration step size).")
      .def("set_precision_rate", &Deep::ConvPCLayer::SetPrecisionRate, nb::arg("pr"),
           "Sets the learning rate used for precision updates.")
      .def("set_lambda", &Deep::ConvPCLayer::SetLambda, nb::arg("l"),
           "Sets the weight-decay (L2 regularization) coefficient.")
      .def(
          "set_weights",
          [](Deep::ConvPCLayer& self, FloatArray values)
          {
            SetWeightsHelper(self,
                             values,
                             (size_t)self.GetOutChannels() * (size_t)self.GetInChannels() *
                                 (size_t)self.GetKernelH() * (size_t)self.GetKernelW());
          },
          nb::arg("values"),
          "Device-safe replacement for `layer.weights[:] = ...`; works on both CPU and GPU.")
      .def(
          "set_biases",
          [](Deep::ConvPCLayer& self, FloatArray values)
          { SetBiasesHelper(self, values, (size_t)self.GetOutChannels()); },
          nb::arg("values"),
          "Device-safe replacement for `layer.biases[:] = ...`; works on both CPU and GPU.")
      .def_prop_ro("beliefs",
                   [](Deep::ConvPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetBatchSize(),
                                   (size_t)self.GetInChannels(),
                                   (size_t)self.GetInHeight(),
                                   (size_t)self.GetInWidth()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetBeliefs(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetBeliefs(), shape));
                   },
                   "This layer's belief state z, shape (batch_size, in_channels, in_height, in_width).")
      .def_prop_ro("errors",
                   [](Deep::ConvPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetBatchSize(),
                                   (size_t)self.GetInChannels(),
                                   (size_t)self.GetInHeight(),
                                   (size_t)self.GetInWidth()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetErrors(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetErrors(), shape));
                   },
                   "This layer's prediction error e, shape (batch_size, in_channels, in_height, in_width).")
      .def_prop_ro("weights",
                   [](Deep::ConvPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetOutChannels(),
                                   (size_t)self.GetInChannels(),
                                   (size_t)self.GetKernelH(),
                                   (size_t)self.GetKernelW()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetWeights(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetWeights(), shape));
                   },
                   "This layer's convolution kernels, shape (out_channels, in_channels, kernel_h, kernel_w).")
      .def_prop_ro("biases",
                   [](Deep::ConvPCLayer& self) -> nb::object
                   {
                     auto shape = {(size_t)self.GetOutChannels()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetBiases(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetBiases(), shape));
                   },
                   "This layer's biases, shape (out_channels,).")
      .def_prop_ro("batch_size", &Deep::ConvPCLayer::GetBatchSize, "Number of samples processed per call.")
      .def_prop_ro("in_channels", &Deep::ConvPCLayer::GetInChannels, "Number of input channels.")
      .def_prop_ro("out_channels", &Deep::ConvPCLayer::GetOutChannels, "Number of output channels.")
      .def_prop_ro("in_height", &Deep::ConvPCLayer::GetInHeight, "Input image height.")
      .def_prop_ro("in_width", &Deep::ConvPCLayer::GetInWidth, "Input image width.")
      .def_prop_ro("out_height", &Deep::ConvPCLayer::GetOutHeight, "Output image height.")
      .def_prop_ro("out_width", &Deep::ConvPCLayer::GetOutWidth, "Output image width.")
      .def_prop_ro("kernel_h", &Deep::ConvPCLayer::GetKernelH, "Convolution kernel height.")
      .def_prop_ro("kernel_w", &Deep::ConvPCLayer::GetKernelW, "Convolution kernel width.")
      .def_prop_ro("input_size", &Deep::ConvPCLayer::GetInputSize, "Flattened input size.")
      .def_prop_ro("output_size", &Deep::ConvPCLayer::GetOutputSize, "Flattened output size.")
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
                   [](Deep::SimpleConvPCLayer& self) -> nb::object
                   {
                     size_t n = self.GetBatchSize() * self.GetInputSize();
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(CopyToNewArray(self.GetBeliefs(), {n}));
                     return nb::cast(CopyFromBackend(self, self.GetBeliefs(), {n}));
                   },
                   "This layer's flattened belief state z.")
      .def_prop_ro("errors",
                   [](Deep::SimpleConvPCLayer& self) -> nb::object
                   {
                     size_t n = self.GetBatchSize() * self.GetInputSize();
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(CopyToNewArray(self.GetErrors(), {n}));
                     return nb::cast(CopyFromBackend(self, self.GetErrors(), {n}));
                   },
                   "This layer's flattened prediction error e.")
      .def(
          "set_weights",
          [](Deep::SimpleConvPCLayer& self, FloatArray values)
          {
            size_t colRows = (size_t)self.GetInChannels() * self.GetKernelH() * self.GetKernelW();
            SetWeightsHelper(self, values, (size_t)self.GetOutChannels() * colRows);
          },
          nb::arg("values"),
          "Device-safe replacement for `layer.weights[:] = ...`; works on both CPU and GPU.")
      .def(
          "set_biases",
          [](Deep::SimpleConvPCLayer& self, FloatArray values)
          { SetBiasesHelper(self, values, (size_t)self.GetOutChannels()); },
          nb::arg("values"),
          "Device-safe replacement for `layer.biases[:] = ...`; works on both CPU and GPU.")
      .def_prop_ro("weights",
                   [](Deep::SimpleConvPCLayer& self) -> nb::object
                   {
                     if (self.GetOutChannels() == 0)
                       return nb::cast(nb::ndarray<nb::numpy, float>());
                     size_t colRows =
                         (size_t)self.GetInChannels() * self.GetKernelH() * self.GetKernelW();
                     auto shape = {(size_t)self.GetOutChannels(), colRows};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetWeights(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetWeights(), shape));
                   },
                   "This layer's convolution kernels, shape (out_channels, "
                   "in_channels * kernel_h * kernel_w).")
      .def_prop_ro("biases",
                   [](Deep::SimpleConvPCLayer& self) -> nb::object
                   {
                     if (self.GetOutChannels() == 0)
                       return nb::cast(nb::ndarray<nb::numpy, float>());
                     auto shape = {(size_t)self.GetOutChannels()};
                     if (self.GetBackend()->GetDeviceType() == Deep::DeviceType::DEVICE_CPU)
                       return nb::cast(ViewOwnedBy(self.GetBiases(), shape, self));
                     return nb::cast(CopyFromBackend(self, self.GetBiases(), shape));
                   },
                   "This layer's biases, shape (out_channels,).")
      .def_prop_ro("in_channels", &Deep::SimpleConvPCLayer::GetInChannels, "Number of input channels.")
      .def_prop_ro("out_channels", &Deep::SimpleConvPCLayer::GetOutChannels, "Number of output channels.")
      .def_prop_ro("in_height", &Deep::SimpleConvPCLayer::GetInHeight, "Input image height.")
      .def_prop_ro("in_width", &Deep::SimpleConvPCLayer::GetInWidth, "Input image width.")
      .def_prop_ro("out_height", &Deep::SimpleConvPCLayer::GetOutHeight, "Output image height.")
      .def_prop_ro("out_width", &Deep::SimpleConvPCLayer::GetOutWidth, "Output image width.")
      .def_prop_ro("kernel_h", &Deep::SimpleConvPCLayer::GetKernelH, "Convolution kernel height.")
      .def_prop_ro("kernel_w", &Deep::SimpleConvPCLayer::GetKernelW, "Convolution kernel width.")
      .def_prop_ro("batch_size", &Deep::SimpleConvPCLayer::GetBatchSize, "Number of samples processed per call.")
      .def("set_learning_rate", &Deep::SimpleConvPCLayer::SetLearningRate, nb::arg("lr"),
           "Sets the learning rate used for weight updates.")
      .def("set_inference_rate", &Deep::SimpleConvPCLayer::SetInferenceRate, nb::arg("ir"),
           "Sets the inference rate (Euler integration step size).")
      .def("set_lambda", &Deep::SimpleConvPCLayer::SetLambda, nb::arg("l"),
           "Sets the weight-decay (L2 regularization) coefficient.")
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
          nb::arg("optimizer"),
          "Sets the optimizer used for weight updates: \"SGD\", \"ADAM\", or \"ADAMW\".")
      .def(
          "randomize_weights",
          [](Deep::SimpleConvPCLayer& self,
             std::optional<uint32_t>
                 seed,
             const std::string& distribution)
          {
            std::mt19937 rng = seed.has_value() ? std::mt19937(seed.value())
                                                : std::mt19937(std::random_device{}());
            self.RandomizeWeights(rng, distribution.c_str());
          },
          nb::arg("seed") = nb::none(),
          nb::arg("distribution") = "normal",
          "Randomizes this layer's weights (and biases) in place. `seed` gives "
          "a reproducible run (None uses a nondeterministic seed); `distribution` "
          "is \"normal\" or \"uniform\".")
      .def(
          "clamp_state",
          [](Deep::SimpleConvPCLayer& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.ClampState(values);
          },
          nb::arg("input"),
          "Clamps this layer's beliefs to externally-provided data, "
          "preventing them from being updated by update_state().")
      .def("unclamp_state", &Deep::SimpleConvPCLayer::UnclampState,
           "Releases a previous clamp_state() call, allowing this layer's "
           "beliefs to update normally again.")
      .def("__repr__",
           [](Deep::SimpleConvPCLayer& self)
           {
             return "<SimpleConvPCLayer in_channels=" + std::to_string(self.GetInChannels()) +
                    " out_channels=" + std::to_string(self.GetOutChannels()) +
                    " in=" + std::to_string(self.GetInHeight()) + "x" +
                    std::to_string(self.GetInWidth()) + ">";
           });

  BindFullPCLayerT<float, float>(m, "FullPCLayer_f32");
  BindFullPCLayerT<double, double>(m, "FullPCLayer_f64");
  BindFullPCLayerT<Deep::Fp4E2M1Value, float>(m, "FullPCLayer_fp4_e2m1");
  BindFullPCLayerT<Deep::Posit4_0Value, float>(m, "FullPCLayer_posit4_0");
  BindFullPCLayerT<Deep::Posit4_1Value, float>(m, "FullPCLayer_posit4_1");
  BindFullPCLayerT<Deep::Posit4_2Value, float>(m, "FullPCLayer_posit4_2");
}
