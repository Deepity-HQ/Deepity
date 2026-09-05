#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

// Core Deepity
#include <deepity/utils/Activations.h>
#include <deepity/utils/Optimize.h>
#include <deepity/utils/StreamAlignedBatcher.h>

#include <deepity/layers/Layer.h>
#include <deepity/layers/ConvPCLayer.h>
#include <deepity/layers/DiscriminativePCLayer.h>
#include <deepity/layers/RBLayer.h>
#include <deepity/layers/SimpleConvPCLayer.h>
#include <deepity/layers/SimplePCLayer.h>
#include <deepity/layers/GaussSeidelPCLayer.h>
#include <deepity/layers/DirectKPPCLayer.h>

#include <deepity/networks/ConvPCNetwork.h>
#include <deepity/networks/DiscriminativePCNetwork.h>
#include <deepity/networks/SimplePCNetwork.h>
#include <deepity/networks/SimpleConvPCNetwork.h>
#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <deepity/networks/DirectKPPCNetwork.h>

namespace nb = nanobind;

// Convenience alias for the "give me a flat/any-shape, contiguous, CPU,
// dtype=float32" arrays that the original code obtained via
// py::array_t<float, py::array::c_style | py::array::forcecast>.

using FloatArray = nb::ndarray<float, nb::c_contig, nb::device::cpu>;
using IntArray = nb::ndarray<int, nb::c_contig, nb::device::cpu>;

// ============================================================================
// Internal Helpers
// ============================================================================
namespace
{

    auto resolveAct(const std::string &act) -> void (*)(float *, size_t)
    {
        if (act == "tanh")
            return Deep::tanh;
        if (act == "sigmoid")
            return Deep::sigmoid;
        if (act == "esigmoid")
            return Deep::e_sigmoid;
        if (act == "relu")
            return Deep::relu;
        if (act == "linear")
            return Deep::linear;
        return Deep::relu;
    }

    auto resolveDAct(const std::string &act) -> void (*)(float *, size_t, bool)
    {
        if (act == "dtanh")
            return Deep::dTanh;
        if (act == "dsigmoid")
            return Deep::dSigmoid;
        if (act == "d_esigmoid")
            return Deep::d_eSigmoid;
        if (act == "drelu")
            return Deep::dRelu;
        if (act == "dLinear")
            return Deep::dLinear;
        return Deep::dRelu;
    }

    Deep::ActivationType resolveActEnum(const std::string &act)
    {
        if (act == "tanh")
            return Deep::ActivationType::TANH;
        if (act == "dtanh")
            return Deep::ActivationType::dTANH;
        if (act == "relu")
            return Deep::ActivationType::RELU;
        if (act == "drelu")
            return Deep::ActivationType::dRELU;
        if (act == "sigmoid")
            return Deep::ActivationType::SIGMOID;
        if (act == "dsigmoid")
            return Deep::ActivationType::dSIGMOID;
        if (act == "esigmoid")
            return Deep::ActivationType::eSIGMOID;
        if (act == "d_esigmoid")
            return Deep::ActivationType::d_eSIGMOID;
        if (act == "dlinear")
            return Deep::ActivationType::dLINEAR;
        return Deep::ActivationType::LINEAR;
    }

    template <typename LayerT>
    void RandomizeWeightsHelper(LayerT &self)
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        self.RandomizeWeights(rng);
    }

    template <typename LayerT>
    void ClampStateHelper(LayerT &self, FloatArray input)
    {
        std::vector<float> values(input.data(), input.data() + input.size());
        self.ClampState(values);
    }

    // Wrap a pointer that is owned by (and must outlive as long as) `self`
    // into a zero-copy numpy view, the nanobind equivalent of pybind11's
    // `py::array_t<float>(shape, ptr, py::cast(&self))`.
    template <typename T, typename Owner>
    nb::ndarray<nb::numpy, T> ViewOwnedBy(T *data, std::initializer_list<size_t> shape, Owner &owner)
    {
        return nb::ndarray<nb::numpy, T>(data, shape, nb::find(&owner));
    }

    // Allocate a brand-new numpy array and copy `n` elements from `src` into
    // it. Used where the original code called `py::array_t<float>(shape, ptr)`
    // with no base object (which pybind11 implements as a copy).
    template <typename T>
    nb::ndarray<nb::numpy, T> CopyToNewArray(const T *src, std::initializer_list<size_t> shape)
    {
        size_t n = 1;
        for (auto s : shape)
            n *= s;
        T *data = new T[n];
        std::memcpy(data, src, n * sizeof(T));
        nb::capsule owner(data, [](void *p) noexcept
                          { delete[] static_cast<T *>(p); });
        return nb::ndarray<nb::numpy, T>(data, shape, owner);
    }

    template <typename LayerT>
    void BindCommonPCLayer(nb::class_<LayerT, Deep::Layer> &cls, const char *className)
    {
        cls.def("calculate_state", &LayerT::CalculateState)
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
            .def_prop_ro("beliefs", [](LayerT &self)
                         { return ViewOwnedBy(self.GetBeliefs(), {(size_t)self.GetBatchSize(), (size_t)self.GetInputSize()}, self); })
            .def_prop_ro("errors", [](LayerT &self)
                         { return ViewOwnedBy(self.GetErrors(), {(size_t)self.GetBatchSize(), (size_t)self.GetInputSize()}, self); })
            .def_prop_ro("weights", [](LayerT &self)
                         { return ViewOwnedBy(self.GetWeights(), {(size_t)self.GetOutputSize(), (size_t)self.GetInputSize()}, self); })
            .def_prop_ro("batch_size", &LayerT::GetBatchSize)
            .def_prop_ro("input_size", &LayerT::GetInputSize)
            .def_prop_ro("output_size", &LayerT::GetOutputSize)
            .def("__repr__", [className](const LayerT &self)
                 { return "<" + std::string(className) + " in=" + std::to_string(self.GetInputSize()) + ", out=" + std::to_string(self.GetOutputSize()) + ", batch=" + std::to_string(self.GetBatchSize()) + ">"; });
    }

    template <typename NetT>
    void BindCommonPCNetwork(nb::class_<NetT> &cls, const char *className)
    {
        cls.def(nb::init<int>(), nb::arg("batch_size"), "Construct a network with a fixed batch size.")
            .def("randomize_weights", [](NetT &self)
                 {
                std::random_device rd;
                std::mt19937 rng(rd());
                self.RandomizeWeights(rng); }, "Initialize every layer's weights randomly.")
            .def("clamp_input", [](NetT &self, FloatArray input)
                 {
                std::vector<float> values(input.data(), input.data() + input.size());
                self.Clamp(values); }, nb::arg("input"), "Clamp the first (input) layer to the supplied, flattened batch.")
            .def("compile", &NetT::Compile, "Compiles all layers into a contiguous block.")
            .def("calculate_state", &NetT::CalculateState, "Compute the total network energy.")
            .def("update_state", &NetT::UpdateState, "Run one inference step.")
            .def("update_weights", &NetT::UpdateWeights, "Apply weight updates to every layer.")
            .def("reset_state", &NetT::ResetState, "Resets the beliefs 'z' on every layer.")
            .def("get_terminal_layer", &NetT::GetTerminalLayer, nb::rv_policy::reference, "Return the last layer.")
            .def_prop_ro("batch_size", &NetT::GetBatchSize)
            .def_prop_ro("layers", [](NetT &self)
                         {
                nb::list result;
                for (const auto &layer : self.GetLayers()) result.append(nb::cast(layer.get(), nb::rv_policy::reference));
                return result; }, "List of layer objects owned by the network.")
            .def("__len__", [](const NetT &self)
                 { return self.GetLayers().size(); })
            .def("__getitem__", [](NetT &self, std::ptrdiff_t index)
                 {
                auto &layers = self.GetLayers();
                if (index < 0) index += static_cast<std::ptrdiff_t>(layers.size());
                if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size())) throw nb::index_error();
                return layers[index].get(); }, nb::rv_policy::reference_internal)
            .def("__repr__", [className](const NetT &self)
                 { return "<" + std::string(className) + " layers=" + std::to_string(self.GetLayers().size()) + " batch_size=" + std::to_string(self.GetBatchSize()) + ">"; });
    }

} // namespace

// ============================================================================
// Layer Bindings
// ============================================================================
void bind_layers(nb::module_ &m)
{
    nb::class_<Deep::Layer>(m, "Layer", "Abstract base class for all Predictive Coding layers.");

    auto discLayerCls = nb::class_<Deep::DiscriminativePCLayer, Deep::Layer>(m, "DiscriminativePCLayer", "Predictive Coding layer.")
                            .def("__init__", [](Deep::DiscriminativePCLayer *self, int size, int next_size, int batch_size, float learning_rate, float inference_rate, float precision_rate, float lmbda, const std::string &activation, const std::string &activation_deriv)
                                 { new (self) Deep::DiscriminativePCLayer(size, next_size, batch_size, learning_rate, inference_rate, precision_rate, lmbda, resolveAct(activation), resolveDAct(activation_deriv)); }, nb::arg("size"), nb::arg("next_size"), nb::arg("batch_size") = 1, nb::arg("learning_rate") = 1e-6f, nb::arg("inference_rate") = 0.01f, nb::arg("precision_rate") = 0.01f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu");
    BindCommonPCLayer<Deep::DiscriminativePCLayer>(discLayerCls, "DiscriminativePCLayer");
    discLayerCls.def("update_precision", &Deep::DiscriminativePCLayer::UpdatePrecision).def("set_precision_rate", &Deep::DiscriminativePCLayer::SetPrecisionRate, nb::arg("pr"));

    auto simpleLayerCls = nb::class_<Deep::SimplePCLayer, Deep::Layer>(m, "SimplePCLayer", "Predictive Coding layer without precision weighting.")
                              .def("__init__", [](Deep::SimplePCLayer *self, int size, int next_size, int batch_size, float learning_rate, float inference_rate, float lmbda, const std::string &activation, const std::string &activation_deriv)
                                   { new (self) Deep::SimplePCLayer(size, next_size, batch_size, learning_rate, inference_rate, lmbda, resolveAct(activation), resolveDAct(activation_deriv)); }, nb::arg("size"), nb::arg("next_size"), nb::arg("batch_size") = 1, nb::arg("learning_rate") = 1e-6f, nb::arg("inference_rate") = 0.01f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu");
    BindCommonPCLayer<Deep::SimplePCLayer>(simpleLayerCls, "SimplePCLayer");
    simpleLayerCls.def_prop_ro("biases", [](Deep::SimplePCLayer &self)
                               { return ViewOwnedBy(self.GetBiases(), {(size_t)self.GetOutputSize()}, self); });

    simpleLayerCls.def_prop_ro("biases", [](Deep::SimplePCLayer &self)
                               { return ViewOwnedBy(self.GetBiases(), {(size_t)self.GetOutputSize()}, self); })
        .def("set_mu_cache_threshold", &Deep::SimplePCLayer::SetMuCacheThreshold, nb::arg("threshold"),
             "Sets the mu-cache staleness threshold: -1 disables caching (default), "
             "0 reproduces the exact clamped-only behavior, >0 extends caching to "
             "unclamped layers as a genuine approximation.")
        .def("get_mu_cache_threshold", &Deep::SimplePCLayer::GetMuCacheThreshold);

    auto gsLayerCls = nb::class_<Deep::GaussSeidelPCLayer, Deep::Layer>(m, "GaussSeidelPCLayer", "PC layer with Gauss-Seidel (sequential-sweep) settling dynamics.")
                          .def("__init__", [](Deep::GaussSeidelPCLayer *self, int size, int next_size, int batch_size, float learning_rate, float inference_rate, float lmbda, const std::string &activation, const std::string &activation_deriv)
                               { new (self) Deep::GaussSeidelPCLayer(size, next_size, batch_size, learning_rate, inference_rate, lmbda, resolveAct(activation), resolveDAct(activation_deriv)); }, nb::arg("size"), nb::arg("next_size"), nb::arg("batch_size") = 1, nb::arg("learning_rate") = 1e-6f, nb::arg("inference_rate") = 0.01f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu");
    gsLayerCls.def("update_state", &Deep::GaussSeidelPCLayer::UpdateState)
        .def("compute_prediction", &Deep::GaussSeidelPCLayer::ComputePrediction)
        .def("compute_error", &Deep::GaussSeidelPCLayer::ComputeError)
        .def_prop_ro("mu", [](Deep::GaussSeidelPCLayer &self)
                     { return ViewOwnedBy(self.GetMu(), {(size_t)(self.GetBatchSize() * self.GetOutputSize())}, self); });
    BindCommonPCLayer<Deep::GaussSeidelPCLayer>(gsLayerCls, "GaussSeidelPCLayer");

    nb::class_<Deep::RBLayer, Deep::Layer>(m, "RBLayer", "Restricted Boltzmann-style Predictive Coding layer.")
        .def("__init__", [](Deep::RBLayer *self, size_t in_size, size_t out_size, float var, float var_td, float k1, float k2, float lambda, float alpha, size_t batch_size, int step_size, const std::string &activation, const std::string &activation_deriv)
             { new (self) Deep::RBLayer(in_size, out_size, var, var_td, k1, k2, lambda, alpha, batch_size, step_size, resolveAct(activation), resolveDAct(activation_deriv)); }, nb::arg("in_size"), nb::arg("out_size"), nb::arg("var") = 1.0f, nb::arg("var_td") = 10.0f, nb::arg("k1") = 1e-3f, nb::arg("k2") = 1e-5f, nb::arg("lambda") = 1e-6f, nb::arg("alpha") = 1.0f, nb::arg("batch_size") = 64, nb::arg("step_size") = 30, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu")
        .def("run_prediction", [](Deep::RBLayer &self, FloatArray input, size_t current_batch_size)
             { self.RunPrediction(input.data(), current_batch_size); }, nb::arg("input"), nb::arg("current_batch_size"))
        .def("run_inference_step", [](Deep::RBLayer &self, FloatArray bottom_up, FloatArray top_down, size_t current_batch_size)
             { self.RunInferenceStep(bottom_up.data(), top_down.data(), current_batch_size); }, nb::arg("bottom_up"), nb::arg("top_down"), nb::arg("current_batch_size"))
        .def("calc_error", [](Deep::RBLayer &self, FloatArray bottom_up, FloatArray top_down, size_t current_batch_size)
             { self.CalcError(bottom_up.data(), top_down.data(), current_batch_size); }, nb::arg("bottom_up"), nb::arg("top_down"), nb::arg("current_batch_size"))
        .def("update_beliefs", [](Deep::RBLayer &self, FloatArray bottom_up, FloatArray top_down, size_t current_batch_size)
             { self.UpdateBeliefs(bottom_up.data(), top_down.data(), current_batch_size); }, nb::arg("bottom_up"), nb::arg("top_down"), nb::arg("current_batch_size"))
        .def("calculate_state", &Deep::RBLayer::CalculateState)
        .def("update_state", &Deep::RBLayer::UpdateState)
        .def("update_weights", static_cast<void (Deep::RBLayer::*)()>(&Deep::RBLayer::UpdateWeights))
        .def("update_weights_batch", static_cast<void (Deep::RBLayer::*)(size_t)>(&Deep::RBLayer::UpdateWeights), nb::arg("current_batch_size"))
        .def("flush", &Deep::RBLayer::Flush)
        .def("attach", [](Deep::RBLayer &self, FloatArray arena)
             { self.Attach(arena.data()); }, nb::arg("arena"))
        .def_prop_ro("beliefs", [](Deep::RBLayer &self)
                     { return ViewOwnedBy(self.GetBeliefs(), {(size_t)self.GetBatchSize(), (size_t)self.GetOutputSize()}, self); })
        .def_prop_ro("errors", [](Deep::RBLayer &self)
                     { return ViewOwnedBy(self.GetErrors(), {(size_t)self.GetBatchSize(), (size_t)self.GetInputSize()}, self); })
        .def_prop_ro("weights", [](Deep::RBLayer &self)
                     { return ViewOwnedBy(self.GetWeights(), {(size_t)self.GetOutputSize(), (size_t)self.GetInputSize()}, self); })
        .def_prop_ro("batch_size", &Deep::RBLayer::GetBatchSize)
        .def_prop_ro("input_size", &Deep::RBLayer::GetInputSize)
        .def_prop_ro("output_size", &Deep::RBLayer::GetOutputSize)
        .def("total_size", &Deep::RBLayer::GetTotalSize)
        .def("__repr__", [](const Deep::RBLayer &self)
             { return "<RBLayer in=" + std::to_string(self.GetInputSize()) + ", out=" + std::to_string(self.GetOutputSize()) + ", batch=" + std::to_string(self.GetBatchSize()) + ">"; });

    auto dkpLayerCls = nb::class_<Deep::DirectKPPCLayer, Deep::Layer>(m, "DirectKPPCLayer", "Direct Kolen-Pollack Predictive Coding layer.")
                           .def("__init__", [](Deep::DirectKPPCLayer *self, size_t size, size_t next_size, size_t terminal_size, size_t batch_size, float learning_rate, float inference_rate, float feedback_rate, float lmbda, const std::string &activation, const std::string &activation_deriv)
                                { new (self) Deep::DirectKPPCLayer(size, next_size, terminal_size, batch_size, learning_rate, inference_rate, feedback_rate, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, nb::arg("size"), nb::arg("next_size"), nb::arg("terminal_size"), nb::arg("batch_size") = 1, nb::arg("learning_rate") = 1e-6f, nb::arg("inference_rate") = 0.01f, nb::arg("feedback_rate") = 1e-4f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu");
    BindCommonPCLayer<Deep::DirectKPPCLayer>(dkpLayerCls, "DirectKPPCLayer");
    dkpLayerCls.def("direct_feedback_update", &Deep::DirectKPPCLayer::DirectFeedbackUpdate,
                    "Perturbs W using the layer above's Psi and the terminal layer's error -- "
                    "the DFA phase, run once per batch before settling begins.")
        .def("set_terminal_layer", &Deep::DirectKPPCLayer::SetTerminalLayer, nb::arg("layer"))
        .def("set_psi_optimizer", [](Deep::DirectKPPCLayer &self, const std::string &opt)
             { if (opt == "ADAM") self.SetPsiOptimizer(Deep::OptimizerType::ADAM);
           else if (opt == "ADAMW") self.SetPsiOptimizer(Deep::OptimizerType::ADAMW);
           else self.SetPsiOptimizer(Deep::OptimizerType::SGD); }, nb::arg("optimizer"))
        .def("set_optimizer", [](Deep::DirectKPPCLayer &self, const std::string &opt)
             { if (opt == "ADAM") self.SetOptimizer(Deep::OptimizerType::ADAM);
           else if (opt == "ADAMW") self.SetOptimizer(Deep::OptimizerType::ADAMW);
           else self.SetOptimizer(Deep::OptimizerType::SGD); }, nb::arg("optimizer"))
        .def("set_feedback_rate", &Deep::DirectKPPCLayer::SetFeedbackRate, nb::arg("fl"))
        .def_prop_ro("terminal_size", &Deep::DirectKPPCLayer::GetTerminalSize)
        .def_prop_ro("psi", [](Deep::DirectKPPCLayer &self)
                     { return ViewOwnedBy(self.GetDirectFeedbackWeights(), {(size_t)self.GetInputSize(), (size_t)self.GetTerminalSize()}, self); })
        .def_prop_ro("biases", [](Deep::DirectKPPCLayer &self)
                     { return ViewOwnedBy(self.GetBiases(), {(size_t)self.GetOutputSize()}, self); });

    nb::class_<Deep::ConvPCLayer, Deep::Layer>(m, "ConvPCLayer", "Convolutional Predictive Coding layer.")
        .def("__init__", [](Deep::ConvPCLayer *self, int in_channels, int out_channels, int in_height, int in_width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int batch_size, float learning_rate, float inference_rate, float precision_rate, float lmbda, const std::string &activation, const std::string &activation_deriv)
             { new (self) Deep::ConvPCLayer(in_channels, out_channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, batch_size, learning_rate, inference_rate, precision_rate, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, nb::arg("in_channels"), nb::arg("out_channels"), nb::arg("in_height"), nb::arg("in_width"), nb::arg("kernel_h"), nb::arg("kernel_w"), nb::arg("stride_h") = 1, nb::arg("stride_w") = 1, nb::arg("pad_h") = 0, nb::arg("pad_w") = 0, nb::arg("batch_size") = 1, nb::arg("learning_rate") = 1e-6f, nb::arg("inference_rate") = 0.1f, nb::arg("precision_rate") = 0.0f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu")
        .def("calculate_state", &Deep::ConvPCLayer::CalculateState)
        .def("update_state", &Deep::ConvPCLayer::UpdateState)
        .def("update_weights", &Deep::ConvPCLayer::UpdateWeights)
        .def("update_precision", &Deep::ConvPCLayer::UpdatePrecision)
        .def("flush", &Deep::ConvPCLayer::Flush)
        .def("reset_state", &Deep::ConvPCLayer::ResetState)
        .def("resync_log_precision", &Deep::ConvPCLayer::ResyncLogPrecision)
        .def("clamp_state", [](Deep::ConvPCLayer &self, FloatArray input)
             {
        std::vector<float> values(input.data(), input.data() + input.size());
        self.ClampState(values); }, nb::arg("input"))
        .def("unclamp_state", &Deep::ConvPCLayer::UnclampState)
        .def("randomize_weights", [](Deep::ConvPCLayer &self)
             {
        std::random_device rd;
        std::mt19937 rng(rd());
        self.RandomizeWeights(rng); })
        .def("set_layer_above", &Deep::ConvPCLayer::SetLayerAbove, nb::rv_policy::reference)
        .def("set_layer_below", &Deep::ConvPCLayer::SetLayerBelow, nb::rv_policy::reference)
        .def("set_learning_rate", &Deep::ConvPCLayer::SetLearningRate, nb::arg("lr"))
        .def("set_inference_rate", &Deep::ConvPCLayer::SetInferenceRate, nb::arg("ir"))
        .def("set_precision_rate", &Deep::ConvPCLayer::SetPrecisionRate, nb::arg("pr"))
        .def("set_lambda", &Deep::ConvPCLayer::SetLambda, nb::arg("l"))
        .def_prop_ro("beliefs", [](Deep::ConvPCLayer &self)
                     { return ViewOwnedBy(self.GetBeliefs(), {(size_t)self.GetBatchSize(), (size_t)self.GetInChannels(), (size_t)self.GetInHeight(), (size_t)self.GetInWidth()}, self); })
        .def_prop_ro("errors", [](Deep::ConvPCLayer &self)
                     { return ViewOwnedBy(self.GetErrors(), {(size_t)self.GetBatchSize(), (size_t)self.GetInChannels(), (size_t)self.GetInHeight(), (size_t)self.GetInWidth()}, self); })
        .def_prop_ro("weights", [](Deep::ConvPCLayer &self)
                     { return ViewOwnedBy(self.GetWeights(), {(size_t)self.GetOutChannels(), (size_t)self.GetInChannels(), (size_t)self.GetKernelH(), (size_t)self.GetKernelW()}, self); })
        .def_prop_ro("biases", [](Deep::ConvPCLayer &self)
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
        .def("__repr__", [](const Deep::ConvPCLayer &self)
             { return "<ConvPCLayer in=(" + std::to_string(self.GetInChannels()) + "," + std::to_string(self.GetInHeight()) + "," + std::to_string(self.GetInWidth()) + ") out_channels=" + std::to_string(self.GetOutChannels()) + " kernel=(" + std::to_string(self.GetKernelH()) + "," + std::to_string(self.GetKernelW()) + ") batch=" + std::to_string(self.GetBatchSize()) + ">"; });

    nb::class_<Deep::SimpleConvPCLayer>(m, "SimpleConvPCLayer", "Convolutional PC layer, precision-free, AdamW-capable.")
        .def_prop_ro("beliefs", [](Deep::SimpleConvPCLayer &self)
                     {
            size_t n = self.GetBatchSize() * self.GetInputSize();
            return CopyToNewArray(self.GetBeliefs(), {n}); })
        .def_prop_ro("errors", [](Deep::SimpleConvPCLayer &self)
                     {
            size_t n = self.GetBatchSize() * self.GetInputSize();
            return CopyToNewArray(self.GetErrors(), {n}); })
        .def_prop_ro("weights", [](Deep::SimpleConvPCLayer &self)
                     {
            if (self.GetOutChannels() == 0) return nb::ndarray<nb::numpy, float>();
            size_t colRows = (size_t)self.GetInChannels() * self.GetKernelH() * self.GetKernelW();
            return ViewOwnedBy(self.GetWeights(), {(size_t)self.GetOutChannels(), colRows}, self); })
        .def_prop_ro("biases", [](Deep::SimpleConvPCLayer &self)
                     {
            if (self.GetOutChannels() == 0) return nb::ndarray<nb::numpy, float>();
            return ViewOwnedBy(self.GetBiases(), {(size_t)self.GetOutChannels()}, self); })
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
        .def("set_optimizer", [](Deep::SimpleConvPCLayer &self, const std::string &opt)
             {
            if (opt == "ADAM") self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW") self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else self.SetOptimizer(Deep::OptimizerType::SGD); }, nb::arg("optimizer"))
        .def("clamp_state", [](Deep::SimpleConvPCLayer &self, FloatArray input)
             {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.ClampState(values); }, nb::arg("input"))
        .def("unclamp_state", &Deep::SimpleConvPCLayer::UnclampState)
        .def("__repr__", [](Deep::SimpleConvPCLayer &self)
             { return "<SimpleConvPCLayer in_channels=" + std::to_string(self.GetInChannels()) + " out_channels=" + std::to_string(self.GetOutChannels()) + " in=" + std::to_string(self.GetInHeight()) + "x" + std::to_string(self.GetInWidth()) + ">"; });
}

// ============================================================================
// Network Bindings
// ============================================================================
void bind_networks(nb::module_ &m)
{
    auto discNetCls = nb::class_<Deep::DiscriminativePCNetwork>(m, "DiscriminativePCNetwork", "Predictive Coding Network.");
    BindCommonPCNetwork<Deep::DiscriminativePCNetwork>(discNetCls, "DiscriminativePCNetwork");
    discNetCls.def(nb::init<>(), "Construct a network with automatic batch-size detection.")
        .def("add_layer", [](Deep::DiscriminativePCNetwork &self, int size, int next_size, float lr, float ir, float pr, float lmbda, const std::string &activation, const std::string &activation_deriv)
             { self.AddLayer(size, next_size, lr, ir, pr, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, nb::arg("size"), nb::arg("next_size"), nb::arg("lr") = 1e-6f, nb::arg("ir") = 0.1f, nb::arg("pr") = 0.01f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu", "Add a layer to the network.")
        .def("set_inference_rate", &Deep::DiscriminativePCNetwork::SetInferenceRate, "Sets the inference rate of each layer.", nb::arg("ir"))
        .def("set_learning_rate", &Deep::DiscriminativePCNetwork::SetLearningRate, "Sets the learning rate of each layer.", nb::arg("lr"))
        .def("set_precision_rate", &Deep::DiscriminativePCNetwork::SetPrecisionRate, "Sets the precision rate of each layer.", nb::arg("pr"))
        .def("set_lambda", &Deep::DiscriminativePCNetwork::SetLambda, "Sets lambda of each layer.", nb::arg("l"))
        .def("set_optimizer", [](Deep::DiscriminativePCNetwork &self, const std::string &opt)
             { if (opt == "ADAM") self.SetOptimizer(Deep::OptimizerType::ADAM);
           else if (opt == "ADAMW") self.SetOptimizer(Deep::OptimizerType::ADAMW);
           else self.SetOptimizer(Deep::OptimizerType::SGD); }, nb::arg("optimizer"))
        .def("save", &Deep::DiscriminativePCNetwork::Save, "Saves the network architecture and weights to a structured directory.", nb::arg("dir_path"))
        .def("load", &Deep::DiscriminativePCNetwork::Load, "Loads network weights from a structured directory into the compiled MemoryArena.", nb::arg("dir_path"))
        .def("update_precision", &Deep::DiscriminativePCNetwork::UpdatePrecision, "Apply precision updates to every layer.")
        .def("project_forward", &Deep::DiscriminativePCNetwork::ProjectForward)
        .def("train_step_with_projection", [](Deep::DiscriminativePCNetwork &self, FloatArray x, FloatArray y, int steps)
             {
                 std::vector<float> xvec(x.data(), x.data() + x.size());
                 std::vector<float> yvec(y.data(), y.data() + y.size());
                 return self.TrainStepWithProjection(xvec, yvec, steps); }, nb::arg("x"), nb::arg("y"), nb::arg("steps"))
        .def("predict_with_projection", [](Deep::DiscriminativePCNetwork &self, FloatArray x, int steps)
             {
                 std::vector<float> xvec(x.data(), x.data() + x.size());
                 std::vector<float> result = self.PredictWithProjection(xvec, steps);
                 Deep::DiscriminativePCLayer *terminal = self.GetTerminalLayer();
                 return CopyToNewArray(result.data(), {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()}); }, nb::arg("x"), nb::arg("steps"));

    auto simpleNetCls = nb::class_<Deep::SimplePCNetwork>(m, "SimplePCNetwork", "Predictive Coding Network built from SimplePCLayers.");
    BindCommonPCNetwork<Deep::SimplePCNetwork>(simpleNetCls, "SimplePCNetwork");
    simpleNetCls.def("add_layer", [](Deep::SimplePCNetwork &self, int size, int next_size, float lr, float ir, float lmbda, const std::string &activation, const std::string &activation_deriv)
                     { self.AddLayer(size, next_size, lr, ir, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, nb::arg("size"), nb::arg("next_size"), nb::arg("lr") = 1e-6f, nb::arg("ir") = 0.1f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu", "Add a layer to the network.")
        .def("set_optimizer", [](Deep::SimplePCNetwork &self, const std::string &opt)
             {
            if (opt == "ADAM") self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW") self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else self.SetOptimizer(Deep::OptimizerType::SGD); }, nb::arg("optimizer"), "Sets the optimizer: ADAM, ADAMW, or SGD.")

        .def("project_forward", &Deep::SimplePCNetwork::ProjectForward, "Seeds hidden layers from a genuine forward pass through current "
                                                                        "weights, instead of zero-init. Call AFTER clamp_input(), BEFORE "
                                                                        "the settling loop.")

        .def("train_step_with_projection", [](Deep::SimplePCNetwork &self, FloatArray x, FloatArray y, int steps)
             {
    std::vector<float> xvec(x.data(), x.data() + x.size());
    std::vector<float> yvec(y.data(), y.data() + y.size());
    return self.TrainStepWithProjection(xvec, yvec, steps); }, nb::arg("x"), nb::arg("y"), nb::arg("steps"))
        .def("predict_with_projection", [](Deep::SimplePCNetwork &self, FloatArray x, int steps)
             {
    std::vector<float> xvec(x.data(), x.data() + x.size());

    std::vector<float> out_beliefs = self.PredictWithProjection(xvec, steps);

    return CopyToNewArray(out_beliefs.data(), {out_beliefs.size()}); }, nb::arg("x"), nb::arg("steps"), "Runs forward-projection init and settling loop entirely in C++, returning terminal beliefs.");
    nb::class_<Deep::GaussSeidelPCNetwork>(m, "GaussSeidelPCNetwork", "Predictive Coding Network with Gauss-Seidel settling dynamics.")
        .def(nb::init<int>(), nb::arg("batch_size"))
        .def("add_layer", [](Deep::GaussSeidelPCNetwork &self, int size, int next_size, float lr, float ir, float lmbda, const std::string &activation, const std::string &activation_deriv)
             { self.AddLayer(size, next_size, lr, ir, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, nb::arg("size"), nb::arg("next_size"), nb::arg("lr") = 1e-6f, nb::arg("ir") = 0.1f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu")
        .def("compile", &Deep::GaussSeidelPCNetwork::Compile)
        .def("randomize_weights", [](Deep::GaussSeidelPCNetwork &self)
             { std::random_device rd; std::mt19937 rng(rd()); self.RandomizeWeights(rng); })
        .def("reset_state", &Deep::GaussSeidelPCNetwork::ResetState)
        .def("clamp_input", [](Deep::GaussSeidelPCNetwork &self, FloatArray input)
             { std::vector<float> values(input.data(), input.data() + input.size());
           self.Clamp(values); }, nb::arg("input"))
        .def("step", &Deep::GaussSeidelPCNetwork::Step)
        .def("update_weights", &Deep::GaussSeidelPCNetwork::UpdateWeights)
        .def("project_forward", &Deep::GaussSeidelPCNetwork::ProjectForward)
        .def("set_optimizer", [](Deep::GaussSeidelPCNetwork &self, const std::string &opt)
             { if (opt == "ADAM") self.SetOptimizer(Deep::OptimizerType::ADAM);
           else if (opt == "ADAMW") self.SetOptimizer(Deep::OptimizerType::ADAMW);
           else self.SetOptimizer(Deep::OptimizerType::SGD); }, nb::arg("optimizer"))
        .def("set_learning_rate", &Deep::GaussSeidelPCNetwork::SetLearningRate)
        .def("train_step", [](Deep::GaussSeidelPCNetwork &self, FloatArray x, FloatArray y, int steps)
             { std::vector<float> xvec(x.data(), x.data() + x.size());
           std::vector<float> yvec(y.data(), y.data() + y.size());
           return self.TrainStep(xvec, yvec, steps); }, nb::arg("x"), nb::arg("y"), nb::arg("steps"))
        .def("train_step_with_projection", [](Deep::GaussSeidelPCNetwork &self, FloatArray x, FloatArray y, int steps)
             { std::vector<float> xvec(x.data(), x.data() + x.size());
           std::vector<float> yvec(y.data(), y.data() + y.size());
           return self.TrainStepWithProjection(xvec, yvec, steps); }, nb::arg("x"), nb::arg("y"), nb::arg("steps"))
        .def("predict", [](Deep::GaussSeidelPCNetwork &self, FloatArray x, int steps)
             { std::vector<float> xvec(x.data(), x.data() + x.size());
           std::vector<float> result = self.Predict(xvec, steps);
           Deep::GaussSeidelPCLayer *terminal = self.GetTerminalLayer();
           return CopyToNewArray(result.data(), {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()}); }, nb::arg("x"), nb::arg("steps"))
        .def_prop_ro("batch_size", &Deep::GaussSeidelPCNetwork::GetBatchSize)
        .def_prop_ro("layers", [](Deep::GaussSeidelPCNetwork &self)
                     { nb::list result;
           for (auto &layer : self.GetLayers()) result.append(nb::cast(layer.get(), nb::rv_policy::reference));
           return result; })
        .def("__len__", [](const Deep::GaussSeidelPCNetwork &self)
             { return self.GetLayers().size(); })
        .def("__getitem__", [](Deep::GaussSeidelPCNetwork &self, std::ptrdiff_t index)
             { auto &layers = self.GetLayers();
           if (index < 0) index += static_cast<std::ptrdiff_t>(layers.size());
           if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size())) throw nb::index_error();
           return layers[index].get(); }, nb::rv_policy::reference_internal);

    nb::class_<Deep::DirectKPPCNetwork>(m, "DirectKPPCNetwork", "Predictive Coding Network with Direct Kolen-Pollack feedback alignment.")
        .def(nb::init<int>(), nb::arg("batch_size"))
        .def("add_layer", [](Deep::DirectKPPCNetwork &self, size_t size, size_t next_size, size_t terminal_size, float lr, float ir, float fl, float lmbda, const std::string &activation, const std::string &activation_deriv)
             { self.AddLayer(size, next_size, terminal_size, lr, ir, fl, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, nb::arg("size"), nb::arg("next_size"), nb::arg("terminal_size"), nb::arg("lr") = 1e-6f, nb::arg("ir") = 0.1f, nb::arg("fl") = 1e-4f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu")
        .def("compile", &Deep::DirectKPPCNetwork::Compile)
        .def("randomize_weights", [](Deep::DirectKPPCNetwork &self)
             { std::random_device rd; std::mt19937 rng(rd()); self.RandomizeWeights(rng); })
        .def("reset_state", &Deep::DirectKPPCNetwork::ResetState)
        .def("clamp_input", [](Deep::DirectKPPCNetwork &self, FloatArray input)
             { std::vector<float> values(input.data(), input.data() + input.size());
           self.Clamp(values); }, nb::arg("input"))
        .def("project_forward", &Deep::DirectKPPCNetwork::ProjectForward)
        .def("calculate_terminal_error", &Deep::DirectKPPCNetwork::CalculateTerminalError)
        .def("direct_feedback_update", &Deep::DirectKPPCNetwork::DirectFeedbackUpdate)
        .def("step", &Deep::DirectKPPCNetwork::Step)
        .def("update_weights", &Deep::DirectKPPCNetwork::UpdateWeights)
        .def("set_optimizer", [](Deep::DirectKPPCNetwork &self, const std::string &opt)
             { if (opt == "ADAM") self.SetOptimizer(Deep::OptimizerType::ADAM);
           else if (opt == "ADAMW") self.SetOptimizer(Deep::OptimizerType::ADAMW);
           else self.SetOptimizer(Deep::OptimizerType::SGD); }, nb::arg("optimizer"))
        .def("set_psi_optimizer", [](Deep::DirectKPPCNetwork &self, const std::string &opt)
             { if (opt == "ADAM") self.SetPsiOptimizer(Deep::OptimizerType::ADAM);
           else if (opt == "ADAMW") self.SetPsiOptimizer(Deep::OptimizerType::ADAMW);
           else self.SetPsiOptimizer(Deep::OptimizerType::SGD); }, nb::arg("optimizer"))
        .def("set_learning_rate", &Deep::DirectKPPCNetwork::SetLearningRate, nb::arg("lr"))
        .def("set_feedback_rate", &Deep::DirectKPPCNetwork::SetFeedbackRate, nb::arg("fl"))
        .def("train_step", [](Deep::DirectKPPCNetwork &self, FloatArray x, FloatArray y, int inference_steps)
             { std::vector<float> xvec(x.data(), x.data() + x.size());
           std::vector<float> yvec(y.data(), y.data() + y.size());
           return self.TrainStep(xvec, yvec, inference_steps); }, nb::arg("x"), nb::arg("y"), nb::arg("inference_steps") = 1)
        .def("predict", [](Deep::DirectKPPCNetwork &self, FloatArray x, int inference_steps)
             { std::vector<float> xvec(x.data(), x.data() + x.size());
           std::vector<float> result = self.Predict(xvec, inference_steps);
           Deep::DirectKPPCLayer *terminal = self.GetTerminalLayer();
           return CopyToNewArray(result.data(), {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()}); }, nb::arg("x"), nb::arg("inference_steps"))
        .def_prop_ro("batch_size", &Deep::DirectKPPCNetwork::GetBatchSize)
        .def_prop_ro("layers", [](Deep::DirectKPPCNetwork &self)
                     { nb::list result;
                   for (auto &layer : self.GetLayers()) result.append(nb::cast(layer.get(), nb::rv_policy::reference));
                   return result; })
        .def("get_terminal_layer", &Deep::DirectKPPCNetwork::GetTerminalLayer, nb::rv_policy::reference)
        .def("__len__", [](const Deep::DirectKPPCNetwork &self)
             { return self.GetLayers().size(); })
        .def("__getitem__", [](Deep::DirectKPPCNetwork &self, std::ptrdiff_t index)
             { auto &layers = self.GetLayers();
           if (index < 0) index += static_cast<std::ptrdiff_t>(layers.size());
           if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size())) throw nb::index_error();
           return layers[index].get(); }, nb::rv_policy::reference_internal)
        .def("__repr__", [](const Deep::DirectKPPCNetwork &self)
             { return "<DirectKPPCNetwork layers=" + std::to_string(self.GetLayers().size()) + " batch_size=" + std::to_string(self.GetBatchSize()) + ">"; });

    nb::class_<Deep::ConvPCNetwork>(m, "ConvPCNetwork", "Convolutional Predictive Coding Network.")
        .def(nb::init<int>(), nb::arg("batch_size"), "Construct a network with a fixed batch size.")
        .def("add_layer", [](Deep::ConvPCNetwork &self, int in_channels, int out_channels, int in_height, int in_width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, float lr, float ir, float pr, float lmbda, const std::string &activation, const std::string &activation_deriv)
             { self.AddLayer(in_channels, out_channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, lr, ir, pr, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, nb::arg("in_channels"), nb::arg("out_channels"), nb::arg("in_height"), nb::arg("in_width"), nb::arg("kernel_h"), nb::arg("kernel_w"), nb::arg("stride_h") = 1, nb::arg("stride_w") = 1, nb::arg("pad_h") = 0, nb::arg("pad_w") = 0, nb::arg("lr") = 1e-6f, nb::arg("ir") = 0.1f, nb::arg("pr") = 0.0f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu")
        .def("compile", &Deep::ConvPCNetwork::Compile, "Compiles all layers into a contiguous block.")
        .def("randomize_weights", [](Deep::ConvPCNetwork &self)
             {
            std::random_device rd; std::mt19937 rng(rd()); self.RandomizeWeights(rng); })
        .def("clamp_input", [](Deep::ConvPCNetwork &self, FloatArray input)
             {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.Clamp(values); }, nb::arg("input"))
        .def("calculate_state", &Deep::ConvPCNetwork::CalculateState)
        .def("update_state", &Deep::ConvPCNetwork::UpdateState)
        .def("update_weights", &Deep::ConvPCNetwork::UpdateWeights)
        .def("update_precision", &Deep::ConvPCNetwork::UpdatePrecision)
        .def("reset_state", &Deep::ConvPCNetwork::ResetState)
        .def("train_step", [](Deep::ConvPCNetwork &self, FloatArray x, FloatArray y, int steps)
             {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStep(xvec, yvec, steps); }, nb::arg("x"), nb::arg("y"), nb::arg("steps"))
        .def("predict", [](Deep::ConvPCNetwork &self, FloatArray x, int steps)
             {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> result = self.Predict(xvec, steps);
            Deep::ConvPCLayer *terminal = self.GetTerminalLayer();
            return CopyToNewArray(result.data(), {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()}); }, nb::arg("x"), nb::arg("steps"))
        .def("get_terminal_layer", &Deep::ConvPCNetwork::GetTerminalLayer, nb::rv_policy::reference)
        .def_prop_ro("batch_size", &Deep::ConvPCNetwork::GetBatchSize)
        .def_prop_ro("layers", [](Deep::ConvPCNetwork &self)
                     {
            nb::list result;
            for (auto &layer : self.GetLayers()) result.append(nb::cast(layer.get(), nb::rv_policy::reference));
            return result; })
        .def("__len__", [](const Deep::ConvPCNetwork &self)
             { return self.GetLayers().size(); })
        .def("__getitem__", [](Deep::ConvPCNetwork &self, std::ptrdiff_t index)
             {
            auto &layers = self.GetLayers();
            if (index < 0) index += static_cast<std::ptrdiff_t>(layers.size());
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size())) throw nb::index_error();
            return layers[index].get(); }, nb::rv_policy::reference_internal)
        .def("__repr__", [](const Deep::ConvPCNetwork &self)
             { return "<ConvPCNetwork layers=" + std::to_string(self.GetLayers().size()) + " batch_size=" + std::to_string(self.GetBatchSize()) + ">"; })
        .def("train_step_with_projection", &Deep::ConvPCNetwork::TrainStepWithProjection)
        .def("predict_with_projection", &Deep::ConvPCNetwork::PredictWithProjection)
        .def("project_forward", &Deep::ConvPCNetwork::ProjectForward);

    nb::class_<Deep::SimpleConvPCNetwork>(m, "SimpleConvPCNetwork", "Convolutional Predictive Coding Network built from SimpleConvPCLayers (precision-free, AdamW-capable).")
        .def(nb::init<int>(), nb::arg("batch_size"), "Construct a network with a fixed batch size.")
        .def("add_layer", [](Deep::SimpleConvPCNetwork &self, int in_channels, int out_channels, int in_height, int in_width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, float lr, float ir, float lmbda, const std::string &activation, const std::string &activation_deriv)
             { self.AddLayer(in_channels, out_channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, lr, ir, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, nb::arg("in_channels"), nb::arg("out_channels"), nb::arg("in_height"), nb::arg("in_width"), nb::arg("kernel_h"), nb::arg("kernel_w"), nb::arg("stride_h") = 1, nb::arg("stride_w") = 1, nb::arg("pad_h") = 0, nb::arg("pad_w") = 0, nb::arg("lr") = 1e-6f, nb::arg("ir") = 0.1f, nb::arg("lmbda") = 1e-2f, nb::arg("activation") = "relu", nb::arg("activation_deriv") = "drelu")
        .def("set_optimizer", [](Deep::SimpleConvPCNetwork &self, const std::string &opt)
             {
            if (opt == "ADAM") self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW") self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else self.SetOptimizer(Deep::OptimizerType::SGD); }, nb::arg("optimizer"), "Sets the optimizer: ADAM, ADAMW, or SGD. Call BEFORE compile().")
        .def("compile", &Deep::SimpleConvPCNetwork::Compile, "Compiles all layers into a contiguous block.")
        .def("randomize_weights", [](Deep::SimpleConvPCNetwork &self)
             {
            std::random_device rd; std::mt19937 rng(rd()); self.RandomizeWeights(rng); })
        .def("clamp_input", [](Deep::SimpleConvPCNetwork &self, FloatArray input)
             {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.Clamp(values); }, nb::arg("input"))
        .def("calculate_state", &Deep::SimpleConvPCNetwork::CalculateState)
        .def("update_state", &Deep::SimpleConvPCNetwork::UpdateState)
        .def("update_weights", &Deep::SimpleConvPCNetwork::UpdateWeights)
        .def("reset_state", &Deep::SimpleConvPCNetwork::ResetState)
        .def("train_step", [](Deep::SimpleConvPCNetwork &self, FloatArray x, FloatArray y, int steps)
             {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStep(xvec, yvec, steps); }, nb::arg("x"), nb::arg("y"), nb::arg("steps"))
        .def("predict", [](Deep::SimpleConvPCNetwork &self, FloatArray x, int steps)
             {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> result = self.Predict(xvec, steps);
            Deep::SimpleConvPCLayer *terminal = self.GetTerminalLayer();
            return CopyToNewArray(result.data(), {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()}); }, nb::arg("x"), nb::arg("steps"))
        .def("get_terminal_layer", &Deep::SimpleConvPCNetwork::GetTerminalLayer, nb::rv_policy::reference)
        .def_prop_ro("batch_size", &Deep::SimpleConvPCNetwork::GetBatchSize)
        .def_prop_ro("layers", [](Deep::SimpleConvPCNetwork &self)
                     {
            nb::list result;
            for (auto &layer : self.GetLayers()) result.append(nb::cast(layer.get(), nb::rv_policy::reference));
            return result; })
        .def("__len__", [](const Deep::SimpleConvPCNetwork &self)
             { return self.GetLayers().size(); })
        .def("__getitem__", [](Deep::SimpleConvPCNetwork &self, std::ptrdiff_t index)
             {
            auto &layers = self.GetLayers();
            if (index < 0) index += static_cast<std::ptrdiff_t>(layers.size());
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size())) throw nb::index_error();
            return layers[index].get(); }, nb::rv_policy::reference_internal)
        .def("__repr__", [](const Deep::SimpleConvPCNetwork &self)
             { return "<SimpleConvPCNetwork layers=" + std::to_string(self.GetLayers().size()) + " batch_size=" + std::to_string(self.GetBatchSize()) + ">"; })
        .def("train_step_with_projection", &Deep::SimpleConvPCNetwork::TrainStepWithProjection)
        .def("predict_with_projection", &Deep::SimpleConvPCNetwork::PredictWithProjection)
        .def("project_forward", &Deep::SimpleConvPCNetwork::ProjectForward);
}

// ============================================================================
// Utilities Bindings
// ============================================================================
void bind_utilities(nb::module_ &m)
{
    nb::class_<Deep::StreamAlignedBatcher>(m, "StreamAlignedBatcher", "Builds batches with a fixed number of examples per class.")
        .def("__init__", [](Deep::StreamAlignedBatcher *self, FloatArray X, FloatArray Y, IntArray labels, size_t x_stride, size_t y_stride, int num_classes, int per_class, unsigned seed)
             {
            size_t num_samples = (size_t)X.shape(0);
            new (self) Deep::StreamAlignedBatcher(X.data(), Y.data(), labels.data(), num_samples, x_stride, y_stride, num_classes, per_class, seed); }, nb::arg("X"), nb::arg("Y"), nb::arg("labels"), nb::arg("x_stride"), nb::arg("y_stride"), nb::arg("num_classes"), nb::arg("per_class"), nb::arg("seed") = 42, nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
        .def("num_batches_per_epoch", &Deep::StreamAlignedBatcher::NumBatchesPerEpoch)
        .def_prop_ro("batch_size", &Deep::StreamAlignedBatcher::GetBatchSize)
        .def("get_batch", [](Deep::StreamAlignedBatcher &self)
             {
            int bsz = self.GetBatchSize();
            size_t xstride = self.GetXStride();
            size_t ystride = self.GetYStride();

            float *xdata = new float[(size_t)bsz * xstride];
            float *ydata = new float[(size_t)bsz * ystride];
            int *ldata = new int[(size_t)bsz];

            self.GetBatch(xdata, ydata, ldata);

            nb::capsule xowner(xdata, [](void *p) noexcept { delete[] static_cast<float *>(p); });
            nb::capsule yowner(ydata, [](void *p) noexcept { delete[] static_cast<float *>(p); });
            nb::capsule lowner(ldata, [](void *p) noexcept { delete[] static_cast<int *>(p); });

            nb::ndarray<nb::numpy, float> X_out(xdata, {(size_t)bsz, xstride}, xowner);
            nb::ndarray<nb::numpy, float> Y_out(ydata, {(size_t)bsz, ystride}, yowner);
            nb::ndarray<nb::numpy, int> labels_out(ldata, {(size_t)bsz}, lowner);

            return nb::make_tuple(X_out, Y_out, labels_out); });

    m.def("get_l2_cache_bytes", &Deep::GetL2CacheBytes);
    m.def("auto_batch_size", &Deep::AutoBatchSize);
    m.def("dynamic_thread", &Deep::DynamicThread, nb::arg("batch_size"));

    m.def("omp_max_threads", []()
          { return omp_get_max_threads(); });
    m.def("omp_num_procs", []()
          { return omp_get_num_procs(); });

    m.def("relu", [](FloatArray x)
          { Deep::relu(x.data(), x.size()); });
    m.def("drelu", [](FloatArray x)
          { Deep::dRelu(x.data(), x.size()); });
    m.def("tanh", [](FloatArray x)
          { Deep::tanh(x.data(), x.size()); });
    m.def("dtanh", [](FloatArray x)
          { Deep::dTanh(x.data(), x.size()); });
    m.def("sigmoid", [](FloatArray x)
          { Deep::sigmoid(x.data(), x.size()); });
    m.def("dsigmoid", [](FloatArray x)
          { Deep::dSigmoid(x.data(), x.size()); });
}

// ============================================================================
// Main Module Entry
// ============================================================================
NB_MODULE(pydeepity, m)
{
    m.doc() = "Deepity: A high-performance Predictive Coding library.";

    bind_layers(m);
    bind_networks(m);
    bind_utilities(m);
}
