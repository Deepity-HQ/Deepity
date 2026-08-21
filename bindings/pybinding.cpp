#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

// Core Deepity
#include <deepity/Activations.h>
#include <deepity/Optimize.h>
#include <deepity/StreamAlignedBatcher.h>

// Deepity Layers
#include <deepity/layers/Layer.h>
#include <deepity/layers/ConvPCLayer.h>
#include <deepity/layers/DiscriminativePCLayer.h>
#include <deepity/layers/RBLayer.h>
#include <deepity/layers/SimpleConvPCLayer.h>
#include <deepity/layers/SimplePCLayer.h>

// Deepity Networks
#include <deepity/networks/ConvPCNetwork.h>
#include <deepity/networks/DiscriminativePCNetwork.h>
#include <deepity/networks/SimplePCNetwork.h>

namespace py = pybind11;

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
    void ClampStateHelper(LayerT &self, py::array_t<float, py::array::c_style | py::array::forcecast> input)
    {
        auto buf = input.request();
        std::vector<float> values(static_cast<float *>(buf.ptr), static_cast<float *>(buf.ptr) + buf.size);
        self.ClampState(values);
    }

    template <typename LayerT>
    void BindCommonPCLayer(py::class_<LayerT, Deep::Layer> &cls, const char *className)
    {
        cls.def("calculate_state", &LayerT::CalculateState)
            .def("update_state", &LayerT::UpdateState)
            .def("update_weights", &LayerT::UpdateWeights)
            .def("flush", &LayerT::Flush)
            .def("clamp_state", &ClampStateHelper<LayerT>, py::arg("input"))
            .def("unclamp_state", &LayerT::UnclampState)
            .def("randomize_weights", &RandomizeWeightsHelper<LayerT>)
            .def("set_layer_above", &LayerT::SetLayerAbove, py::return_value_policy::reference)
            .def("set_layer_below", &LayerT::SetLayerBelow, py::return_value_policy::reference)
            .def("set_learning_rate", &LayerT::SetLearningRate, py::arg("lr"))
            .def("set_inference_rate", &LayerT::SetInferenceRate, py::arg("ir"))
            .def("set_lambda", &LayerT::SetLambda, py::arg("l"))
            .def_property_readonly("beliefs", [](LayerT &self)
                                   { return py::array_t<float>({(py::ssize_t)self.GetBatchSize(), (py::ssize_t)self.GetInputSize()}, self.GetBeliefs(), py::cast(&self)); })
            .def_property_readonly("errors", [](LayerT &self)
                                   { return py::array_t<float>({(py::ssize_t)self.GetBatchSize(), (py::ssize_t)self.GetInputSize()}, self.GetErrors(), py::cast(&self)); })
            .def_property_readonly("weights", [](LayerT &self)
                                   { return py::array_t<float>({(py::ssize_t)self.GetOutputSize(), (py::ssize_t)self.GetInputSize()}, self.GetWeights(), py::cast(&self)); })
            .def_property_readonly("batch_size", &LayerT::GetBatchSize)
            .def_property_readonly("input_size", &LayerT::GetInputSize)
            .def_property_readonly("output_size", &LayerT::GetOutputSize)
            .def("__repr__", [className](const LayerT &self)
                 { return "<" + std::string(className) + " in=" + std::to_string(self.GetInputSize()) + ", out=" + std::to_string(self.GetOutputSize()) + ", batch=" + std::to_string(self.GetBatchSize()) + ">"; });
    }

    template <typename NetT>
    void BindCommonPCNetwork(py::class_<NetT> &cls, const char *className)
    {
        cls.def(py::init<int>(), py::arg("batch_size"), "Construct a network with a fixed batch size.")
            .def("randomize_weights", [](NetT &self)
                 {
                std::random_device rd;
                std::mt19937 rng(rd());
                self.RandomizeWeights(rng); }, "Initialize every layer's weights randomly.")
            .def("clamp_input", [](NetT &self, py::array_t<float, py::array::c_style | py::array::forcecast> input)
                 {
                auto buf = input.request();
                std::vector<float> values(static_cast<float *>(buf.ptr), static_cast<float *>(buf.ptr) + buf.size);
                self.Clamp(values); }, py::arg("input"), "Clamp the first (input) layer to the supplied, flattened batch.")
            .def("compile", &NetT::Compile, "Compiles all layers into a contiguous block.")
            .def("calculate_state", &NetT::CalculateState, "Compute the total network energy.")
            .def("update_state", &NetT::UpdateState, "Run one inference step.")
            .def("update_weights", &NetT::UpdateWeights, "Apply weight updates to every layer.")
            .def("reset_state", &NetT::ResetState, "Resets the beliefs 'z' on every layer.")
            .def("get_terminal_layer", &NetT::GetTerminalLayer, py::return_value_policy::reference, "Return the last layer.")
            .def_property_readonly("batch_size", &NetT::GetBatchSize)
            .def_property_readonly("layers", [](NetT &self)
                                   {
                py::list result;
                for (auto *layer : self.GetLayers()) result.append(py::cast(layer, py::return_value_policy::reference));
                return result; }, "List of layer objects owned by the network.")
            .def("__len__", [](const NetT &self)
                 { return self.GetLayers().size(); })
            .def("__getitem__", [](NetT &self, std::ptrdiff_t index)
                 {
                auto &layers = self.GetLayers();
                if (index < 0) index += static_cast<std::ptrdiff_t>(layers.size());
                if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size())) throw py::index_error();
                return layers[index]; }, py::return_value_policy::reference_internal)
            .def("__repr__", [className](const NetT &self)
                 { return "<" + std::string(className) + " layers=" + std::to_string(self.GetLayers().size()) + " batch_size=" + std::to_string(self.GetBatchSize()) + ">"; });
    }

} // namespace

// ============================================================================
// Layer Bindings
// ============================================================================
void bind_layers(py::module_ &m)
{
    py::class_<Deep::Layer>(m, "Layer", "Abstract base class for all Predictive Coding layers.");

    auto discLayerCls = py::class_<Deep::DiscriminativePCLayer, Deep::Layer>(m, "DiscriminativePCLayer", "Predictive Coding layer.")
                            .def(py::init([](int size, int next_size, int batch_size, float learning_rate, float inference_rate, float precision_rate, float lmbda, const std::string &activation, const std::string &activation_deriv)
                                          { return std::make_unique<Deep::DiscriminativePCLayer>(size, next_size, batch_size, learning_rate, inference_rate, precision_rate, lmbda, resolveAct(activation), resolveDAct(activation_deriv)); }),
                                 py::arg("size"), py::arg("next_size"), py::arg("batch_size") = 1, py::arg("learning_rate") = 1e-6f, py::arg("inference_rate") = 0.01f, py::arg("precision_rate") = 0.01f, py::arg("lmbda") = 1e-2f, py::arg("activation") = "relu", py::arg("activation_deriv") = "drelu");
    BindCommonPCLayer<Deep::DiscriminativePCLayer>(discLayerCls, "DiscriminativePCLayer");
    discLayerCls.def("update_precision", &Deep::DiscriminativePCLayer::UpdatePrecision).def("set_precision_rate", &Deep::DiscriminativePCLayer::SetPrecisionRate, py::arg("pr"));

    auto simpleLayerCls = py::class_<Deep::SimplePCLayer, Deep::Layer>(m, "SimplePCLayer", "Predictive Coding layer without precision weighting.")
                              .def(py::init([](int size, int next_size, int batch_size, float learning_rate, float inference_rate, float lmbda, const std::string &activation, const std::string &activation_deriv)
                                            { return std::make_unique<Deep::SimplePCLayer>(size, next_size, batch_size, learning_rate, inference_rate, lmbda, resolveAct(activation), resolveDAct(activation_deriv)); }),
                                   py::arg("size"), py::arg("next_size"), py::arg("batch_size") = 1, py::arg("learning_rate") = 1e-6f, py::arg("inference_rate") = 0.01f, py::arg("lmbda") = 1e-2f, py::arg("activation") = "relu", py::arg("activation_deriv") = "drelu");
    BindCommonPCLayer<Deep::SimplePCLayer>(simpleLayerCls, "SimplePCLayer");
    simpleLayerCls.def_property_readonly("biases", [](Deep::SimplePCLayer &self)
                                         { return py::array_t<float>((py::ssize_t)self.GetOutputSize(), self.GetBiases(), py::cast(&self)); });

    py::class_<Deep::RBLayer, Deep::Layer>(m, "RBLayer", "Restricted Boltzmann-style Predictive Coding layer.")
        .def(py::init([](size_t in_size, size_t out_size, float var, float var_td, float k1, float k2, float lambda, float alpha, size_t batch_size, int step_size, const std::string &activation, const std::string &activation_deriv)
                      { return std::make_unique<Deep::RBLayer>(in_size, out_size, var, var_td, k1, k2, lambda, alpha, batch_size, step_size, resolveAct(activation), resolveDAct(activation_deriv)); }),
             py::arg("in_size"), py::arg("out_size"), py::arg("var") = 1.0f, py::arg("var_td") = 10.0f, py::arg("k1") = 1e-3f, py::arg("k2") = 1e-5f, py::arg("lambda") = 1e-6f, py::arg("alpha") = 1.0f, py::arg("batch_size") = 64, py::arg("step_size") = 30, py::arg("activation") = "relu", py::arg("activation_deriv") = "drelu")
        .def("run_prediction", [](Deep::RBLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> input, size_t current_batch_size)
             { self.RunPrediction(input.data(), current_batch_size); }, py::arg("input"), py::arg("current_batch_size"))
        .def("run_inference_step", [](Deep::RBLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> bottom_up, py::array_t<float, py::array::c_style | py::array::forcecast> top_down, size_t current_batch_size)
             { self.RunInferenceStep(bottom_up.data(), top_down.data(), current_batch_size); }, py::arg("bottom_up"), py::arg("top_down"), py::arg("current_batch_size"))
        .def("calc_error", [](Deep::RBLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> bottom_up, py::array_t<float, py::array::c_style | py::array::forcecast> top_down, size_t current_batch_size)
             { self.CalcError(bottom_up.data(), top_down.data(), current_batch_size); }, py::arg("bottom_up"), py::arg("top_down"), py::arg("current_batch_size"))
        .def("update_beliefs", [](Deep::RBLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> bottom_up, py::array_t<float, py::array::c_style | py::array::forcecast> top_down, size_t current_batch_size)
             { self.UpdateBeliefs(bottom_up.data(), top_down.data(), current_batch_size); }, py::arg("bottom_up"), py::arg("top_down"), py::arg("current_batch_size"))
        .def("calculate_state", &Deep::RBLayer::CalculateState)
        .def("update_state", &Deep::RBLayer::UpdateState)
        .def("update_weights", py::overload_cast<>(&Deep::RBLayer::UpdateWeights))
        .def("update_weights_batch", py::overload_cast<size_t>(&Deep::RBLayer::UpdateWeights), py::arg("current_batch_size"))
        .def("flush", &Deep::RBLayer::Flush)
        .def("attach", [](Deep::RBLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> arena)
             { self.Attach(arena.mutable_data()); }, py::arg("arena"))
        .def_property_readonly("beliefs", [](Deep::RBLayer &self)
                               { return py::array_t<float>({(py::ssize_t)self.GetBatchSize(), (py::ssize_t)self.GetOutputSize()}, self.GetBeliefs(), py::cast(&self)); })
        .def_property_readonly("errors", [](Deep::RBLayer &self)
                               { return py::array_t<float>({(py::ssize_t)self.GetBatchSize(), (py::ssize_t)self.GetInputSize()}, self.GetErrors(), py::cast(&self)); })
        .def_property_readonly("weights", [](Deep::RBLayer &self)
                               { return py::array_t<float>({(py::ssize_t)self.GetOutputSize(), (py::ssize_t)self.GetInputSize()}, self.GetWeights(), py::cast(&self)); })
        .def_property_readonly("batch_size", &Deep::RBLayer::GetBatchSize)
        .def_property_readonly("input_size", &Deep::RBLayer::GetInputSize)
        .def_property_readonly("output_size", &Deep::RBLayer::GetOutputSize)
        .def("total_size", &Deep::RBLayer::GetTotalSize)
        .def("__repr__", [](const Deep::RBLayer &self)
             { return "<RBLayer in=" + std::to_string(self.GetInputSize()) + ", out=" + std::to_string(self.GetOutputSize()) + ", batch=" + std::to_string(self.GetBatchSize()) + ">"; });

    py::class_<Deep::ConvPCLayer, Deep::Layer>(m, "ConvPCLayer", "Convolutional Predictive Coding layer.")
        .def(py::init([](int in_channels, int out_channels, int in_height, int in_width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int batch_size, float learning_rate, float inference_rate, float precision_rate, float lmbda, const std::string &activation, const std::string &activation_deriv)
                      { return std::make_unique<Deep::ConvPCLayer>(in_channels, out_channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, batch_size, learning_rate, inference_rate, precision_rate, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("in_height"), py::arg("in_width"), py::arg("kernel_h"), py::arg("kernel_w"), py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("batch_size") = 1, py::arg("learning_rate") = 1e-6f, py::arg("inference_rate") = 0.1f, py::arg("precision_rate") = 0.0f, py::arg("lmbda") = 1e-2f, py::arg("activation") = "relu", py::arg("activation_deriv") = "drelu")
        .def("calculate_state", &Deep::ConvPCLayer::CalculateState)
        .def("update_state", &Deep::ConvPCLayer::UpdateState)
        .def("update_weights", &Deep::ConvPCLayer::UpdateWeights)
        .def("update_precision", &Deep::ConvPCLayer::UpdatePrecision)
        .def("flush", &Deep::ConvPCLayer::Flush)
        .def("reset_state", &Deep::ConvPCLayer::ResetState)
        .def("resync_log_precision", &Deep::ConvPCLayer::ResyncLogPrecision)
        .def("clamp_state", [](Deep::ConvPCLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> input)
             {
        auto buf = input.request();
        std::vector<float> values(static_cast<float *>(buf.ptr), static_cast<float *>(buf.ptr) + buf.size);
        self.ClampState(values); }, py::arg("input"))
        .def("unclamp_state", &Deep::ConvPCLayer::UnclampState)
        .def("randomize_weights", [](Deep::ConvPCLayer &self)
             {
        std::random_device rd;
        std::mt19937 rng(rd());
        self.RandomizeWeights(rng); })
        .def("set_layer_above", &Deep::ConvPCLayer::SetLayerAbove, py::return_value_policy::reference)
        .def("set_layer_below", &Deep::ConvPCLayer::SetLayerBelow, py::return_value_policy::reference)
        .def("set_learning_rate", &Deep::ConvPCLayer::SetLearningRate, py::arg("lr"))
        .def("set_inference_rate", &Deep::ConvPCLayer::SetInferenceRate, py::arg("ir"))
        .def("set_precision_rate", &Deep::ConvPCLayer::SetPrecisionRate, py::arg("pr"))
        .def("set_lambda", &Deep::ConvPCLayer::SetLambda, py::arg("l"))
        .def_property_readonly("beliefs", [](Deep::ConvPCLayer &self)
                               { return py::array_t<float>({(py::ssize_t)self.GetBatchSize(), (py::ssize_t)self.GetInChannels(), (py::ssize_t)self.GetInHeight(), (py::ssize_t)self.GetInWidth()}, self.GetBeliefs(), py::cast(&self)); })
        .def_property_readonly("errors", [](Deep::ConvPCLayer &self)
                               { return py::array_t<float>({(py::ssize_t)self.GetBatchSize(), (py::ssize_t)self.GetInChannels(), (py::ssize_t)self.GetInHeight(), (py::ssize_t)self.GetInWidth()}, self.GetErrors(), py::cast(&self)); })
        .def_property_readonly("weights", [](Deep::ConvPCLayer &self)
                               { return py::array_t<float>({(py::ssize_t)self.GetOutChannels(), (py::ssize_t)self.GetInChannels(), (py::ssize_t)self.GetKernelH(), (py::ssize_t)self.GetKernelW()}, self.GetWeights(), py::cast(&self)); })
        .def_property_readonly("biases", [](Deep::ConvPCLayer &self)
                               { return py::array_t<float>({(py::ssize_t)self.GetOutChannels()}, self.GetBiases(), py::cast(&self)); })
        .def_property_readonly("batch_size", &Deep::ConvPCLayer::GetBatchSize)
        .def_property_readonly("in_channels", &Deep::ConvPCLayer::GetInChannels)
        .def_property_readonly("out_channels", &Deep::ConvPCLayer::GetOutChannels)
        .def_property_readonly("in_height", &Deep::ConvPCLayer::GetInHeight)
        .def_property_readonly("in_width", &Deep::ConvPCLayer::GetInWidth)
        .def_property_readonly("out_height", &Deep::ConvPCLayer::GetOutHeight)
        .def_property_readonly("out_width", &Deep::ConvPCLayer::GetOutWidth)
        .def_property_readonly("kernel_h", &Deep::ConvPCLayer::GetKernelH)
        .def_property_readonly("kernel_w", &Deep::ConvPCLayer::GetKernelW)
        .def_property_readonly("input_size", &Deep::ConvPCLayer::GetInputSize)
        .def_property_readonly("output_size", &Deep::ConvPCLayer::GetOutputSize)
        .def("__repr__", [](const Deep::ConvPCLayer &self)
             { return "<ConvPCLayer in=(" + std::to_string(self.GetInChannels()) + "," + std::to_string(self.GetInHeight()) + "," + std::to_string(self.GetInWidth()) + ") out_channels=" + std::to_string(self.GetOutChannels()) + " kernel=(" + std::to_string(self.GetKernelH()) + "," + std::to_string(self.GetKernelW()) + ") batch=" + std::to_string(self.GetBatchSize()) + ">"; });
}

// ============================================================================
// Network Bindings
// ============================================================================
void bind_networks(py::module_ &m)
{
    auto discNetCls = py::class_<Deep::DiscriminativePCNetwork>(m, "DiscriminativePCNetwork", "Predictive Coding Network.");
    BindCommonPCNetwork<Deep::DiscriminativePCNetwork>(discNetCls, "DiscriminativePCNetwork");
    discNetCls.def(py::init<>(), "Construct a network with automatic batch-size detection.")
        .def("add_layer", [](Deep::DiscriminativePCNetwork &self, int size, int next_size, float lr, float ir, float pr, float lmbda, const std::string &activation, const std::string &activation_deriv)
             { self.AddLayer(size, next_size, lr, ir, pr, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, py::arg("size"), py::arg("next_size"), py::arg("lr") = 1e-6f, py::arg("ir") = 0.1f, py::arg("pr") = 0.01f, py::arg("lmbda") = 1e-2f, py::arg("activation") = "relu", py::arg("activation_deriv") = "drelu", "Add a layer to the network.")
        .def("set_inference_rate", &Deep::DiscriminativePCNetwork::SetInferenceRate, "Sets the inference rate of each layer.", py::arg("ir"))
        .def("set_learning_rate", &Deep::DiscriminativePCNetwork::SetLearningRate, "Sets the learning rate of each layer.", py::arg("lr"))
        .def("set_precision_rate", &Deep::DiscriminativePCNetwork::SetPrecisionRate, "Sets the precision rate of each layer.", py::arg("pr"))
        .def("set_lambda", &Deep::DiscriminativePCNetwork::SetLambda, "Sets lambda of each layer.", py::arg("l"))
        .def("save", &Deep::DiscriminativePCNetwork::Save, "Saves the network architecture and weights to a structured directory.", py::arg("dir_path"))
        .def("load", &Deep::DiscriminativePCNetwork::Load, "Loads network weights from a structured directory into the compiled MemoryArena.", py::arg("dir_path"))
        .def("update_precision", &Deep::DiscriminativePCNetwork::UpdatePrecision, "Apply precision updates to every layer.");

    auto simpleNetCls = py::class_<Deep::SimplePCNetwork>(m, "SimplePCNetwork", "Predictive Coding Network built from SimplePCLayers.");
    BindCommonPCNetwork<Deep::SimplePCNetwork>(simpleNetCls, "SimplePCNetwork");
    simpleNetCls.def("add_layer", [](Deep::SimplePCNetwork &self, int size, int next_size, float lr, float ir, float lmbda, const std::string &activation, const std::string &activation_deriv)
                     { self.AddLayer(size, next_size, lr, ir, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, py::arg("size"), py::arg("next_size"), py::arg("lr") = 1e-6f, py::arg("ir") = 0.1f, py::arg("lmbda") = 1e-2f, py::arg("activation") = "relu", py::arg("activation_deriv") = "drelu", "Add a layer to the network.")
        .def("set_optimizer", [](Deep::SimplePCNetwork &self, const std::string &opt)
             {
            if (opt == "ADAM") self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW") self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else self.SetOptimizer(Deep::OptimizerType::SGD); }, py::arg("optimizer"), "Sets the optimizer: ADAM, ADAMW, or SGD.");

    py::class_<Deep::ConvPCNetwork>(m, "ConvPCNetwork", "Convolutional Predictive Coding Network.")
        .def(py::init<int>(), py::arg("batch_size"), "Construct a network with a fixed batch size.")
        .def("add_layer", [](Deep::ConvPCNetwork &self, int in_channels, int out_channels, int in_height, int in_width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, float lr, float ir, float pr, float lmbda, const std::string &activation, const std::string &activation_deriv)
             { self.AddLayer(in_channels, out_channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, lr, ir, pr, lmbda, resolveActEnum(activation), resolveActEnum(activation_deriv)); }, py::arg("in_channels"), py::arg("out_channels"), py::arg("in_height"), py::arg("in_width"), py::arg("kernel_h"), py::arg("kernel_w"), py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("lr") = 1e-6f, py::arg("ir") = 0.1f, py::arg("pr") = 0.0f, py::arg("lmbda") = 1e-2f, py::arg("activation") = "relu", py::arg("activation_deriv") = "drelu")
        .def("compile", &Deep::ConvPCNetwork::Compile, "Compiles all layers into a contiguous block.")
        .def("randomize_weights", [](Deep::ConvPCNetwork &self)
             {
            std::random_device rd; std::mt19937 rng(rd()); self.RandomizeWeights(rng); })
        .def("clamp_input", [](Deep::ConvPCNetwork &self, py::array_t<float, py::array::c_style | py::array::forcecast> input)
             {
            auto buf = input.request();
            std::vector<float> values(static_cast<float *>(buf.ptr), static_cast<float *>(buf.ptr) + buf.size);
            self.Clamp(values); }, py::arg("input"))
        .def("calculate_state", &Deep::ConvPCNetwork::CalculateState)
        .def("update_state", &Deep::ConvPCNetwork::UpdateState)
        .def("update_weights", &Deep::ConvPCNetwork::UpdateWeights)
        .def("update_precision", &Deep::ConvPCNetwork::UpdatePrecision)
        .def("reset_state", &Deep::ConvPCNetwork::ResetState)
        .def("train_step", [](Deep::ConvPCNetwork &self, py::array_t<float, py::array::c_style | py::array::forcecast> x, py::array_t<float, py::array::c_style | py::array::forcecast> y, int steps)
             {
            auto xbuf = x.request(); auto ybuf = y.request();
            std::vector<float> xvec(static_cast<float *>(xbuf.ptr), static_cast<float *>(xbuf.ptr) + xbuf.size);
            std::vector<float> yvec(static_cast<float *>(ybuf.ptr), static_cast<float *>(ybuf.ptr) + ybuf.size);
            return self.TrainStep(xvec, yvec, steps); }, py::arg("x"), py::arg("y"), py::arg("steps"))
        .def("predict", [](Deep::ConvPCNetwork &self, py::array_t<float, py::array::c_style | py::array::forcecast> x, int steps)
             {
            auto xbuf = x.request();
            std::vector<float> xvec(static_cast<float *>(xbuf.ptr), static_cast<float *>(xbuf.ptr) + xbuf.size);
            std::vector<float> result = self.Predict(xvec, steps);
            Deep::ConvPCLayer *terminal = self.GetTerminalLayer();
            py::array_t<float> out({(py::ssize_t)terminal->GetBatchSize(), (py::ssize_t)terminal->GetInputSize()});
            std::memcpy(out.mutable_data(), result.data(), result.size() * sizeof(float));
            return out; }, py::arg("x"), py::arg("steps"))
        .def("get_terminal_layer", &Deep::ConvPCNetwork::GetTerminalLayer, py::return_value_policy::reference)
        .def_property_readonly("batch_size", &Deep::ConvPCNetwork::GetBatchSize)
        .def_property_readonly("layers", [](Deep::ConvPCNetwork &self)
                               {
            py::list result;
            for (auto *layer : self.GetLayers()) result.append(py::cast(layer, py::return_value_policy::reference));
            return result; })
        .def("__len__", [](const Deep::ConvPCNetwork &self)
             { return self.GetLayers().size(); })
        .def("__getitem__", [](Deep::ConvPCNetwork &self, std::ptrdiff_t index)
             {
            auto &layers = self.GetLayers();
            if (index < 0) index += static_cast<std::ptrdiff_t>(layers.size());
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size())) throw py::index_error();
            return layers[index]; }, py::return_value_policy::reference_internal)
        .def("__repr__", [](const Deep::ConvPCNetwork &self)
             { return "<ConvPCNetwork layers=" + std::to_string(self.GetLayers().size()) + " batch_size=" + std::to_string(self.GetBatchSize()) + ">"; });
}

// ============================================================================
// Utilities Bindings
// ============================================================================
void bind_utilities(py::module_ &m)
{
    py::class_<Deep::StreamAlignedBatcher>(m, "StreamAlignedBatcher", "Builds batches with a fixed number of examples per class.")
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> X, py::array_t<float, py::array::c_style | py::array::forcecast> Y, py::array_t<int, py::array::c_style | py::array::forcecast> labels, size_t x_stride, size_t y_stride, int num_classes, int per_class, unsigned seed)
                      {
            auto xbuf = X.request();
            auto ybuf = Y.request();
            auto lbuf = labels.request();
            size_t num_samples = (size_t)xbuf.shape[0];
            return std::make_unique<Deep::StreamAlignedBatcher>(static_cast<float *>(xbuf.ptr), static_cast<float *>(ybuf.ptr), static_cast<int *>(lbuf.ptr), num_samples, x_stride, y_stride, num_classes, per_class, seed); }),
             py::arg("X"), py::arg("Y"), py::arg("labels"), py::arg("x_stride"), py::arg("y_stride"), py::arg("num_classes"), py::arg("per_class"), py::arg("seed") = 42, py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("num_batches_per_epoch", &Deep::StreamAlignedBatcher::NumBatchesPerEpoch)
        .def_property_readonly("batch_size", &Deep::StreamAlignedBatcher::GetBatchSize)
        .def("get_batch", [](Deep::StreamAlignedBatcher &self)
             {
            int bsz = self.GetBatchSize();
            py::array_t<float> X_out({(py::ssize_t)bsz, (py::ssize_t)self.GetXStride()});
            py::array_t<float> Y_out({(py::ssize_t)bsz, (py::ssize_t)self.GetYStride()});
            py::array_t<int> labels_out(bsz);
            self.GetBatch(X_out.mutable_data(), Y_out.mutable_data(), labels_out.mutable_data());
            return py::make_tuple(X_out, Y_out, labels_out); });

    m.def("get_l2_cache_bytes", &Deep::GetL2CacheBytes);
    m.def("auto_batch_size", &Deep::AutoBatchSize);
    m.def("dynamic_thread", &Deep::DynamicThread, py::arg("batch_size"));

    m.def("omp_max_threads", []()
          { return omp_get_max_threads(); });
    m.def("omp_num_procs", []()
          { return omp_get_num_procs(); });

    m.def("relu", [](py::array_t<float> x)
          { Deep::relu(x.mutable_data(), x.size()); });
    m.def("drelu", [](py::array_t<float> x)
          { Deep::dRelu(x.mutable_data(), x.size()); });
    m.def("tanh", [](py::array_t<float> x)
          { Deep::tanh(x.mutable_data(), x.size()); });
    m.def("dtanh", [](py::array_t<float> x)
          { Deep::dTanh(x.mutable_data(), x.size()); });
    m.def("sigmoid", [](py::array_t<float> x)
          { Deep::sigmoid(x.mutable_data(), x.size()); });
    m.def("dsigmoid", [](py::array_t<float> x)
          { Deep::dSigmoid(x.mutable_data(), x.size()); });
}

// ============================================================================
// Main Module Entry
// ============================================================================
PYBIND11_MODULE(pydeepity, m)
{
    m.doc() = "Deepity: A high-performance Predictive Coding library.";

    bind_layers(m);
    bind_networks(m);
    bind_utilities(m);
}
