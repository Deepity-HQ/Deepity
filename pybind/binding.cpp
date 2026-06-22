#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "PCNNetwork.h"
#include "PCNLayer.h"
#include "Activations.h"
#include "Optimize.h"

namespace py = pybind11;

PYBIND11_MODULE(deepity, m)
{
     m.doc() = "Deepity: A high-performance Predictive Coding Network library with AVX2/AVX-512 SIMD optimizations.";
     // NETWORK
     py::class_<Deep::PCNetwork>(m, "PCNetwork", "A bidirectional predictive coding network. Call compile() after adding all layers before running inference or generation.")
         .def(py::init<>())
         .def("add_layer", [](Deep::PCNetwork &self, size_t inSize, size_t outSize, float lr, float ir, int stepSize, const std::string &act)
              {
ActivationFn fn;
if (act == "tanh")         fn = Deep::tanh;
else if (act == "sigmoid") fn = Deep::sigmoid;
else                       fn = Deep::relu;
self.AddLayer(inSize, outSize, lr, ir, stepSize, fn); }, py::arg("in_size"), py::arg("out_size"), py::arg("lr") = 1e-4f, py::arg("ir") = 1e-4f, py::arg("step_size") = 30, py::arg("act") = "relu", "Add a layer to the network.\n\n"
                                                                                                                                                              "Args:\n"
                                                                                                                                                              "    in_size: Number of input features.\n"
                                                                                                                                                              "    out_size: Number of output features.\n"
                                                                                                                                                              "    lr: Learning rate for weight updates. Default 1e-4.\n"
                                                                                                                                                              "    ir: Inference rate for belief updates. Default 1e-4.\n"
                                                                                                                                                              "    step_size: Number of inference steps per input. Default 30.\n"
                                                                                                                                                              "    act: Activation function. One of 'relu', 'tanh', 'sigmoid'. Default 'relu'.")
         .def("add_layer", [](Deep::PCNetwork &self, Deep::PCLayer &layer)
              { self.AddLayer(std::move(layer)); }, py::arg("layer"), "Add a pre-constructed PCLayer to the network.")
         .def("compile", &Deep::PCNetwork::Compile, "Allocate a contiguous memory arena for all layers. Must be called before inference or generation.")
         .def("inference_step", [](Deep::PCNetwork &self, py::array_t<float> x)
              { self.InferenceStep(x.mutable_data()); }, py::arg("x"), "Feed one input bottom-up. Triggers a full cascade once a batch is filled.")
         .def("generation_step", [](Deep::PCNetwork &self, py::array_t<float> x)
              { self.GenerationStep(x.mutable_data()); }, py::arg("x"), "Feed one target top-down. Triggers a full cascade once a batch is filled.")
         .def("flush_inference", &Deep::PCNetwork::FlushInference, "Flush any remaining partial inference batch.")
         .def("flush_generation", &Deep::PCNetwork::FlushGeneration, "Flush any remaining partial generation batch.")
         .def("total_energy", &Deep::PCNetwork::GetTotalEnergy, "Return the total prediction error energy across all layers (0.5 * sum of squared errors).")
         .def("save", &Deep::PCNetwork::SaveModel, py::arg("path"), "Save the network weights to a binary file.")
         .def("load", &Deep::PCNetwork::LoadModel, py::arg("path"), "Load network weights from a binary file.");
     // LAYER
     py::class_<Deep::PCLayer>(m, "PCLayer", "A single predictive coding layer with SIMD-optimized inference and weight updates.")
         .def(py::init([](size_t inSize, size_t outSize, float lr, float ir, int stepSize, const std::string &act)
                       {
    ActivationFn fn;
    if (act == "tanh")         fn = Deep::tanh;
    else if (act == "sigmoid") fn = Deep::sigmoid;
    else                       fn = Deep::relu;
    return Deep::PCLayer(inSize, outSize, lr, ir, stepSize, fn); }),
              py::arg("in_size"), py::arg("out_size"),
              py::arg("lr") = 1e-4f, py::arg("ir") = 1e-4f,
              py::arg("step_size") = 30, py::arg("act") = "relu",
              "Initialize a PCLayer.\n\n"
              "Args:\n"
              "    in_size: Number of input features.\n"
              "    out_size: Number of output features.\n"
              "    lr: Learning rate. Default 1e-4.\n"
              "    ir: Inference rate. Default 1e-4.\n"
              "    step_size: Inference steps per input. Default 30.\n"
              "    act: Activation function. One of 'relu', 'tanh', 'sigmoid'. Default 'relu'.")
         .def("calc_prediction", &Deep::PCLayer::CalcPrediction,
              "Compute prediction p = z @ W.T.")
         .def("calc_step_error", [](Deep::PCLayer &self, py::array_t<float> x)
              { self.CalcStepError(x.data(), self.GetBatchSize() ); }, py::arg("x"), "Compute prediction error err = x - p.")
         .def("update_beliefs", &Deep::PCLayer::UpdateBeliefs, "Update beliefs z += ir * err @ W, then apply activation.")
         .def("update_weights", &Deep::PCLayer::UpdateWeights, "Update weights W += lr * err.T @ z.")
         .def("run_prediction", [](Deep::PCLayer &self, py::array_t<float> x)
              { self.RunPrediction(x.data()); }, py::arg("x"), "Queue one input sample. Triggers a full batched prediction once the batch is filled.")
         .def("run_prediction_batched", [](Deep::PCLayer &self, py::array_t<float> x, size_t batchSize)
              { self.RunBatchedPrediction(x.data(), batchSize ); }, py::arg("x"), py::arg("batch_size"), "Run a full batched prediction immediately on the provided buffer.")
         .def("flush", &Deep::PCLayer::Flush, "Flush any remaining partial batch.")
         .def("attach", [](Deep::PCLayer &self, py::array_t<float> x)
              { self.Attach(x.mutable_data()); }, py::arg("ptr"), "Attach the layer to an external 64-byte aligned memory buffer.")
         .def("get_prediction", [](Deep::PCLayer &self)
              {
size_t n = self.GetBatchSize() * self.GetInputSize();
return py::array_t<float>({(py::ssize_t)n}, self.GetPrediction()); }, "Return the current prediction buffer as a NumPy array (no copy).")
         .def("get_inference_err", [](Deep::PCLayer &self)
              {
size_t n = self.GetBatchSize() * self.GetInputSize();
return py::array_t<float>({(py::ssize_t)n}, self.GetInferenceError()); }, "Return the current error buffer as a NumPy array (no copy).")
         .def("get_beliefs", [](Deep::PCLayer &self)
              {
size_t n = self.GetBatchSize() * self.GetOutputSize();
return py::array_t<float>({(py::ssize_t)n}, self.GetBeliefs()); }, "Return the current belief buffer z as a NumPy array (no copy).")
         .def("get_weights", [](Deep::PCLayer &self)
              {
size_t n = self.GetInputSize() * self.GetOutputSize();
return py::array_t<float>({(py::ssize_t)n}, self.GetWeights()); }, "Return the weight matrix W as a NumPy array (no copy).")
         .def("get_lr", &Deep::PCLayer::GetLR, "Return the learning rate.")
         .def("get_ir", &Deep::PCLayer::GetIR, "Return the inference rate.")
         .def("get_input_size", &Deep::PCLayer::GetInputSize, "Return the input dimension.")
         .def("get_output_size", &Deep::PCLayer::GetOutputSize, "Return the output dimension.")
         .def("get_batch_size", &Deep::PCLayer::GetBatchSize, "Return the batch size.")
         .def("get_size", &Deep::PCLayer::GetTotalSize, "Return the total number of floats in the layer's memory buffer.");
     // OTHER
     m.def("get_L2_cache_bytes", &Deep::GetL2CacheBytes,
           "Return the L2 cache size in bytes for the current CPU.");
     m.def("auto_batch_size", &Deep::AutoBatchSize,
           py::arg("element_size"), py::arg("input_size"),
           "Compute an optimal batch size based on L2 cache size and input dimensions.");
     m.def("relu", [](py::array_t<float> x)
           { Deep::relu(x.mutable_data(), x.size()); }, py::arg("x"), "Apply ReLU in-place to a NumPy float32 array.");
     m.def("tanh", [](py::array_t<float> x)
           { Deep::tanh(x.mutable_data(), x.size()); }, py::arg("x"), "Apply TanH in-place to a NumPy float32 array using a Pade approximation.");
     m.def("sigmoid", [](py::array_t<float> x)
           { Deep::sigmoid(x.mutable_data(), x.size()); }, py::arg("x"), "Apply Sigmoid in-place to a NumPy float32 array using the Elliot approximation.");
}