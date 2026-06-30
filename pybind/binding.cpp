#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <memory>

#include "Layer.h"
#include "PCLayer.h"
#include "PCNetwork.h"
#include "Activations.h"

namespace py = pybind11;

static void (*resolveAct(const std::string &act))(float *, size_t)
{
    if (act == "tanh")    return Deep::tanh;
    if (act == "dtanh") return Deep::dTanh;
    if (act == "sigmoid") return Deep::sigmoid;
    if (act == "drelu") return Deep::dRelu;
    if (act == "relu") return Deep::relu;
    return Deep::relu;
}

PYBIND11_MODULE(deepity, m)
{
    m.doc() = "Deepity: A high-performance Predictive Coding Network library.";

    // --- Network ---
    py::class_<Deep::PCNetwork>(m, "PCNetwork",
        "A predictive coding network. Add layers, then call inference methods.")
        .def(py::init<int>(), py::arg("batch_size"))
        .def("add_layer", [](Deep::PCNetwork &self, int size, int nextSize, float lr,
                              const std::string &act, const std::string &dAct)
             {
                 self.AddLayer(size, nextSize, lr, resolveAct(act), resolveAct(dAct));
             },
             py::arg("size"), py::arg("next_size"),
             py::arg("lr") = 1e-6f,
             py::arg("act") = "relu",
             py::arg("dact") = "relu",
             "Add a PCLayer to the network.")
        .def("randomize_weights", [](Deep::PCNetwork &self)
             {
                 std::mt19937 rng(std::random_device{}());
                 self.RandomizeWeights(rng);
             }, "Randomize all layer weights.")
        .def("clamp", [](Deep::PCNetwork &self, py::array_t<float> x)
             {
                 std::vector<float> v(x.data(), x.data() + x.size());
                 self.Clamp(v);
             }, py::arg("x"), "Clamp input to the bottom layer.")
        .def("calculate_state", &Deep::PCNetwork::CalculateState,
             "Calculate total energy across all layers.")
        .def("update_state", &Deep::PCNetwork::UpdateState,
             "Update latent states across all layers.")
        .def("update_weights", &Deep::PCNetwork::UpdateWeights,
             "Update weights across all layers.");

    // --- Activations ---
    m.def("relu",    [](py::array_t<float> x) { Deep::relu(x.mutable_data(), x.size()); });
    m.def("tanh",    [](py::array_t<float> x) { Deep::tanh(x.mutable_data(), x.size()); });
    m.def("sigmoid", [](py::array_t<float> x) { Deep::sigmoid(x.mutable_data(), x.size()); });
}