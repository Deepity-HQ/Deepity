#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <memory>

#include "Layer.h"
#include "PCLayer.h"
#include "RBLayer.h"
#include "PCNetwork.h"
#include "Activations.h"

namespace py = pybind11;

static void (*resolveAct(const std::string &act))(float *, size_t)
{
    if (act == "tanh")
        return Deep::tanh;
    if (act == "dtanh")
        return Deep::dTanh;
    if (act == "sigmoid")
        return Deep::sigmoid;
    if (act == "drelu")
        return Deep::dRelu;
    if (act == "relu")
        return Deep::relu;
    return Deep::relu;
}

PYBIND11_MODULE(deepity, m)
{
    m.doc() = "Deepity: A high-performance Predictive Coding library.";

    py::class_<Deep::Layer>(m, "Layer",
                            R"pbdoc(
    Abstract base class for all Predictive Coding layers.
)pbdoc");

    py::class_<Deep::PCNetwork>(m, "PCNetwork",
                                R"pbdoc(
        Predictive Coding Network.

        A network composed of one or more PCLayers. Layers are connected
        automatically as they are added.
    )pbdoc")
        .def(py::init<>(),
             "Construct a network with automatic batch-size detection.")

        .def(py::init<int>(),
             py::arg("batch_size"),
             "Construct a network with a fixed batch size.")

        .def(
            "add_layer",
            [](Deep::PCNetwork &self,
               int size,
               int next_size,
               float lr,
               float ir,
               const std::string &activation,
               const std::string &activation_deriv)
            {
                self.AddLayer(
                    size,
                    next_size,
                    lr,
                    ir,
                    resolveAct(activation),
                    resolveAct(activation_deriv));
            },
            py::arg("size"),
            py::arg("next_size"),
            py::arg("lr") = 1e-6f,
            py::arg("ir") = 0.1f,
            py::arg("activation") = "relu",
            py::arg("activation_deriv") = "drelu",
            R"pbdoc(
            Add a layer to the network.
        )pbdoc")

        .def(
            "randomize_weights",
            [](Deep::PCNetwork &self)
            {
                std::random_device rd;
                std::mt19937 rng(rd());
                self.RandomizeWeights(rng);
            },
            "Initialize every layer's weights randomly.")

        .def(
            "clamp_input",
            [](Deep::PCNetwork &self,
               py::array_t<float,
                           py::array::c_style | py::array::forcecast>
                   input)
            {
                auto buf = input.request();

                std::vector<float> values(
                    static_cast<float *>(buf.ptr),
                    static_cast<float *>(buf.ptr) + buf.size);

                self.Clamp(values);
            },
            py::arg("input"),
            "Clamp the first layer to the supplied input.")

        .def(
            "calculate_state",
            &Deep::PCNetwork::CalculateState,
            "Compute the total network energy.")

        .def(
            "update_state",
            &Deep::PCNetwork::UpdateState,
            "Run one inference step.")

        .def(
            "update_weights",
            &Deep::PCNetwork::UpdateWeights,
            "Apply weight updates to every layer.")

        .def_property_readonly(
            "batch_size",
            &Deep::PCNetwork::GetBatchSize)

        .def_property_readonly(
            "layers",
            [](Deep::PCNetwork &self)
            {
                py::list result;

                for (auto *layer : self.GetLayers())
                {
                    result.append(
                        py::cast(
                            layer,
                            py::return_value_policy::reference));
                }

                return result;
            },
            "List of PCLayer objects owned by the network.")

        .def(
            "__len__",
            [](const Deep::PCNetwork &self)
            {
                return self.GetLayers().size();
            })

        .def(
            "__getitem__",
            [](Deep::PCNetwork &self, ssize_t index)
            {
                auto &layers = self.GetLayers();

                if (index < 0)
                    index += static_cast<ssize_t>(layers.size());

                if (index < 0 ||
                    index >= static_cast<ssize_t>(layers.size()))
                    throw py::index_error();

                return layers[index];
            },
            py::return_value_policy::reference_internal)

        .def(
            "__repr__",
            [](const Deep::PCNetwork &self)
            {
                return "<PCNetwork layers=" +
                       std::to_string(self.GetLayers().size()) +
                       " batch_size=" +
                       std::to_string(self.GetBatchSize()) +
                       ">";
            });
    py::class_<Deep::RBLayer, Deep::Layer>(m, "RBLayer",
                                           R"pbdoc(
        Restricted Boltzmann-style Predictive Coding layer.
    )pbdoc")

        .def(py::init([](
                          size_t in_size,
                          size_t out_size,
                          float var,
                          float var_td,
                          float k1,
                          float k2,
                          float lambda,
                          float alpha,
                          size_t batch_size,
                          int step_size,
                          const std::string &activation,
                          const std::string &activation_deriv)
                      { return std::make_unique<Deep::RBLayer>(
                            in_size,
                            out_size,
                            var,
                            var_td,
                            k1,
                            k2,
                            lambda,
                            alpha,
                            batch_size,
                            step_size,
                            resolveAct(activation),
                            resolveAct(activation_deriv)); }),
             py::arg("in_size"),
             py::arg("out_size"),
             py::arg("var") = 1.0f,
             py::arg("var_td") = 10.0f,
             py::arg("k1") = 1e-3f,
             py::arg("k2") = 1e-5f,
             py::arg("lambda") = 1e-6f,
             py::arg("alpha") = 1.0f,
             py::arg("batch_size") = 64,
             py::arg("step_size") = 30,
             py::arg("activation") = "relu",
             py::arg("activation_deriv") = "drelu")

        .def("run_prediction", [](Deep::RBLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> input, size_t current_batch_size)
             { self.RunPrediction(input.data(), current_batch_size); }, py::arg("input"), py::arg("current_batch_size"))

        .def("run_inference_step", [](Deep::RBLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> bottom_up, py::array_t<float, py::array::c_style | py::array::forcecast> top_down, size_t current_batch_size)
             { self.RunInferenceStep(
                   bottom_up.data(),
                   top_down.data(),
                   current_batch_size); }, py::arg("bottom_up"), py::arg("top_down"), py::arg("current_batch_size"))

        .def("calc_error", [](Deep::RBLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> bottom_up, py::array_t<float, py::array::c_style | py::array::forcecast> top_down, size_t current_batch_size)
             { self.CalcError(
                   bottom_up.data(),
                   top_down.data(),
                   current_batch_size); }, py::arg("bottom_up"), py::arg("top_down"), py::arg("current_batch_size"))

        .def("update_beliefs", [](Deep::RBLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> bottom_up, py::array_t<float, py::array::c_style | py::array::forcecast> top_down, size_t current_batch_size)
             { self.UpdateBeliefs(
                   bottom_up.data(),
                   top_down.data(),
                   current_batch_size); }, py::arg("bottom_up"), py::arg("top_down"), py::arg("current_batch_size"))

        .def("calculate_state", &Deep::RBLayer::CalculateState)

        .def("update_state", &Deep::RBLayer::UpdateState)

        .def("update_weights", py::overload_cast<>(&Deep::RBLayer::UpdateWeights))

        .def("update_weights_batch", py::overload_cast<size_t>(&Deep::RBLayer::UpdateWeights), py::arg("current_batch_size"))

        .def("flush", &Deep::RBLayer::Flush)

        .def("attach", [](Deep::RBLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> arena)
             { self.Attach(arena.mutable_data()); }, py::arg("arena"))

        .def_property_readonly("beliefs", [](Deep::RBLayer &self)
                               { return py::array_t<float>(
                                     {(py::ssize_t)self.GetBatchSize(),
                                      (py::ssize_t)self.GetOutputSize()},
                                     self.GetBeliefs(),
                                     py::cast(&self)); })

        .def_property_readonly("errors", [](Deep::RBLayer &self)
                               { return py::array_t<float>(
                                     {(py::ssize_t)self.GetBatchSize(),
                                      (py::ssize_t)self.GetInputSize()},
                                     self.GetErrors(),
                                     py::cast(&self)); })

        .def_property_readonly("weights", [](Deep::RBLayer &self)
                               { return py::array_t<float>(
                                     {(py::ssize_t)self.GetOutputSize(),
                                      (py::ssize_t)self.GetInputSize()},
                                     self.GetWeights(),
                                     py::cast(&self)); })

        .def_property_readonly("batch_size", &Deep::RBLayer::GetBatchSize)

        .def_property_readonly("input_size", &Deep::RBLayer::GetInputSize)

        .def_property_readonly("output_size", &Deep::RBLayer::GetOutputSize)

        .def("total_size", &Deep::RBLayer::GetTotalSize)

        .def("__repr__", [](const Deep::RBLayer &self)
             { return "<RBLayer in=" +
                      std::to_string(self.GetInputSize()) +
                      ", out=" +
                      std::to_string(self.GetOutputSize()) +
                      ", batch=" +
                      std::to_string(self.GetBatchSize()) +
                      ">"; });

    py::class_<Deep::PCLayer, Deep::Layer>(m, "PCLayer",
                                           R"pbdoc(
        Predictive Coding layer.
    )pbdoc")

        .def(py::init([](
                          int size,
                          int next_size,
                          int batch_size,
                          float learning_rate,
                          float inference_rate,
                          const std::string &activation,
                          const std::string &activation_deriv)
                      { return std::make_unique<Deep::PCLayer>(
                            size,
                            next_size,
                            batch_size,
                            learning_rate,
                            inference_rate,
                            resolveAct(activation),
                            resolveAct(activation_deriv)); }),
             py::arg("size"),
             py::arg("next_size"),
             py::arg("batch_size") = 1,
             py::arg("learning_rate") = 1e-6f,
             py::arg("inference_rate") = 0.01f,
             py::arg("activation") = "relu",
             py::arg("activation_deriv") = "drelu")

        .def("calculate_state",
             &Deep::PCLayer::CalculateState)

        .def("update_state",
             &Deep::PCLayer::UpdateState)

        .def("update_weights",
             &Deep::PCLayer::UpdateWeights)

        .def("flush",
             &Deep::PCLayer::Flush)

        .def("clamp_state", [](Deep::PCLayer &self, py::array_t<float, py::array::c_style | py::array::forcecast> input)
             {
            auto buf = input.request();

            std::vector<float> values(
                static_cast<float *>(buf.ptr),
                static_cast<float *>(buf.ptr) + buf.size);

            self.ClampState(values); }, py::arg("input"))

        .def("unclamp_state", &Deep::PCLayer::UnclampState)

        .def("randomize_weights", [](Deep::PCLayer &self)
             {
            std::random_device rd;
            std::mt19937 rng(rd());
            self.RandomizeWeights(rng); })

        .def("set_layer_above", &Deep::PCLayer::SetLayerAbove, py::return_value_policy::reference)

        .def("set_layer_below", &Deep::PCLayer::SetLayerBelow, py::return_value_policy::reference)

        .def_property_readonly("beliefs", [](Deep::PCLayer &self)
                               { return py::array_t<float>(
                                     {(py::ssize_t)self.GetBatchSize(),
                                      (py::ssize_t)self.GetInputSize()},
                                     self.GetBeliefs(),
                                     py::cast(&self)); })

        .def_property_readonly("errors", [](Deep::PCLayer &self)
                               { return py::array_t<float>(
                                     {(py::ssize_t)self.GetBatchSize(),
                                      (py::ssize_t)self.GetInputSize()},
                                     self.GetErrors(),
                                     py::cast(&self)); })

        .def_property_readonly("weights", [](Deep::PCLayer &self)
                               { return py::array_t<float>(
                                     {(py::ssize_t)self.GetInputSize(),
                                      (py::ssize_t)self.GetOutputSize()},
                                     self.GetWeights(),
                                     py::cast(&self)); })

        .def_property_readonly("batch_size", &Deep::PCLayer::GetBatchSize)

        .def_property_readonly("input_size", &Deep::PCLayer::GetInputSize)

        .def_property_readonly("output_size", &Deep::PCLayer::GetOutputSize)

        .def("__repr__", [](const Deep::PCLayer &self)
             { return "<PCLayer in=" +
                      std::to_string(self.GetInputSize()) +
                      ", out=" +
                      std::to_string(self.GetOutputSize()) +
                      ", batch=" +
                      std::to_string(self.GetBatchSize()) +
                      ">"; });

    // --- Activations ---
    m.def("relu", [](py::array_t<float> x)
          { Deep::relu(x.mutable_data(), x.size()); });
    m.def("tanh", [](py::array_t<float> x)
          { Deep::tanh(x.mutable_data(), x.size()); });
    m.def("sigmoid", [](py::array_t<float> x)
          { Deep::sigmoid(x.mutable_data(), x.size()); });
}