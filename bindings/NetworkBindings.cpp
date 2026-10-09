/**
 * @file NetworkBindings.cpp
 * @brief nanobind bindings for every concrete PC network type. Split out of
 * the former monolithic pybinding.cpp, see LayerBindings.cpp /
 * UtilityBindings.cpp for the rest.
 */
#include "BindingHelpers.h"
#include "NetworkBindings.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <random>
#include <string>
#include <vector>

#include <deepity/networks/ConvPCNetwork.h>
#include <deepity/networks/FullConvPCNetwork.h>
#include <deepity/networks/DirectKPPCNetwork.h>
#include <deepity/networks/DiscriminativePCNetwork.h>
#include <deepity/networks/FullPCNetwork.h>
#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <deepity/networks/SimpleConvPCNetwork.h>
#include <deepity/networks/SimplePCNetwork.h>

// ============================================================================
// Internal Helpers
// ============================================================================
namespace
{

template <typename NetT> void BindCommonPCNetwork(nb::class_<NetT>& cls, const char* className)
{
  cls.def(nb::init<int>(), nb::arg("batch_size"), "Construct a network with a fixed batch size.")
      .def(
          "randomize_weights",
          [](NetT& self)
          {
            std::random_device rd;
            std::mt19937 rng(rd());
            self.RandomizeWeights(rng);
          },
          "Initialize every layer's weights randomly.")
      .def(
          "clamp_input",
          [](NetT& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.Clamp(values);
          },
          nb::arg("input"),
          "Clamp the first (input) layer to the supplied, flattened batch.")
      .def("compile", &NetT::Compile, "Compiles all layers into a contiguous block.")
      .def("calculate_state", &NetT::CalculateState, "Compute the total network energy.")
      .def("update_state", &NetT::UpdateState, "Run one inference step.")
      .def("update_weights", &NetT::UpdateWeights, "Apply weight updates to every layer.")
      .def("reset_state", &NetT::ResetState, "Resets the beliefs 'z' on every layer.")
      .def("get_terminal_layer",
           &NetT::GetTerminalLayer,
           nb::rv_policy::reference,
           "Return the last layer.")
      .def_prop_ro("batch_size", &NetT::GetBatchSize)
      .def_prop_ro(
          "layers",
          [](NetT& self)
          {
            nb::list result;
            for (const auto& layer : self.GetLayers())
              result.append(nb::cast(layer.get(), nb::rv_policy::reference));
            return result;
          },
          "List of layer objects owned by the network.")
      .def("__len__", [](const NetT& self) { return self.GetLayers().size(); })
      .def(
          "__getitem__",
          [](NetT& self, std::ptrdiff_t index)
          {
            auto& layers = self.GetLayers();
            if (index < 0)
              index += static_cast<std::ptrdiff_t>(layers.size());
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size()))
              throw nb::index_error();
            return layers[index].get();
          },
          nb::rv_policy::reference_internal)
      .def("__repr__",
           [className](const NetT& self)
           {
             return "<" + std::string(className) +
                    " layers=" + std::to_string(self.GetLayers().size()) +
                    " batch_size=" + std::to_string(self.GetBatchSize()) + ">";
           });
}

} // namespace

// ============================================================================
// Network Bindings
// ============================================================================
void bind_networks(nb::module_& m)
{
  auto discNetCls = nb::class_<Deep::DiscriminativePCNetwork>(
      m, "DiscriminativePCNetwork", "Predictive Coding Network.");
  BindCommonPCNetwork<Deep::DiscriminativePCNetwork>(discNetCls, "DiscriminativePCNetwork");
  discNetCls
      .def(
          "__init__",
          [](Deep::DiscriminativePCNetwork* self, const std::string& device)
          {
            Deep::DeviceType dt = (device == "cuda" || device == "gpu")
                                      ? Deep::DeviceType::DEVICE_GPU
                                      : Deep::DeviceType::DEVICE_CPU;
            new (self) Deep::DiscriminativePCNetwork(dt);
          },
          nb::arg("device") = "cpu",
          "Construct a network with automatic batch-size detection, on the given device.")
      .def(
          "add_layer",
          [](Deep::DiscriminativePCNetwork& self,
             int size,
             int next_size,
             float lr,
             float ir,
             float pr,
             float lmbda,
             const std::string& activation,
             const std::string& activation_deriv)
          {
            self.AddLayer(size,
                          next_size,
                          lr,
                          ir,
                          pr,
                          lmbda,
                          resolveActEnum(activation),
                          resolveActEnum(activation_deriv));
          },
          nb::arg("size"),
          nb::arg("next_size"),
          nb::arg("lr") = 1e-6f,
          nb::arg("ir") = 0.1f,
          nb::arg("pr") = 0.01f,
          nb::arg("lmbda") = 1e-2f,
          nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu",
          "Add a layer to the network.")
      .def("set_inference_rate",
           &Deep::DiscriminativePCNetwork::SetInferenceRate,
           "Sets the inference rate of each layer.",
           nb::arg("ir"))
      .def("set_learning_rate",
           &Deep::DiscriminativePCNetwork::SetLearningRate,
           "Sets the learning rate of each layer.",
           nb::arg("lr"))
      .def("set_precision_rate",
           &Deep::DiscriminativePCNetwork::SetPrecisionRate,
           "Sets the precision rate of each layer.",
           nb::arg("pr"))
      .def("set_lambda",
           &Deep::DiscriminativePCNetwork::SetLambda,
           "Sets lambda of each layer.",
           nb::arg("l"))
      .def(
          "set_optimizer",
          [](Deep::DiscriminativePCNetwork& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"))
      .def("save",
           &Deep::DiscriminativePCNetwork::Save,
           "Saves the network architecture and weights to a structured directory.",
           nb::arg("dir_path"))
      .def("load",
           &Deep::DiscriminativePCNetwork::Load,
           "Loads network weights from a structured directory into the compiled MemoryArena.",
           nb::arg("dir_path"))
      .def("update_precision",
           &Deep::DiscriminativePCNetwork::UpdatePrecision,
           "Apply precision updates to every layer.")
      .def("project_forward", &Deep::DiscriminativePCNetwork::ProjectForward)
      .def(
          "train_step_with_projection",
          [](Deep::DiscriminativePCNetwork& self, FloatArray x, FloatArray y, int steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStepWithProjection(xvec, yvec, steps);
          },
          nb::arg("x"),
          nb::arg("y"),
          nb::arg("steps"))
      .def(
          "predict_with_projection",
          [](Deep::DiscriminativePCNetwork& self, FloatArray x, int steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> result = self.PredictWithProjection(xvec, steps);
            Deep::DiscriminativePCLayer* terminal = self.GetTerminalLayer();
            return CopyToNewArray(
                result.data(),
                {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()});
          },
          nb::arg("x"),
          nb::arg("steps"));

  auto simpleNetCls = nb::class_<Deep::SimplePCNetwork>(
      m, "SimplePCNetwork", "Predictive Coding Network built from SimplePCLayers.");
  BindCommonPCNetwork<Deep::SimplePCNetwork>(simpleNetCls, "SimplePCNetwork");
  simpleNetCls.def(
      "__init__",
      [](Deep::SimplePCNetwork* self, int batch_size, const std::string& device)
      {
        Deep::DeviceType dt = (device == "cuda" || device == "gpu") ? Deep::DeviceType::DEVICE_GPU
                                                                    : Deep::DeviceType::DEVICE_CPU;
        new (self) Deep::SimplePCNetwork(batch_size, dt);
      },
      nb::arg("batch_size"),
      nb::arg("device") = "cpu",
      "Construct a network with a fixed batch size and device (\"cpu\" or \"cuda\"/\"gpu\").");
  simpleNetCls
      .def(
          "add_layer",
          [](Deep::SimplePCNetwork& self,
             int size,
             int next_size,
             float lr,
             float ir,
             float lmbda,
             const std::string& activation,
             const std::string& activation_deriv)
          {
            self.AddLayer(size,
                          next_size,
                          lr,
                          ir,
                          lmbda,
                          resolveActEnum(activation),
                          resolveActEnum(activation_deriv));
          },
          nb::arg("size"),
          nb::arg("next_size"),
          nb::arg("lr") = 1e-6f,
          nb::arg("ir") = 0.1f,
          nb::arg("lmbda") = 1e-2f,
          nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu",
          "Add a layer to the network.")
      .def(
          "set_optimizer",
          [](Deep::SimplePCNetwork& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"),
          "Sets the optimizer: ADAM, ADAMW, or SGD.")
      .def("project_forward",
           &Deep::SimplePCNetwork::ProjectForward,
           "Seeds hidden layers from a genuine forward pass through current "
           "weights, instead of zero-init. Call AFTER clamp_input(), BEFORE "
           "the settling loop.")
      .def(
          "train_step_with_projection",
          [](Deep::SimplePCNetwork& self, FloatArray x, FloatArray y, int steps, bool computeEnergy)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStepWithProjection(xvec, yvec, steps, computeEnergy);
          },
          nb::arg("x"),
          nb::arg("y"),
          nb::arg("steps"),
          nb::arg("computeEnergy") = true)
      .def(
          "predict_with_projection",
          [](Deep::SimplePCNetwork& self, FloatArray x, int steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> out_beliefs = self.PredictWithProjection(xvec, steps);
            return CopyToNewArray(out_beliefs.data(), {out_beliefs.size()});
          },
          nb::arg("x"),
          nb::arg("steps"),
          "Runs forward-projection init and settling loop entirely in C++, returning terminal "
          "beliefs.")
      .def(
          "randomize_weights",
          [](Deep::SimplePCNetwork& self, const std::string& distribution)
          {
            std::random_device rd;
            std::mt19937 rng(rd());
            self.RandomizeWeights(rng, distribution.c_str());
          },
          nb::arg("distribution"),
          "Initialize every layer's weights using a distribution string, "
          "e.g. \"normal(0, 1)\" or \"uniform(-0.3, 0.3)\".");

  nb::class_<Deep::GaussSeidelPCNetwork>(
      m, "GaussSeidelPCNetwork", "Predictive Coding Network with Gauss-Seidel settling dynamics.")
      .def(
          "__init__",
          [](Deep::GaussSeidelPCNetwork* self, int batch_size, const std::string& device)
          {
            Deep::DeviceType dt = (device == "cuda" || device == "gpu")
                                      ? Deep::DeviceType::DEVICE_GPU
                                      : Deep::DeviceType::DEVICE_CPU;
            new (self) Deep::GaussSeidelPCNetwork(batch_size, dt);
          },
          nb::arg("batch_size"),
          nb::arg("device") = "cpu")
      .def(
          "add_layer",
          [](Deep::GaussSeidelPCNetwork& self,
             int size,
             int next_size,
             float lr,
             float ir,
             float lmbda,
             const std::string& activation,
             const std::string& activation_deriv)
          {
            self.AddLayer(size,
                          next_size,
                          lr,
                          ir,
                          lmbda,
                          resolveActEnum(activation),
                          resolveActEnum(activation_deriv));
          },
          nb::arg("size"),
          nb::arg("next_size"),
          nb::arg("lr") = 1e-6f,
          nb::arg("ir") = 0.1f,
          nb::arg("lmbda") = 1e-2f,
          nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu")
      .def("compile", &Deep::GaussSeidelPCNetwork::Compile)
      .def("randomize_weights",
           [](Deep::GaussSeidelPCNetwork& self)
           {
             std::random_device rd;
             std::mt19937 rng(rd());
             self.RandomizeWeights(rng);
           })
      .def("reset_state", &Deep::GaussSeidelPCNetwork::ResetState)
      .def(
          "clamp_input",
          [](Deep::GaussSeidelPCNetwork& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.Clamp(values);
          },
          nb::arg("input"))
      .def("step", &Deep::GaussSeidelPCNetwork::Step)
      .def("update_weights", &Deep::GaussSeidelPCNetwork::UpdateWeights)
      .def("project_forward", &Deep::GaussSeidelPCNetwork::ProjectForward)
      .def(
          "set_optimizer",
          [](Deep::GaussSeidelPCNetwork& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"))
      .def("set_learning_rate", &Deep::GaussSeidelPCNetwork::SetLearningRate)
      .def(
          "train_step",
          [](Deep::GaussSeidelPCNetwork& self, FloatArray x, FloatArray y, int steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStep(xvec, yvec, steps);
          },
          nb::arg("x"),
          nb::arg("y"),
          nb::arg("steps"))
      .def(
          "train_step_with_projection",
          [](Deep::GaussSeidelPCNetwork& self, FloatArray x, FloatArray y, int steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStepWithProjection(xvec, yvec, steps);
          },
          nb::arg("x"),
          nb::arg("y"),
          nb::arg("steps"))
      .def(
          "predict",
          [](Deep::GaussSeidelPCNetwork& self, FloatArray x, int steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> result = self.Predict(xvec, steps);
            Deep::GaussSeidelPCLayer* terminal = self.GetTerminalLayer();
            return CopyToNewArray(
                result.data(),
                {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()});
          },
          nb::arg("x"),
          nb::arg("steps"))
      .def_prop_ro("batch_size", &Deep::GaussSeidelPCNetwork::GetBatchSize)
      .def_prop_ro("layers",
                   [](Deep::GaussSeidelPCNetwork& self)
                   {
                     nb::list result;
                     for (auto& layer : self.GetLayers())
                       result.append(nb::cast(layer.get(), nb::rv_policy::reference));
                     return result;
                   })
      .def("__len__",
           [](const Deep::GaussSeidelPCNetwork& self) { return self.GetLayers().size(); })
      .def(
          "__getitem__",
          [](Deep::GaussSeidelPCNetwork& self, std::ptrdiff_t index)
          {
            auto& layers = self.GetLayers();
            if (index < 0)
              index += static_cast<std::ptrdiff_t>(layers.size());
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size()))
              throw nb::index_error();
            return layers[index].get();
          },
          nb::rv_policy::reference_internal);

  nb::class_<Deep::DirectKPPCNetwork>(
      m,
      "DirectKPPCNetwork",
      "Predictive Coding Network with Direct Kolen-Pollack feedback alignment.")
      .def(
          "__init__",
          [](Deep::DirectKPPCNetwork* self, int batch_size, const std::string& device)
          {
            Deep::DeviceType dt = (device == "cuda" || device == "gpu")
                                      ? Deep::DeviceType::DEVICE_GPU
                                      : Deep::DeviceType::DEVICE_CPU;
            new (self) Deep::DirectKPPCNetwork(batch_size, dt);
          },
          nb::arg("batch_size"),
          nb::arg("device") = "cpu")
      .def(
          "add_layer",
          [](Deep::DirectKPPCNetwork& self,
             size_t size,
             size_t next_size,
             size_t terminal_size,
             float lr,
             float ir,
             float fl,
             float lmbda,
             const std::string& activation,
             const std::string& activation_deriv)
          {
            self.AddLayer(size,
                          next_size,
                          terminal_size,
                          lr,
                          ir,
                          fl,
                          lmbda,
                          resolveActEnum(activation),
                          resolveActEnum(activation_deriv));
          },
          nb::arg("size"),
          nb::arg("next_size"),
          nb::arg("terminal_size"),
          nb::arg("lr") = 1e-6f,
          nb::arg("ir") = 0.1f,
          nb::arg("fl") = 1e-4f,
          nb::arg("lmbda") = 1e-2f,
          nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu")
      .def("compile", &Deep::DirectKPPCNetwork::Compile)
      .def("randomize_weights",
           [](Deep::DirectKPPCNetwork& self)
           {
             std::random_device rd;
             std::mt19937 rng(rd());
             self.RandomizeWeights(rng);
           })
      .def("reset_state", &Deep::DirectKPPCNetwork::ResetState)
      .def(
          "clamp_input",
          [](Deep::DirectKPPCNetwork& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.Clamp(values);
          },
          nb::arg("input"))
      .def("project_forward", &Deep::DirectKPPCNetwork::ProjectForward)
      .def("calculate_terminal_error", &Deep::DirectKPPCNetwork::CalculateTerminalError)
      .def("direct_feedback_update", &Deep::DirectKPPCNetwork::DirectFeedbackUpdate)
      .def("step", &Deep::DirectKPPCNetwork::Step)
      .def("update_weights", &Deep::DirectKPPCNetwork::UpdateWeights)
      .def(
          "set_optimizer",
          [](Deep::DirectKPPCNetwork& self, const std::string& opt)
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
          "set_psi_optimizer",
          [](Deep::DirectKPPCNetwork& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetPsiOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"))
      .def("set_learning_rate", &Deep::DirectKPPCNetwork::SetLearningRate, nb::arg("lr"))
      .def("set_feedback_rate", &Deep::DirectKPPCNetwork::SetFeedbackRate, nb::arg("fl"))
      .def(
          "train_step",
          [](Deep::DirectKPPCNetwork& self, FloatArray x, FloatArray y, int inference_steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStep(xvec, yvec, inference_steps);
          },
          nb::arg("x"),
          nb::arg("y"),
          nb::arg("inference_steps") = 1)
      .def(
          "predict",
          [](Deep::DirectKPPCNetwork& self, FloatArray x, int inference_steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> result = self.Predict(xvec, inference_steps);
            Deep::DirectKPPCLayer* terminal = self.GetTerminalLayer();
            return CopyToNewArray(
                result.data(),
                {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()});
          },
          nb::arg("x"),
          nb::arg("inference_steps"))
      .def_prop_ro("batch_size", &Deep::DirectKPPCNetwork::GetBatchSize)
      .def_prop_ro("layers",
                   [](Deep::DirectKPPCNetwork& self)
                   {
                     nb::list result;
                     for (auto& layer : self.GetLayers())
                       result.append(nb::cast(layer.get(), nb::rv_policy::reference));
                     return result;
                   })
      .def("get_terminal_layer",
           &Deep::DirectKPPCNetwork::GetTerminalLayer,
           nb::rv_policy::reference)
      .def("__len__", [](const Deep::DirectKPPCNetwork& self) { return self.GetLayers().size(); })
      .def(
          "__getitem__",
          [](Deep::DirectKPPCNetwork& self, std::ptrdiff_t index)
          {
            auto& layers = self.GetLayers();
            if (index < 0)
              index += static_cast<std::ptrdiff_t>(layers.size());
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size()))
              throw nb::index_error();
            return layers[index].get();
          },
          nb::rv_policy::reference_internal)
      .def("__repr__",
           [](const Deep::DirectKPPCNetwork& self)
           {
             return "<DirectKPPCNetwork layers=" + std::to_string(self.GetLayers().size()) +
                    " batch_size=" + std::to_string(self.GetBatchSize()) + ">";
           });

  nb::class_<Deep::FullPCNetwork>(
      m,
      "FullPCNetwork",
      "Predictive Coding Network combining muPC scaling, optional "
      "residual connections, and DKP direct feedback, every extra OFF "
      "by default, matching FullPCLayer's own defaults.")
      .def(
          "__init__",
          [](Deep::FullPCNetwork* self, int batch_size, const std::string& device)
          {
            Deep::DeviceType dt = (device == "cuda" || device == "gpu")
                                      ? Deep::DeviceType::DEVICE_GPU
                                      : Deep::DeviceType::DEVICE_CPU;
            new (self) Deep::FullPCNetwork(batch_size, dt);
          },
          nb::arg("batch_size"),
          nb::arg("device") = "cpu")
      .def(
          "add_layer",
          [](Deep::FullPCNetwork& self,
             size_t size,
             size_t next_size,
             size_t terminal_size,
             float lr,
             float ir,
             float fl,
             float lmbda,
             const std::string& activation,
             const std::string& activation_deriv)
          {
            self.AddLayer(size,
                          next_size,
                          terminal_size,
                          lr,
                          ir,
                          fl,
                          lmbda,
                          resolveActEnum(activation),
                          resolveActEnum(activation_deriv));
          },
          nb::arg("size"),
          nb::arg("next_size"),
          nb::arg("terminal_size"),
          nb::arg("lr") = 1e-6f,
          nb::arg("ir") = 0.1f,
          nb::arg("fl") = 1e-4f,
          nb::arg("lmbda") = 1e-2f,
          nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu")
      .def("set_use_mu_pc_scaling",
           &Deep::FullPCNetwork::SetUseMuPCScaling,
           nb::arg("enabled"),
           "Enable/disable muPC's full Table 1 parameterization: per-layer "
           "forward scaling (`a`) AND unit-variance weight init, in place "
           "of plain PC's fan-scaled init. OFF by default. Must be called "
           "before compile() and before randomize_weights().")
      .def("set_use_residual_connections",
           &Deep::FullPCNetwork::SetUseResidualConnections,
           nb::arg("enabled"),
           "Enable/disable residual/skip connections on middle hidden "
           "layers. OFF by default. Must be called before compile(), "
           "compile() raises ValueError if enabled middle layers don't "
           "have matching width.")
      .def("set_use_ipc",
           &Deep::FullPCNetwork::SetUseIPC,
           nb::arg("enabled"),
           "Weights are updated every settling step instead of at the end of all steps when "
           "enabled.")
      .def("set_use_epc",
           &Deep::FullPCNetwork::SetUseEPC,
           nb::arg("enabled"),
           "Enable/disable ePC settling: each settling step becomes a full forward+backward "
           "sweep through every hidden layer (exact PC gradient, no signal decay with depth) "
           "instead of the usual one-hop local update. OFF by default.")
      .def("set_use_momentum",
           &Deep::FullPCNetwork::SetUseMomentum,
           nb::arg("enabled"),
           nb::arg("beta") = 0.9f,
           "Enable/disable momentum (inertial) settling on every layer. "
           "OFF by default. beta is the EMA decay (higher = more smoothing).")
      .def("set_use_cross_entropy",
           &Deep::FullPCNetwork::SetUseCrossEntropy,
           nb::arg("enabled"),
           "Enable/disable softmax cross-entropy energy on the terminal layer "
           "only (not every layer). OFF by default (plain Gaussian energy).")
      .def("compile", &Deep::FullPCNetwork::Compile)
      .def(
          "randomize_weights",
          [](Deep::FullPCNetwork& self, std::optional<uint32_t> seed)
          {
            std::mt19937 rng = seed.has_value() ? std::mt19937(seed.value())
                                                : std::mt19937(std::random_device{}());
            self.RandomizeWeights(rng);
          },
          nb::arg("seed") = nb::none())
      .def("reset_state", &Deep::FullPCNetwork::ResetState)
      .def(
          "clamp_input",
          [](Deep::FullPCNetwork& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.Clamp(values);
          },
          nb::arg("input"))
      .def("project_forward", &Deep::FullPCNetwork::ProjectForward)
      .def("calculate_terminal_error", &Deep::FullPCNetwork::CalculateTerminalError)
      .def("direct_feedback_update", &Deep::FullPCNetwork::DirectFeedbackUpdate)
      .def("step", &Deep::FullPCNetwork::Step)
      .def("update_weights", &Deep::FullPCNetwork::UpdateWeights)
      .def(
          "set_optimizer",
          [](Deep::FullPCNetwork& self, const std::string& opt)
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
          "set_psi_optimizer",
          [](Deep::FullPCNetwork& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetPsiOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"))
      .def("set_learning_rate", &Deep::FullPCNetwork::SetLearningRate, nb::arg("lr"))
      .def("set_feedback_rate", &Deep::FullPCNetwork::SetFeedbackRate, nb::arg("fl"))
      .def(
          "train_step",
          [](Deep::FullPCNetwork& self, FloatArray x, FloatArray y, int inference_steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStep(xvec, yvec, inference_steps);
          },
          nb::arg("x"),
          nb::arg("y"),
          nb::arg("inference_steps") = 1)
      .def(
          "predict",
          [](Deep::FullPCNetwork& self, FloatArray x, int inference_steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> result = self.Predict(xvec, inference_steps);
            Deep::FullPCLayer* terminal = self.GetTerminalLayer();
            return CopyToNewArray(
                result.data(),
                {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()});
          },
          nb::arg("x"),
          nb::arg("inference_steps"))
      .def_prop_ro("batch_size", &Deep::FullPCNetwork::GetBatchSize)
      .def_prop_ro("layers",
                   [](Deep::FullPCNetwork& self)
                   {
                     nb::list result;
                     for (auto& layer : self.GetLayers())
                       result.append(nb::cast(layer.get(), nb::rv_policy::reference));
                     return result;
                   })
      .def("get_terminal_layer", &Deep::FullPCNetwork::GetTerminalLayer, nb::rv_policy::reference)
      .def("__len__", [](const Deep::FullPCNetwork& self) { return self.GetLayers().size(); })
      .def(
          "__getitem__",
          [](Deep::FullPCNetwork& self, std::ptrdiff_t index)
          {
            auto& layers = self.GetLayers();
            if (index < 0)
              index += static_cast<std::ptrdiff_t>(layers.size());
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size()))
              throw nb::index_error();
            return layers[index].get();
          },
          nb::rv_policy::reference_internal)
      .def("__repr__",
           [](const Deep::FullPCNetwork& self)
           {
             return "<FullPCNetwork layers=" + std::to_string(self.GetLayers().size()) +
                    " batch_size=" + std::to_string(self.GetBatchSize()) + ">";
           });

  nb::class_<Deep::FullConvPCNetwork>(
      m,
      "FullConvPCNetwork",
      "Convolutional analog of FullPCNetwork: every PC variant, each independently "
      "toggleable, off by default so plain defaults reproduce plain conv PC settling.")
      .def(
          "__init__",
          [](Deep::FullConvPCNetwork* self, int batch_size, const std::string& device)
          {
            Deep::DeviceType dt = (device == "cuda" || device == "gpu")
                                      ? Deep::DeviceType::DEVICE_GPU
                                      : Deep::DeviceType::DEVICE_CPU;
            new (self) Deep::FullConvPCNetwork(batch_size, dt);
          },
          nb::arg("batch_size"),
          nb::arg("device") = "cpu")
      .def(
          "add_layer",
          [](Deep::FullConvPCNetwork& self,
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
             float lr,
             float ir,
             float fl,
             float lmbda,
             const std::string& activation,
             const std::string& activation_deriv,
             int pool_h,
             int pool_w,
             int pool_stride_h,
             int pool_stride_w)
          {
            self.AddLayer(in_channels,
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
                          lr,
                          ir,
                          fl,
                          lmbda,
                          resolveActEnum(activation),
                          resolveActEnum(activation_deriv),
                          pool_h,
                          pool_w,
                          pool_stride_h,
                          pool_stride_w);
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
          nb::arg("lr") = 1e-6f,
          nb::arg("ir") = 0.1f,
          nb::arg("fl") = 1e-4f,
          nb::arg("lmbda") = 1e-2f,
          nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu",
          nb::arg("pool_h") = 1,
          nb::arg("pool_w") = 1,
          nb::arg("pool_stride_h") = 1,
          nb::arg("pool_stride_w") = 1)
      .def("set_use_mu_pc_scaling", &Deep::FullConvPCNetwork::SetUseMuPCScaling, nb::arg("enabled"))
      .def("set_use_residual_connections",
           &Deep::FullConvPCNetwork::SetUseResidualConnections,
           nb::arg("enabled"))
      .def("set_use_ipc", &Deep::FullConvPCNetwork::SetUseIPC, nb::arg("enabled"))
      .def("set_use_epc", &Deep::FullConvPCNetwork::SetUseEPC, nb::arg("enabled"))
      .def("set_use_momentum",
           &Deep::FullConvPCNetwork::SetUseMomentum,
           nb::arg("enabled"),
           nb::arg("beta") = 0.9f)
      .def("set_use_cross_entropy", &Deep::FullConvPCNetwork::SetUseCrossEntropy, nb::arg("enabled"))
      .def("compile", &Deep::FullConvPCNetwork::Compile)
      .def(
          "randomize_weights",
          [](Deep::FullConvPCNetwork& self, std::optional<uint32_t> seed)
          {
            std::mt19937 rng = seed.has_value() ? std::mt19937(seed.value())
                                                : std::mt19937(std::random_device{}());
            self.RandomizeWeights(rng);
          },
          nb::arg("seed") = nb::none())
      .def("reset_state", &Deep::FullConvPCNetwork::ResetState)
      .def(
          "clamp_input",
          [](Deep::FullConvPCNetwork& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.Clamp(values);
          },
          nb::arg("input"))
      .def("project_forward", &Deep::FullConvPCNetwork::ProjectForward)
      .def("calculate_terminal_error", &Deep::FullConvPCNetwork::CalculateTerminalError)
      .def("direct_feedback_update", &Deep::FullConvPCNetwork::DirectFeedbackUpdate)
      .def("step", &Deep::FullConvPCNetwork::Step)
      .def("update_weights", &Deep::FullConvPCNetwork::UpdateWeights)
      .def(
          "set_optimizer",
          [](Deep::FullConvPCNetwork& self, const std::string& opt)
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
          "set_psi_optimizer",
          [](Deep::FullConvPCNetwork& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetPsiOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetPsiOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"))
      .def("set_learning_rate", &Deep::FullConvPCNetwork::SetLearningRate, nb::arg("lr"))
      .def("set_feedback_rate", &Deep::FullConvPCNetwork::SetFeedbackRate, nb::arg("fl"))
      .def("set_inference_rate", &Deep::FullConvPCNetwork::SetInferenceRate, nb::arg("ir"))
      .def("set_adam_epsilon", &Deep::FullConvPCNetwork::SetAdamEpsilon, nb::arg("eps"))
      .def(
          "train_step",
          [](Deep::FullConvPCNetwork& self, FloatArray x, FloatArray y, int inference_steps,
             bool computeEnergy)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStep(xvec, yvec, inference_steps, computeEnergy);
          },
          nb::arg("x"),
          nb::arg("y"),
          nb::arg("inference_steps") = 1,
          nb::arg("computeEnergy") = true)
      .def(
          "get_all_layer_diagnostics",
          [](Deep::FullConvPCNetwork& self)
          {
            std::vector<float> stats = self.GetAllLayerDiagnostics();
            return CopyToNewArray(stats.data(), {self.GetLayers().size(), (size_t)4});
          },
          "Each layer's [error mean, error RMS, weight norm, dead-mu fraction] as a 2D "
          "[layer, stat] array, read from whatever TrainStep() last left in that layer's "
          "buffers -- call right after train_step() to see the state that drove it.")
      .def(
          "predict",
          [](Deep::FullConvPCNetwork& self, FloatArray x, int inference_steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> result = self.Predict(xvec, inference_steps);
            Deep::FullConvPCLayer* terminal = self.GetTerminalLayer();
            return CopyToNewArray(
                result.data(),
                {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()});
          },
          nb::arg("x"),
          nb::arg("inference_steps"))
      .def_prop_ro("batch_size", &Deep::FullConvPCNetwork::GetBatchSize)
      .def_prop_ro("layers",
                   [](Deep::FullConvPCNetwork& self)
                   {
                     nb::list result;
                     for (auto& layer : self.GetLayers())
                       result.append(nb::cast(layer.get(), nb::rv_policy::reference));
                     return result;
                   })
      .def("get_terminal_layer",
           &Deep::FullConvPCNetwork::GetTerminalLayer,
           nb::rv_policy::reference)
      .def("__len__", [](const Deep::FullConvPCNetwork& self) { return self.GetLayers().size(); })
      .def(
          "__getitem__",
          [](Deep::FullConvPCNetwork& self, std::ptrdiff_t index)
          {
            auto& layers = self.GetLayers();
            if (index < 0)
              index += static_cast<std::ptrdiff_t>(layers.size());
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size()))
              throw nb::index_error();
            return layers[index].get();
          },
          nb::rv_policy::reference_internal)
      .def("__repr__",
           [](const Deep::FullConvPCNetwork& self)
           {
             return "<FullConvPCNetwork layers=" + std::to_string(self.GetLayers().size()) +
                    " batch_size=" + std::to_string(self.GetBatchSize()) + ">";
           });

  nb::class_<Deep::ConvPCNetwork>(m, "ConvPCNetwork", "Convolutional Predictive Coding Network.")
      .def(
          "__init__",
          [](Deep::ConvPCNetwork* self, int batch_size, const std::string& device)
          {
            Deep::DeviceType dt = (device == "cuda" || device == "gpu")
                                      ? Deep::DeviceType::DEVICE_GPU
                                      : Deep::DeviceType::DEVICE_CPU;
            new (self) Deep::ConvPCNetwork(batch_size, dt);
          },
          nb::arg("batch_size"),
          nb::arg("device") = "cpu",
          "Construct a network with a fixed batch size, on the given device.")
      .def(
          "add_layer",
          [](Deep::ConvPCNetwork& self,
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
             float lr,
             float ir,
             float pr,
             float lmbda,
             const std::string& activation,
             const std::string& activation_deriv)
          {
            self.AddLayer(in_channels,
                          out_channels,
                          in_height,
                          in_width,
                          kernel_h,
                          kernel_w,
                          stride_h,
                          stride_w,
                          pad_h,
                          pad_w,
                          lr,
                          ir,
                          pr,
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
          nb::arg("lr") = 1e-6f,
          nb::arg("ir") = 0.1f,
          nb::arg("pr") = 0.0f,
          nb::arg("lmbda") = 1e-2f,
          nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu")
      .def("compile", &Deep::ConvPCNetwork::Compile, "Compiles all layers into a contiguous block.")
      .def("randomize_weights",
           [](Deep::ConvPCNetwork& self)
           {
             std::random_device rd;
             std::mt19937 rng(rd());
             self.RandomizeWeights(rng);
           })
      .def(
          "clamp_input",
          [](Deep::ConvPCNetwork& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.Clamp(values);
          },
          nb::arg("input"))
      .def("calculate_state", &Deep::ConvPCNetwork::CalculateState)
      .def("update_state", &Deep::ConvPCNetwork::UpdateState)
      .def("update_weights", &Deep::ConvPCNetwork::UpdateWeights)
      .def("update_precision", &Deep::ConvPCNetwork::UpdatePrecision)
      .def("reset_state", &Deep::ConvPCNetwork::ResetState)
      .def(
          "train_step",
          [](Deep::ConvPCNetwork& self, FloatArray x, FloatArray y, int steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStep(xvec, yvec, steps);
          },
          nb::arg("x"),
          nb::arg("y"),
          nb::arg("steps"))
      .def(
          "predict",
          [](Deep::ConvPCNetwork& self, FloatArray x, int steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> result = self.Predict(xvec, steps);
            Deep::ConvPCLayer* terminal = self.GetTerminalLayer();
            return CopyToNewArray(
                result.data(),
                {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()});
          },
          nb::arg("x"),
          nb::arg("steps"))
      .def("get_terminal_layer", &Deep::ConvPCNetwork::GetTerminalLayer, nb::rv_policy::reference)
      .def_prop_ro("batch_size", &Deep::ConvPCNetwork::GetBatchSize)
      .def_prop_ro("layers",
                   [](Deep::ConvPCNetwork& self)
                   {
                     nb::list result;
                     for (auto& layer : self.GetLayers())
                       result.append(nb::cast(layer.get(), nb::rv_policy::reference));
                     return result;
                   })
      .def("__len__", [](const Deep::ConvPCNetwork& self) { return self.GetLayers().size(); })
      .def(
          "__getitem__",
          [](Deep::ConvPCNetwork& self, std::ptrdiff_t index)
          {
            auto& layers = self.GetLayers();
            if (index < 0)
              index += static_cast<std::ptrdiff_t>(layers.size());
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size()))
              throw nb::index_error();
            return layers[index].get();
          },
          nb::rv_policy::reference_internal)
      .def("__repr__",
           [](const Deep::ConvPCNetwork& self)
           {
             return "<ConvPCNetwork layers=" + std::to_string(self.GetLayers().size()) +
                    " batch_size=" + std::to_string(self.GetBatchSize()) + ">";
           })
      .def("train_step_with_projection", &Deep::ConvPCNetwork::TrainStepWithProjection)
      .def("predict_with_projection", &Deep::ConvPCNetwork::PredictWithProjection)
      .def("project_forward", &Deep::ConvPCNetwork::ProjectForward);

  nb::class_<Deep::SimpleConvPCNetwork>(m,
                                        "SimpleConvPCNetwork",
                                        "Convolutional Predictive Coding Network built from "
                                        "SimpleConvPCLayers (precision-free, AdamW-capable).")
      .def(
          "__init__",
          [](Deep::SimpleConvPCNetwork* self, int batch_size, const std::string& device)
          {
            Deep::DeviceType dt = (device == "cuda" || device == "gpu")
                                      ? Deep::DeviceType::DEVICE_GPU
                                      : Deep::DeviceType::DEVICE_CPU;
            new (self) Deep::SimpleConvPCNetwork(batch_size, dt);
          },
          nb::arg("batch_size"),
          nb::arg("device") = "cpu",
          "Construct a network with a fixed batch size and device (\"cpu\" or \"gpu\").")
      .def(
          "add_layer",
          [](Deep::SimpleConvPCNetwork& self,
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
             float lr,
             float ir,
             float lmbda,
             const std::string& activation,
             const std::string& activation_deriv)
          {
            self.AddLayer(in_channels,
                          out_channels,
                          in_height,
                          in_width,
                          kernel_h,
                          kernel_w,
                          stride_h,
                          stride_w,
                          pad_h,
                          pad_w,
                          lr,
                          ir,
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
          nb::arg("lr") = 1e-6f,
          nb::arg("ir") = 0.1f,
          nb::arg("lmbda") = 1e-2f,
          nb::arg("activation") = "relu",
          nb::arg("activation_deriv") = "drelu")
      .def(
          "set_optimizer",
          [](Deep::SimpleConvPCNetwork& self, const std::string& opt)
          {
            if (opt == "ADAM")
              self.SetOptimizer(Deep::OptimizerType::ADAM);
            else if (opt == "ADAMW")
              self.SetOptimizer(Deep::OptimizerType::ADAMW);
            else
              self.SetOptimizer(Deep::OptimizerType::SGD);
          },
          nb::arg("optimizer"),
          "Sets the optimizer: ADAM, ADAMW, or SGD. Call BEFORE compile().")
      .def("compile",
           &Deep::SimpleConvPCNetwork::Compile,
           "Compiles all layers into a contiguous block.")
      .def("randomize_weights",
           [](Deep::SimpleConvPCNetwork& self)
           {
             std::random_device rd;
             std::mt19937 rng(rd());
             self.RandomizeWeights(rng);
           })
      .def(
          "clamp_input",
          [](Deep::SimpleConvPCNetwork& self, FloatArray input)
          {
            std::vector<float> values(input.data(), input.data() + input.size());
            self.Clamp(values);
          },
          nb::arg("input"))
      .def("calculate_state", &Deep::SimpleConvPCNetwork::CalculateState)
      .def("update_state", &Deep::SimpleConvPCNetwork::UpdateState)
      .def("update_weights", &Deep::SimpleConvPCNetwork::UpdateWeights)
      .def("reset_state", &Deep::SimpleConvPCNetwork::ResetState)
      .def(
          "train_step",
          [](Deep::SimpleConvPCNetwork& self, FloatArray x, FloatArray y, int steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> yvec(y.data(), y.data() + y.size());
            return self.TrainStep(xvec, yvec, steps);
          },
          nb::arg("x"),
          nb::arg("y"),
          nb::arg("steps"))
      .def(
          "predict",
          [](Deep::SimpleConvPCNetwork& self, FloatArray x, int steps)
          {
            std::vector<float> xvec(x.data(), x.data() + x.size());
            std::vector<float> result = self.Predict(xvec, steps);
            Deep::SimpleConvPCLayer* terminal = self.GetTerminalLayer();
            return CopyToNewArray(
                result.data(),
                {(size_t)terminal->GetBatchSize(), (size_t)terminal->GetInputSize()});
          },
          nb::arg("x"),
          nb::arg("steps"))
      .def("get_terminal_layer",
           &Deep::SimpleConvPCNetwork::GetTerminalLayer,
           nb::rv_policy::reference)
      .def_prop_ro("batch_size", &Deep::SimpleConvPCNetwork::GetBatchSize)
      .def_prop_ro("layers",
                   [](Deep::SimpleConvPCNetwork& self)
                   {
                     nb::list result;
                     for (auto& layer : self.GetLayers())
                       result.append(nb::cast(layer.get(), nb::rv_policy::reference));
                     return result;
                   })
      .def("__len__", [](const Deep::SimpleConvPCNetwork& self) { return self.GetLayers().size(); })
      .def(
          "__getitem__",
          [](Deep::SimpleConvPCNetwork& self, std::ptrdiff_t index)
          {
            auto& layers = self.GetLayers();
            if (index < 0)
              index += static_cast<std::ptrdiff_t>(layers.size());
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(layers.size()))
              throw nb::index_error();
            return layers[index].get();
          },
          nb::rv_policy::reference_internal)
      .def("__repr__",
           [](const Deep::SimpleConvPCNetwork& self)
           {
             return "<SimpleConvPCNetwork layers=" + std::to_string(self.GetLayers().size()) +
                    " batch_size=" + std::to_string(self.GetBatchSize()) + ">";
           })
      .def("train_step_with_projection", &Deep::SimpleConvPCNetwork::TrainStepWithProjection)
      .def("predict_with_projection", &Deep::SimpleConvPCNetwork::PredictWithProjection)
      .def("project_forward", &Deep::SimpleConvPCNetwork::ProjectForward);
}

