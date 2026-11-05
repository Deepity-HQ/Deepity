#include "AttentionBindings.h"
#include "BindingHelpers.h"

#include <deepity/layers/AttentionOutputLayer.h>
#include <deepity/layers/AttentionScoresLayer.h>

#include <nanobind/stl/optional.h>

#include <optional>
#include <stdexcept>

namespace nb = nanobind;

namespace
{

void CheckSize(FloatArray a, size_t expected, const char* argName, const char* fn)
{
  if ((size_t)a.size() != expected)
    throw std::invalid_argument(std::string(fn) + ": " + argName + " expected " +
                                std::to_string(expected) + " values, got " +
                                std::to_string(a.size()));
}

template <typename T, typename Owner>
nb::ndarray<nb::numpy, T> ViewOwnedBy(T* data, std::initializer_list<size_t> shape, Owner& owner)
{
  return nb::ndarray<nb::numpy, T>(data, shape, nb::find(&owner));
}

} // namespace

void bind_attention_layers(nb::module_& m)
{
  nb::class_<Deep::AttentionScoresLayer>(
      m,
      "AttentionScoresLayer",
      "Causal single-head self-attention weights, settled via the categorical/KL PC "
      "energy (Pinchetti et al., arXiv:2211.03481). CPU-only v1 proof of concept; mu "
      "depends on Q and K beliefs together, so this is a free-standing PC node rather "
      "than a FullPCLayer toggle.")
      .def(nb::init<size_t, size_t, size_t>(),
           nb::arg("batch_size"),
           nb::arg("seq_len"),
           nb::arg("head_dim"),
           "Constructs an uninitialized attention-scores node for the given shape.")
      .def(
          "compute_mu_only",
          [](Deep::AttentionScoresLayer& self, FloatArray q, FloatArray k)
          {
            size_t qkElems = self.GetBatchSize() * self.GetSeqLen() * self.GetHeadDim();
            CheckSize(q, qkElems, "q_beliefs", "compute_mu_only");
            CheckSize(k, qkElems, "k_beliefs", "compute_mu_only");
            self.ComputeMuOnly(q.data(), k.data());
          },
          nb::arg("q_beliefs"),
          nb::arg("k_beliefs"),
          "q_beliefs/k_beliefs: shape [batch_size, seq_len, head_dim], flattened.")
      .def("calculate_state",
           &Deep::AttentionScoresLayer::CalculateState,
           nb::arg("need_energy"),
           "Calculates this node's internal state (prediction error against mu). "
           "@return Settling energy for this step if need_energy, else 0.")
      .def(
          "update_state",
          [](Deep::AttentionScoresLayer& self, float ir, std::optional<FloatArray> eAbove,
             std::optional<FloatArray> vBeliefs)
          {
            if (eAbove.has_value())
              CheckSize(*eAbove, self.GetBatchSize() * self.GetSeqLen() * self.GetHeadDim(),
                        "e_above", "update_state");
            if (vBeliefs.has_value())
              CheckSize(*vBeliefs, self.GetBatchSize() * self.GetSeqLen() * self.GetHeadDim(),
                        "v_beliefs", "update_state");
            self.UpdateState(ir,
                             eAbove.has_value() ? eAbove->data() : nullptr,
                             vBeliefs.has_value() ? vBeliefs->data() : nullptr);
          },
          nb::arg("ir"),
          nb::arg("e_above") = nb::none(),
          nb::arg("v_beliefs") = nb::none(),
          "e_above/v_beliefs: AttentionOutputLayer's error buffer and V's beliefs, "
          "shape [batch_size, seq_len, head_dim]; omit both when there's no layer above.")
      .def(
          "beliefs",
          [](Deep::AttentionScoresLayer& self)
          {
            return ViewOwnedBy(self.GetBeliefs(),
                               {self.GetBatchSize(), self.GetSeqLen(), self.GetSeqLen()}, self);
          },
          "Settled attention weights z, shape [batch_size, seq_len, seq_len].")
      .def(
          "errors",
          [](Deep::AttentionScoresLayer& self)
          {
            return ViewOwnedBy(const_cast<float*>(self.GetErrors()),
                               {self.GetBatchSize(), self.GetSeqLen(), self.GetSeqLen()}, self);
          },
          "Prediction error e, shape [batch_size, seq_len, seq_len].")
      .def_prop_ro("batch_size", &Deep::AttentionScoresLayer::GetBatchSize,
                   "Number of samples processed per call.")
      .def_prop_ro("seq_len", &Deep::AttentionScoresLayer::GetSeqLen, "Sequence length.")
      .def_prop_ro("head_dim", &Deep::AttentionScoresLayer::GetHeadDim,
                   "Attention head dimension.");

  nb::class_<Deep::AttentionOutputLayer>(
      m,
      "AttentionOutputLayer",
      "Attention's weighted-sum-with-V step: mu[b,i,:] = sum_j attnZ[b,i,j] * V[b,j,:], "
      "ordinary Gaussian PC energy, no learnable weights. CPU-only v1 proof of concept.")
      .def(nb::init<size_t, size_t, size_t>(),
           nb::arg("batch_size"),
           nb::arg("seq_len"),
           nb::arg("head_dim"),
           "Constructs an uninitialized attention-output node for the given shape.")
      .def(
          "compute_mu_only",
          [](Deep::AttentionOutputLayer& self, FloatArray attnBeliefs, FloatArray vBeliefs)
          {
            size_t bs = self.GetBatchSize(), sl = self.GetSeqLen(), hd = self.GetHeadDim();
            CheckSize(attnBeliefs, bs * sl * sl, "attn_beliefs", "compute_mu_only");
            CheckSize(vBeliefs, bs * sl * hd, "v_beliefs", "compute_mu_only");
            self.ComputeMuOnly(attnBeliefs.data(), vBeliefs.data());
          },
          nb::arg("attn_beliefs"),
          nb::arg("v_beliefs"),
          "attn_beliefs: AttentionScoresLayer's beliefs, shape [batch_size, seq_len, "
          "seq_len]. v_beliefs: shape [batch_size, seq_len, head_dim].")
      .def("calculate_state",
           &Deep::AttentionOutputLayer::CalculateState,
           nb::arg("need_energy"),
           "Calculates this node's internal state (prediction error against mu). "
           "@return Settling energy for this step if need_energy, else 0.")
      .def(
          "update_state",
          [](Deep::AttentionOutputLayer& self, float ir, std::optional<FloatArray> eAbove)
          {
            if (eAbove.has_value())
              CheckSize(*eAbove, self.GetBatchSize() * self.GetSeqLen() * self.GetHeadDim(),
                        "e_above", "update_state");
            self.UpdateState(ir, eAbove.has_value() ? eAbove->data() : nullptr);
          },
          nb::arg("ir"),
          nb::arg("e_above") = nb::none(),
          "Updates this node's beliefs. e_above: the layer above's error "
          "buffer, shape [batch_size, seq_len, head_dim]; omit when there's "
          "no layer above.")
      .def(
          "beliefs",
          [](Deep::AttentionOutputLayer& self)
          {
            return ViewOwnedBy(self.GetBeliefs(),
                               {self.GetBatchSize(), self.GetSeqLen(), self.GetHeadDim()}, self);
          },
          "Belief z, shape [batch_size, seq_len, head_dim].")
      .def(
          "errors",
          [](Deep::AttentionOutputLayer& self)
          {
            return ViewOwnedBy(const_cast<float*>(self.GetErrors()),
                               {self.GetBatchSize(), self.GetSeqLen(), self.GetHeadDim()}, self);
          },
          "Prediction error e, shape [batch_size, seq_len, head_dim].")
      .def_prop_ro("batch_size", &Deep::AttentionOutputLayer::GetBatchSize,
                   "Number of samples processed per call.")
      .def_prop_ro("seq_len", &Deep::AttentionOutputLayer::GetSeqLen, "Sequence length.")
      .def_prop_ro("head_dim", &Deep::AttentionOutputLayer::GetHeadDim,
                   "Attention head dimension.");
}
