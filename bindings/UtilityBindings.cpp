/**
 * @file UtilityBindings.cpp
 * @brief nanobind bindings for free-standing utilities (StreamAlignedBatcher,
 * activation functions, OpenMP/cache introspection). Split out of the former
 * monolithic pybinding.cpp, see LayerBindings.cpp / NetworkBindings.cpp for
 * the rest.
 */
#include "UtilityBindings.h"
#include "BindingHelpers.h"

#include <omp.h>

#include <deepity/utils/Activations.h>
#include <deepity/utils/Optimize.h>
#include <deepity/utils/StreamAlignedBatcher.h>

// ============================================================================
// Utilities Bindings
// ============================================================================
void bind_utilities(nb::module_& m)
{
  nb::class_<Deep::StreamAlignedBatcher>(
      m, "StreamAlignedBatcher", "Builds batches with a fixed number of examples per class.")
      .def(
          "__init__",
          [](Deep::StreamAlignedBatcher* self,
             FloatArray X,
             FloatArray Y,
             IntArray labels,
             size_t x_stride,
             size_t y_stride,
             int num_classes,
             int per_class,
             unsigned seed)
          {
            size_t num_samples = (size_t)X.shape(0);
            new (self) Deep::StreamAlignedBatcher(X.data(),
                                                  Y.data(),
                                                  labels.data(),
                                                  num_samples,
                                                  x_stride,
                                                  y_stride,
                                                  num_classes,
                                                  per_class,
                                                  seed);
          },
          nb::arg("X"),
          nb::arg("Y"),
          nb::arg("labels"),
          nb::arg("x_stride"),
          nb::arg("y_stride"),
          nb::arg("num_classes"),
          nb::arg("per_class"),
          nb::arg("seed") = 42,
          nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(),
          nb::keep_alive<1, 4>(),
          "Does not own or copy X/Y/labels; the caller must keep them alive "
          "for this batcher's lifetime. `seed` controls per-class shuffling.")
      .def("num_batches_per_epoch", &Deep::StreamAlignedBatcher::NumBatchesPerEpoch,
           "Number of complete batches obtainable per epoch, limited by the "
           "rarest class's sample count.")
      .def_prop_ro("batch_size", &Deep::StreamAlignedBatcher::GetBatchSize,
                   "Rows per batch: num_classes * per_class.")
      .def("get_batch",
           [](Deep::StreamAlignedBatcher& self)
           {
             int bsz = self.GetBatchSize();
             size_t xstride = self.GetXStride();
             size_t ystride = self.GetYStride();

             float* xdata = new float[(size_t)bsz * xstride];
             float* ydata = new float[(size_t)bsz * ystride];
             int* ldata = new int[(size_t)bsz];

             self.GetBatch(xdata, ydata, ldata);

             nb::capsule xowner(xdata, [](void* p) noexcept { delete[] static_cast<float*>(p); });
             nb::capsule yowner(ydata, [](void* p) noexcept { delete[] static_cast<float*>(p); });
             nb::capsule lowner(ldata, [](void* p) noexcept { delete[] static_cast<int*>(p); });

             nb::ndarray<nb::numpy, float> X_out(xdata, {(size_t)bsz, xstride}, xowner);
             nb::ndarray<nb::numpy, float> Y_out(ydata, {(size_t)bsz, ystride}, yowner);
             nb::ndarray<nb::numpy, int> labels_out(ldata, {(size_t)bsz}, lowner);

             return nb::make_tuple(X_out, Y_out, labels_out);
           },
           "Returns the next batch as (X, Y, labels), each row-major with "
           "rows grouped by class: rows [0, per_class) are class 0, "
           "[per_class, 2*per_class) are class 1, etc.");

  m.def("get_l2_cache_bytes", &Deep::GetL2CacheBytes,
        "Queries the size of this CPU's L2 cache (per-core, level 2), "
        "falling back to 512 KiB on platforms where it can't be queried.");
  m.def("auto_batch_size", &Deep::AutoBatchSize,
        "Picks a default batch size (a power of 2 clamped to [64, 512]) "
        "sized to fit comfortably in this CPU's L2 cache.");
  m.def("dynamic_thread", &Deep::DynamicThread, nb::arg("batch_size"),
        "Sets OpenMP's (and the BLAS library's) thread count for the given "
        "batch size. Called automatically by every network's train_step()/predict().");

  m.def("omp_max_threads", []() { return omp_get_max_threads(); },
        "Returns OpenMP's current max-threads setting.");
  m.def("omp_num_procs", []() { return omp_get_num_procs(); },
        "Returns the number of logical processors OpenMP sees.");

  m.def("relu", [](FloatArray x) { Deep::relu(x.data(), x.size()); },
        "In-place ReLU activation: x[i] = max(0, x[i]).");
  m.def("drelu", [](FloatArray x) { Deep::dRelu(x.data(), x.size()); },
        "In-place ReLU derivative: 1 where x > 0, else 0.");
  m.def("tanh", [](FloatArray x) { Deep::tanh(x.data(), x.size()); },
        "In-place hyperbolic tangent activation.");
  m.def("dtanh", [](FloatArray x) { Deep::dTanh(x.data(), x.size()); },
        "In-place tanh derivative: 1 - tanh(x)^2.");
  m.def("sigmoid", [](FloatArray x) { Deep::sigmoid(x.data(), x.size()); },
        "In-place logistic sigmoid activation: x[i] = 1 / (1 + e^-x[i]).");
  m.def("dsigmoid", [](FloatArray x) { Deep::dSigmoid(x.data(), x.size()); },
        "In-place logistic sigmoid derivative: sigmoid(x)*(1-sigmoid(x)).");
}
