/**
 * @file UtilityBindings.cpp
 * @brief nanobind bindings for free-standing utilities (StreamAlignedBatcher,
 * activation functions, OpenMP/cache introspection). Split out of the former
 * monolithic pybinding.cpp -- see LayerBindings.cpp / NetworkBindings.cpp for
 * the rest.
 */
#include "BindingHelpers.h"
#include "UtilityBindings.h"

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
          nb::keep_alive<1, 4>())
      .def("num_batches_per_epoch", &Deep::StreamAlignedBatcher::NumBatchesPerEpoch)
      .def_prop_ro("batch_size", &Deep::StreamAlignedBatcher::GetBatchSize)
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
           });

  m.def("get_l2_cache_bytes", &Deep::GetL2CacheBytes);
  m.def("auto_batch_size", &Deep::AutoBatchSize);
  m.def("dynamic_thread", &Deep::DynamicThread, nb::arg("batch_size"));

  m.def("omp_max_threads", []() { return omp_get_max_threads(); });
  m.def("omp_num_procs", []() { return omp_get_num_procs(); });

  m.def("relu", [](FloatArray x) { Deep::relu(x.data(), x.size()); });
  m.def("drelu", [](FloatArray x) { Deep::dRelu(x.data(), x.size()); });
  m.def("tanh", [](FloatArray x) { Deep::tanh(x.data(), x.size()); });
  m.def("dtanh", [](FloatArray x) { Deep::dTanh(x.data(), x.size()); });
  m.def("sigmoid", [](FloatArray x) { Deep::sigmoid(x.data(), x.size()); });
  m.def("dsigmoid", [](FloatArray x) { Deep::dSigmoid(x.data(), x.size()); });
}

