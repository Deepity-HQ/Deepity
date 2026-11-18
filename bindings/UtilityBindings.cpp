/**
 * @file UtilityBindings.cpp
 * @brief nanobind bindings for free-standing utilities (StreamAlignedBatcher,
 * activation functions, OpenMP/cache introspection). Split out of the former
 * monolithic pybinding.cpp, see LayerBindings.cpp / NetworkBindings.cpp for
 * the rest.
 */
#include "UtilityBindings.h"
#include "BindingHelpers.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <omp.h>

#include <optional>
#include <random>
#include <string>

#include <deepity/backend/Backend.h>
#include <deepity/io/SafeTensors.h>
#include <deepity/utils/Activations.h>
#include <deepity/utils/ModelIO.h>
#include <deepity/utils/Optimize.h>
#include <deepity/utils/RandomizeDistribution.h>
#include <deepity/utils/StreamAlignedBatcher.h>

namespace
{
/// @brief One tensor as a numpy array. Zero-copy (a read-only view that keeps the file mapped)
/// when the file is mapped and the bytes are aligned; otherwise a private copy.
template <typename T>
nb::object TensorToNumpy(const Deep::io::SafeTensorsReader& reader, const Deep::io::TensorEntry& e)
{
  const void* bytes = reader.Bytes(e);
  if (e.shape.size() > 8)
    throw nb::value_error(("tensor '" + e.name + "' has more than 8 dimensions").c_str());
  size_t dims[8];
  for (size_t i = 0; i < e.shape.size(); ++i)
    dims[i] = e.shape[i];
  if (bytes != nullptr && reinterpret_cast<uintptr_t>(bytes) % alignof(T) == 0)
  {
    auto mapping = new std::shared_ptr<const Deep::io::MappedFile>(reader.Mapping(e));
    nb::capsule owner(mapping, [](void* p) noexcept { delete static_cast<std::shared_ptr<const Deep::io::MappedFile>*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, const T>(static_cast<const T*>(bytes), e.shape.size(), dims, owner));
  }
  auto* copy = new T[std::max<size_t>(e.ByteSize() / sizeof(T), 1)];
  reader.Read(e, copy, e.ByteSize());
  nb::capsule owner(copy, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::cast(nb::ndarray<nb::numpy, T>(copy, e.shape.size(), dims, owner));
}

nb::dict ReadTensors(const std::string& path, bool verifyHash, bool mmap)
{
  using namespace Deep::io;
  SafeTensorsReader reader(path, verifyHash, mmap);
  nb::dict out;
  for (const auto& e : reader.Tensors())
  {
    nb::object a;
    if (e.packed || e.dtype == "BF16" || e.dtype == "F16")
      throw nb::value_error(("tensor '" + e.name + "' has dtype " + e.dtype + ", which numpy cannot hold here; "
                             "use ModelIO to load it into a network").c_str());
    else if (e.dtype == "F32") a = TensorToNumpy<float>(reader, e);
    else if (e.dtype == "F64") a = TensorToNumpy<double>(reader, e);
    else if (e.dtype == "I64") a = TensorToNumpy<int64_t>(reader, e);
    else if (e.dtype == "I32") a = TensorToNumpy<int32_t>(reader, e);
    else if (e.dtype == "I16") a = TensorToNumpy<int16_t>(reader, e);
    else if (e.dtype == "I8") a = TensorToNumpy<int8_t>(reader, e);
    else if (e.dtype == "U64") a = TensorToNumpy<uint64_t>(reader, e);
    else if (e.dtype == "U32") a = TensorToNumpy<uint32_t>(reader, e);
    else if (e.dtype == "U16") a = TensorToNumpy<uint16_t>(reader, e);
    else if (e.dtype == "U8" || e.dtype == "BOOL") a = TensorToNumpy<uint8_t>(reader, e);
    out[e.name.c_str()] = a;
  }
  return out;
}

/// @brief Fills `x` in place via Deep::RandomizeWithDistribution(), picking
/// the backend (and therefore the RNG: OpenMP-parallel Mersenne Twister on
/// CPU, curand/host-fallback on GPU -- see CUDABackendT's
/// RandomizeNormal/RandomizeUniform) from `device`. `x` is always a host
/// numpy array regardless of `device`; the GPU path randomizes into a
/// throwaway device buffer and copies it back, so the caller gets the same
/// fast RNG either way without having to manage a device tensor themselves.
void RandomizeHelper(FloatArray x, std::optional<uint32_t> seed, const std::string& distribution,
                     const std::string& device)
{
  uint32_t backendSeed = seed.has_value() ? seed.value() : std::random_device{}();
  Deep::DeviceType dt = (device == "cuda" || device == "gpu") ? Deep::DeviceType::DEVICE_GPU
                                                               : Deep::DeviceType::DEVICE_CPU;
  auto backend = Deep::CreateBackend(dt);
  size_t n = x.size();

  if (dt == Deep::DeviceType::DEVICE_CPU)
  {
    Deep::RandomizeWithDistribution(backend.get(), x.data(), n, distribution.c_str(), backendSeed,
                                    0.0f, 1.0f);
    return;
  }

  float* deviceBuf = backend->Allocate(n);
  Deep::RandomizeWithDistribution(backend.get(), deviceBuf, n, distribution.c_str(), backendSeed,
                                  0.0f, 1.0f);
  backend->CopyToHost(x.data(), deviceBuf, n);
  backend->Free(deviceBuf);
}
} // namespace

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

  m.def("randomize", &RandomizeHelper, nb::arg("x"), nb::arg("seed") = nb::none(),
        nb::arg("distribution") = "normal", nb::arg("device") = "cpu",
        "Fills a numpy float32 array `x` in place with random values, using "
        "Deepity's own fast RNG (the same one every layer's randomize_weights() "
        "uses) instead of numpy's -- so this module can seed weights/data "
        "without a separate randomization dependency. `seed` gives a "
        "reproducible run (None uses a nondeterministic seed). `distribution` "
        "is \"normal(mean,stddev)\" or \"uniform(min,max)\"; anything else, "
        "including the bare words \"normal\"/\"uniform\" with no parentheses, "
        "falls back to a standard normal(0,1). `device` is \"cpu\" or \"cuda\" "
        "(GPU randomizes into a device buffer and copies the result back; "
        "worthwhile once `x` has on the order of 1e5+ elements, see issue #67).");

  m.def("read_tensors", &ReadTensors, nb::arg("filepath"), nb::arg("verify_hash") = true, nb::arg("mmap") = true,
        "Reads every tensor of a safetensors file into a dict of numpy arrays in one call. With "
        "mmap the arrays are read-only views of the mapped file (no copy, nothing read until "
        "touched); the mapping stays alive as long as any array does. F16, BF16 and packed 4-bit "
        "tensors raise; load those into a network instead.");
  m.def("read_model_metadata", &Deep::ModelIO::ReadMetadata, nb::arg("filepath"),
        "Validates a model file and returns its string metadata (architecture, inference defaults, "
        "format version) without reading the weights. Raises on a missing or corrupt file.");
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
