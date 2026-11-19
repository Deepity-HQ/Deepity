/**
 * @file tModelIOVerify.cpp
 * @brief Round-trip regression test for ModelIO::Save()/Load().
 *
 * Saves a randomly initialised DirectKPPCNetwork, loads the file into a
 * differently initialised twin, and checks every W, b and Psi value matches.
 * The layer sizes deliberately differ from one layer to the next: a Psi
 * buffer is [size, terminal], so a Save that treats it as [nextSize, terminal]
 * drops rows and only shows up when size != nextSize.
 */
#include <cstdio>
#include <cstring>
#include <deepity/backend/Backend.h>
#include <deepity/networks/ConvPCNetwork.h>
#include <deepity/networks/ConvPCNetworkT.h>
#include <deepity/networks/DirectKPPCNetwork.h>
#include <deepity/networks/FullConvPCNetworkT.h>
#include <deepity/networks/DirectKPPCNetworkT.h>
#include <deepity/networks/DiscriminativePCNetwork.h>
#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <deepity/networks/GaussSeidelPCNetworkT.h>
#include <deepity/networks/SimpleConvPCNetwork.h>
#include <deepity/networks/SimpleConvPCNetworkT.h>
#include <deepity/networks/SimplePCNetwork.h>
#include <deepity/networks/SimplePCNetworkT.h>
#include <deepity/networks/FullConvPCNetwork.h>
#include <deepity/networks/FullPCNetwork.h>
#include <deepity/networks/FullPCNetworkT.h>
#include <deepity/utils/ModelIO.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>

using namespace Deep;

namespace
{
constexpr int kBatch = 2;
const ActivationType kAct = ActivationType::RELU, kDAct = ActivationType::dRELU;
const ActivationType kLin = ActivationType::LINEAR, kDLin = ActivationType::dLINEAR;

/// @brief Selects Adam for W/b (and Psi where the network has it); checkpoints must carry its state.
template <typename Net> void SetAdam(Net& n)
{
  n.SetOptimizer(OptimizerType::ADAM);
  if constexpr (requires { n.SetPsiOptimizer(OptimizerType::ADAM); })
    n.SetPsiOptimizer(OptimizerType::ADAM);
}

std::unique_ptr<DirectKPPCNetwork> MakeNet(unsigned seed)
{
  auto n = std::make_unique<DirectKPPCNetwork>(kBatch);
  // size, next, terminal, lr, ir, fl, lmbda, act, dact
  n->AddLayer(8, 6, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n->AddLayer(6, 4, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 0, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

template <typename T>
bool SameRows(const T* a, const T* b, size_t rows, size_t cols, size_t stride, size_t* firstBad)
{
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c)
      if (a[r * stride + c] != b[r * stride + c])
      {
        *firstBad = r;
        return false;
      }
  return true;
}
std::unique_ptr<FullPCNetwork> MakeFullNet(unsigned seed)
{
  auto n = std::make_unique<FullPCNetwork>(kBatch);
  n->AddLayer(8, 6, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n->AddLayer(6, 4, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 0, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<FullPCNetworkT<double, double>> MakeDoubleNet(unsigned seed)
{
  auto n = std::make_unique<FullPCNetworkT<double, double>>(kBatch);
  n->AddLayer(8, 6, 4, 1e-3, 0.1, 1e-4, 1e-4, kAct, kDAct);
  n->AddLayer(6, 4, 4, 1e-3, 0.1, 1e-4, 1e-4, kAct, kDAct);
  n->AddLayer(4, 0, 4, 1e-3, 0.1, 1e-4, 1e-4, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<FullConvPCNetwork> MakeConvNet(unsigned seed)
{
  auto n = std::make_unique<FullConvPCNetwork>(kBatch);
  // in, out, H, W, kH, kW, strideH, strideW, padH, padW, terminal, lr, ir, fl, lmbda, act, dact
  n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 10, 1e-3f, 0.1f, 0.0f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 10, 8, 8, 8, 8, 1, 1, 0, 0, 10, 1e-3f, 0.1f, 0.0f, 1e-4f, kAct, kDAct);
  n->AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 10, 1e-3f, 0.1f, 0.0f, 1e-4f, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

template <typename Q> std::unique_ptr<FullPCNetworkT<Q, float>> MakeQuant(unsigned seed)
{
  auto n = std::make_unique<FullPCNetworkT<Q, float>>(kBatch);
  n->AddLayer(8, 7, 3, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n->AddLayer(7, 3, 3, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n->AddLayer(3, 0, 3, 1e-3f, 0.1f, 1e-4f, 1e-4f, kLin, kDLin);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

/// @brief A 4-bit network must save as packed U8 (about half the bytes) and load back bit-exactly.
template <typename Q> bool QuantRoundTrips(const char* name, const std::string& path)
{
  auto a = MakeQuant<Q>(21), b = MakeQuant<Q>(22);
  ModelIO::Save(*a, path);
  ModelIO::Load(*b, path);
  bool same = true;
  size_t tensors = 0;
  for (size_t i = 0; i < a->GetLayers().size(); ++i)
  {
    detail::SyncFromDeviceIfSupported(a->GetLayers()[i]);
    detail::SyncFromDeviceIfSupported(b->GetLayers()[i]);
    const auto da = a->GetLayers()[i]->GetStateDict(), db = b->GetLayers()[i]->GetStateDict();
    for (const auto& [key, t] : da)
    {
      const auto [rows, width] = detail::RowsAndWidth(t.shape);
      same &= std::memcmp(t.data, db.at(key).data, rows * width * sizeof(Q)) == 0;
      ++tensors;
    }
  }
  const io::SafeTensorsReader reader(path);
  bool packedOnDisk = reader.Metadata().count(io::kDTypesKey) == 1;
  uint64_t diskBytes = 0, memBytes = 0;
  for (const auto& t : reader.Tensors())
  {
    packedOnDisk &= t.dtype == DTypeTag<Q>::Name() && t.packed;
    diskBytes += t.ByteSize();
    size_t elems = 1;
    for (size_t d : t.shape)
      elems *= d;
    memBytes += elems * sizeof(Q);
  }
  const bool smaller = diskBytes < memBytes;
  printf("  %s: %zu tensors, %llu bytes on disk vs %llu in memory\n", name, tensors, (unsigned long long)diskBytes,
         (unsigned long long)memBytes);
  printf("%s\n", same && packedOnDisk && smaller ? "PASSED: packed 4-bit round trip" : "FAILED: packed 4-bit round trip");
  return same && packedOnDisk && smaller;
}

std::unique_ptr<DirectKPPCNetworkT<double, double>> MakeDkpDouble(unsigned seed)
{
  auto n = std::make_unique<DirectKPPCNetworkT<double, double>>(kBatch);
  n->AddLayer(8, 6, 4, 1e-3, 0.1, 1e-4, 1e-4, kAct, kDAct);
  n->AddLayer(6, 4, 4, 1e-3, 0.1, 1e-4, 1e-4, kAct, kDAct);
  n->AddLayer(4, 0, 4, 1e-3, 0.1, 1e-4, 1e-4, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<GaussSeidelPCNetwork> MakeGaussSeidel(unsigned seed)
{
  auto n = std::make_unique<GaussSeidelPCNetwork>(kBatch);
  n->AddLayer(8, 6, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
  n->AddLayer(6, 4, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 0, 1e-3f, 0.1f, 1e-4f, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<GaussSeidelPCNetworkT<double, double>> MakeGaussSeidelDouble(unsigned seed)
{
  auto n = std::make_unique<GaussSeidelPCNetworkT<double, double>>(kBatch);
  n->AddLayer(8, 6, 1e-3, 0.1, 1e-4, kAct, kDAct);
  n->AddLayer(6, 4, 1e-3, 0.1, 1e-4, kAct, kDAct);
  n->AddLayer(4, 0, 1e-3, 0.1, 1e-4, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<DiscriminativePCNetwork> MakeDiscriminative(unsigned seed)
{
  auto n = std::make_unique<DiscriminativePCNetwork>(kBatch);
  // size, next, lr, ir, pr, lmbda, act, dact
  n->AddLayer(8, 6, 1e-3f, 0.1f, 0.01f, 1e-4f, kAct, kDAct);
  n->AddLayer(6, 4, 1e-3f, 0.1f, 0.01f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 0, 1e-3f, 0.1f, 0.01f, 1e-4f, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

/// @brief A layer that keeps Layer's default (empty) GetStateDict, standing in for a layer family
/// with no weights to expose.
struct NoStateLayer : Layer
{
  float* GetBeliefs() noexcept override { return nullptr; }
  const float* GetErrors() const noexcept override { return nullptr; }
  size_t GetInputSize() const noexcept override { return 4; }
  size_t GetOutputSize() const noexcept override { return 4; }
  size_t GetBatchSize() const noexcept override { return 1; }
  float CalculateState() noexcept override { return 0.f; }
  void UpdateState() noexcept override {}
  void UpdateWeights() noexcept override {}
};

struct NoStateNet
{
  std::vector<std::unique_ptr<Layer>> layers;
  NoStateNet() { layers.push_back(std::make_unique<NoStateLayer>()); }
  const std::vector<std::unique_ptr<Layer>>& GetLayers() const { return layers; }
};

std::unique_ptr<ConvPCNetworkT<double, double>> MakeConvPCDouble(unsigned seed)
{
  auto n = std::make_unique<ConvPCNetworkT<double, double>>(kBatch);
  n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1e-3, 0.1, 0.01, 1e-4, kAct, kDAct);
  n->AddLayer(4, 10, 8, 8, 8, 8, 1, 1, 0, 0, 1e-3, 0.1, 0.01, 1e-4, kAct, kDAct);
  n->AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1e-3, 0.1, 0.01, 1e-4, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<FullConvPCNetworkT<double, double>> MakeFullConvDouble(unsigned seed)
{
  auto n = std::make_unique<FullConvPCNetworkT<double, double>>(kBatch);
  n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 10, 1e-3, 0.1, 0.0, 1e-4, kAct, kDAct);
  n->AddLayer(4, 10, 8, 8, 8, 8, 1, 1, 0, 0, 10, 1e-3, 0.1, 0.0, 1e-4, kAct, kDAct);
  n->AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 10, 1e-3, 0.1, 0.0, 1e-4, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<SimplePCNetwork> MakeSimple(unsigned seed)
{
  auto n = std::make_unique<SimplePCNetwork>(kBatch);
  n->AddLayer(8, 6, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
  n->AddLayer(6, 4, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 0, 1e-3f, 0.1f, 1e-4f, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<SimplePCNetworkT<double, double>> MakeSimpleDouble(unsigned seed)
{
  auto n = std::make_unique<SimplePCNetworkT<double, double>>(kBatch);
  n->AddLayer(8, 6, 1e-3, 0.1, 1e-4, kAct, kDAct);
  n->AddLayer(6, 4, 1e-3, 0.1, 1e-4, kAct, kDAct);
  n->AddLayer(4, 0, 1e-3, 0.1, 1e-4, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<SimpleConvPCNetwork> MakeSimpleConv(unsigned seed)
{
  auto n = std::make_unique<SimpleConvPCNetwork>(kBatch);
  n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 10, 8, 8, 8, 8, 1, 1, 0, 0, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
  n->AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1e-3f, 0.1f, 1e-4f, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<SimpleConvPCNetworkT<double, double>> MakeSimpleConvDouble(unsigned seed)
{
  auto n = std::make_unique<SimpleConvPCNetworkT<double, double>>(kBatch);
  n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1e-3, 0.1, 1e-4, kAct, kDAct);
  n->AddLayer(4, 10, 8, 8, 8, 8, 1, 1, 0, 0, 1e-3, 0.1, 1e-4, kAct, kDAct);
  n->AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1e-3, 0.1, 1e-4, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<ConvPCNetwork> MakeConvPC(unsigned seed)
{
  auto n = std::make_unique<ConvPCNetwork>(kBatch);
  // in, out, H, W, kH, kW, strideH, strideW, padH, padW, lr, ir, pr, lmbda, act, dact
  n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1e-3f, 0.1f, 0.01f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 10, 8, 8, 8, 8, 1, 1, 0, 0, 1e-3f, 0.1f, 0.01f, 1e-4f, kAct, kDAct);
  n->AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1e-3f, 0.1f, 0.01f, 1e-4f, kLin, kDLin);
  SetAdam(*n);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

/// @brief Compares every tensor of every layer's state dict, honouring row padding.
template <typename Net> bool SameState(const Net& a, const Net& b)
{
  bool ok = true;
  for (size_t i = 0; i < a.GetLayers().size(); ++i)
  {
    // Device-backed layers expose staging buffers that must be refreshed before reading.
    detail::SyncFromDeviceIfSupported(a.GetLayers()[i]);
    detail::SyncFromDeviceIfSupported(b.GetLayers()[i]);
    const auto da = a.GetLayers()[i]->GetStateDict();
    const auto db = b.GetLayers()[i]->GetStateDict();
    for (const auto& [name, t] : da)
    {
      const auto& u = db.at(name);
      const auto [rows, width] = detail::RowsAndWidth(t.shape);
      const size_t stride = t.rowStride ? t.rowStride : width;
      size_t bad = 0;
      const bool same = t.shape == u.shape && SameRows(t.data, u.data, rows, width, stride, &bad);
      if (!same)
        printf("  layer %zu tensor %s differs\n", i, name.c_str());
      ok &= same;
    }
  }
  return ok;
}

bool Throws(auto&& fn)
{
  try
  {
    fn();
  }
  catch (const std::exception&)
  {
    return true;
  }
  return false;
}

/// @brief A model split over several shards must load as one, resume as a checkpoint, and reject
/// an index that lies about its shards.
bool ShardingWorks(const std::string& dir)
{
  namespace fs = std::filesystem;
  fs::create_directories(dir);
  const std::string index = dir + "/m.safetensors.index.json";
  auto a = MakeNet(41), b = MakeNet(42);
  ModelIO::SaveSharded(*a, index, 200);
  size_t shardFiles = 0;
  for (const auto& e : fs::directory_iterator(dir))
    shardFiles += e.path().extension() == ".safetensors";
  ModelIO::Load(*b, index);
  const bool loads = shardFiles > 1 && SameState(*a, *b);
  // Scoped: Windows cannot delete shard files while a reader has them mapped.
  const bool oneReader = [&]
  {
    const io::SafeTensorsReader reader(index);
    return reader.ShardCount() == shardFiles && ModelIO::ReadMetadata(index).count("deepity.kind") == 1;
  }();

  // Moving the index to a directory with no shards must fail, not guess.
  auto indexText = [&] { std::ifstream f(index); return std::string((std::istreambuf_iterator<char>(f)), {}); }();
  auto writeIndex = [&](const std::string& name, const std::string& text) { std::ofstream(dir + "/" + name) << text; return dir + "/" + name; };
  const bool missingShard = Throws([&] { io::SafeTensorsReader r(writeIndex("gone.safetensors.index.json",
                                                                              std::string(indexText).replace(indexText.find("m-00001"), 7, "x-00001"))); });
  const bool escapes = Throws([&] { io::SafeTensorsReader r(writeIndex("esc.safetensors.index.json",
                                                                         R"({"weight_map":{"a":"../../etc/passwd"}})")); });
  const bool empty = Throws([&] { io::SafeTensorsReader r(writeIndex("none.safetensors.index.json", R"({"weight_map":{}})")); });
  const bool lies = Throws([&]
                           {
                             std::string t = indexText;
                             const size_t at = t.find("m-00002");
                             t.replace(at, 7, "m-00001");
                             io::SafeTensorsReader r(writeIndex("lie.safetensors.index.json", t));
                           });

  auto c = MakeNet(43), d = MakeNet(44);
  ModelIO::SaveCheckpointSharded(*c, dir + "/c.safetensors.index.json", 400);
  const bool checkpointLoads = !Throws([&] { ModelIO::LoadCheckpoint(*d, dir + "/c.safetensors.index.json"); }) && SameState(*c, *d);
  fs::remove_all(dir);
  const bool ok = loads && oneReader && missingShard && escapes && empty && lies && checkpointLoads;
  printf("  %zu shards: loads=%d oneReader=%d missingShard=%d escapes=%d empty=%d lies=%d checkpoint=%d\n", shardFiles, loads, oneReader,
         missingShard, escapes, empty, lies, checkpointLoads);
  printf("%s\n", ok ? "PASSED: sharded save, load, checkpoint and index validation" : "FAILED: sharding");
  return ok;
}


/// @brief Foreign tensor names: a prefixed copy of a saved file loads through `rename`, and
/// strict mode reports every name that does not line up.
bool LoadWeightsWorks(const std::string& path)
{
  auto a = MakeNet(31);
  ModelIO::Save(*a, path);
  const std::string foreign = path + ".foreign";
  {
    io::SafeTensorsReader reader(path);
    std::vector<std::vector<char>> bufs;
    io::SafeTensorsWriter w;
    for (const auto& t : reader.Tensors())
    {
      size_t elems = 1;
      for (size_t d : t.shape)
        elems *= d;
      bufs.emplace_back(t.ByteSize());
      reader.Read(t, bufs.back().data(), bufs.back().size());
      w.AddTensor("model." + t.name, t.dtype, t.shape, bufs.back().data(), bufs.back().size());
    }
    const float junk[1] = {0.f};
    w.AddTensor("junk", "F32", {1}, junk, sizeof(junk));
    w.Write(foreign);
  }
  auto strip = [](const std::string& n) { return n.rfind("model.", 0) == 0 ? n.substr(6) : n; };

  auto b = MakeNet(32);
  const bool strictThrows = Throws([&] { ModelIO::LoadWeights(*b, foreign, true, strip); });
  auto c = MakeNet(32);
  const auto lenient = ModelIO::LoadWeights(*c, foreign, false, strip);
  const bool reported = lenient.missing.empty() && lenient.unexpected == std::vector<std::string>{"junk"};
  const bool loaded = SameState(*a, *c);
  auto d = MakeNet(32);
  const auto unrenamed = ModelIO::LoadWeights(*d, foreign, false);
  const bool allMissing = !unrenamed.missing.empty() && !SameState(*a, *d);
  const bool ok = strictThrows && reported && loaded && allMissing;
  printf("%s\n", ok ? "PASSED: LoadWeights renames, reports and enforces strict" : "FAILED: LoadWeights");
  return ok;
}

/// @brief Copies a model file with its metadata edited: `edit` may change or erase keys, and
/// `extraTensor` adds an unknown tensor. The hash is dropped (it is rewritten for the copy).
void RewriteMeta(const std::string& src, const std::string& dst,
                 const std::function<void(std::map<std::string, std::string>&)>& edit, bool extraTensor,
                 const std::function<bool(const std::string&)>& skip = {})
{
  io::SafeTensorsReader reader(src);
  std::map<std::string, std::string> meta = reader.Metadata();
  meta.erase(io::kHashKey);
  meta.erase(io::kDTypesKey);
  edit(meta);
  io::SafeTensorsWriter w;
  for (const auto& [k, v] : meta)
    w.SetMetadata(k, v);
  std::vector<std::vector<char>> bufs;
  bufs.reserve(reader.Tensors().size());
  for (const auto& t : reader.Tensors())
  {
    if (skip && skip(t.name)) continue;
    bufs.emplace_back(t.ByteSize());
    reader.Read(t, bufs.back().data(), bufs.back().size());
    w.AddTensor(t.name, t.dtype, t.shape, bufs.back().data(), bufs.back().size());
  }
  const float extra[2] = {1.f, 2.f};
  if (extraTensor) w.AddTensor("from_the_future.extra", "F32", {2}, extra, sizeof(extra));
  w.Write(dst);
}

std::string ErrorText(auto&& fn)
{
  try
  {
    fn();
  }
  catch (const std::exception& e)
  {
    return e.what();
  }
  return {};
}

/// @brief The version rule: a newer format is refused with both versions named, a malformed version is
/// refused, and everything else unknown (keys, tensors, a missing version) is tolerated.
bool FormatVersionRules(const std::string& path)
{
  auto a = MakeNet(41);
  SetAdam(*a);
  ModelIO::Save(*a, path);
  const std::string ckpt = path + ".ckpt", edited = path + ".edited";
  ModelIO::SaveCheckpoint(*a, ckpt);
  auto setVersion = [](const char* v) { return [v](auto& m) { m[io::kFormatVersionKey] = v; }; };
  bool ok = true;

  RewriteMeta(path, edited, setVersion("2"), false);
  auto b = MakeNet(42);
  const std::string text = ErrorText([&] { ModelIO::Load(*b, edited); });
  ok &= text.find("version 2") != std::string::npos && text.find("up to version 1") != std::string::npos;
  ok &= Throws([&] { ModelIO::LoadWeights(*b, edited); }) && Throws([&] { ModelIO::ReadMetadata(edited); });
  RewriteMeta(ckpt, edited, setVersion("2"), false);
  ok &= Throws([&] { ModelIO::LoadCheckpoint(*b, edited); });
  printf("%s\n", ok ? "PASSED: a newer format version is refused by every load path" : "FAILED: newer version accepted");

  bool malformed = true;
  for (const char* bad : {"", "0", "-1", "abc", "1.0", "1 ", "99999999999"})
  {
    RewriteMeta(path, edited, setVersion(bad), false);
    malformed &= ErrorText([&] { ModelIO::Load(*b, edited); }).find("invalid") != std::string::npos;
  }
  printf("%s\n", malformed ? "PASSED: a malformed format version is refused" : "FAILED: malformed version accepted");
  ok &= malformed;

  RewriteMeta(path, edited, [](auto& m) { m["deepity.from_the_future"] = "{\"anything\":1}"; }, true);
  auto c = MakeNet(43);
  const bool tolerant = !Throws([&] { ModelIO::Load(*c, edited); }) && SameState(*a, *c) &&
                        ModelIO::ReadMetadata(edited).count("deepity.from_the_future") == 1;
  printf("%s\n", tolerant ? "PASSED: unknown metadata keys and unknown tensors are ignored" : "FAILED: unknown fields rejected");
  ok &= tolerant;

  RewriteMeta(path, edited, [](auto& m) { m.erase(io::kFormatVersionKey); }, false);
  auto d = MakeNet(44);
  const bool unversioned = !Throws([&] { ModelIO::Load(*d, edited); }) && SameState(*a, *d);
  printf("%s\n", unversioned ? "PASSED: a file with no format version still loads" : "FAILED: unversioned file rejected");
  ok &= unversioned;

  RewriteMeta(path, edited, setVersion("1"), false);
  auto e = MakeNet(45);
  const bool current = !Throws([&] { ModelIO::Load(*e, edited); }) && SameState(*a, *e);
  printf("%s\n", current ? "PASSED: the current format version loads" : "FAILED: version 1 rejected");
  ok &= current;
  for (const auto& f : {ckpt, edited})
    std::filesystem::remove(f);
  return ok;
}

/// @brief An inference artifact leaves out Psi: smaller, still loads, W and b match, Psi stays the
/// fresh initialization, and a file that is merely missing a tensor (no marker) is still an error.
bool InferenceArtifact(const std::string& path)
{
  auto a = MakeNet(51);
  SetAdam(*a);
  const std::string slim = path + ".slim", full = path + ".full", cut = path + ".cut";
  ModelIO::Save(*a, full);
  ModelIO::Save(*a, slim, {{io::kTrainingTensorsKey, "omitted"}});
  auto countTraining = [](const std::string& file)
  {
    io::SafeTensorsReader r(file);
    size_t n = 0;
    for (const auto& t : r.Tensors())
      n += io::IsTrainingOnlyTensor(t.name);
    return n;
  };
  bool ok = true;
  const bool contents = countTraining(full) > 0 && countTraining(slim) == 0 &&
                        std::filesystem::file_size(slim) < std::filesystem::file_size(full) &&
                        io::SafeTensorsReader(slim).Metadata().at("deepity.kind") == "inference" &&
                        io::SafeTensorsReader(slim).Metadata().at(io::kTrainingTensorsKey) == "omitted";
  printf("%s\n", contents ? "PASSED: an inference artifact has no Psi and is smaller" : "FAILED: inference artifact contents");
  ok &= contents;

  auto b = MakeNet(52), fresh = MakeNet(52);
  const bool loads = !Throws([&] { ModelIO::Load(*b, slim); });
  bool weightsMatch = true, psiKept = true;
  for (size_t i = 0; loads && i + 1 < a->GetLayers().size(); ++i)
  {
    const auto& la = a->GetLayers()[i];
    const auto& lb = b->GetLayers()[i];
    const auto& lf = fresh->GetLayers()[i];
    const size_t size = la->GetInputSize(), next = la->GetOutputSize(), term = la->GetTerminalSize();
    size_t bad = 0;
    weightsMatch &= SameRows(la->GetWeights(), lb->GetWeights(), next, size, size, &bad) &&
                    SameRows(la->GetBiases(), lb->GetBiases(), 1, next, next, &bad);
    psiKept &= SameRows(lf->GetDirectFeedbackWeights(), lb->GetDirectFeedbackWeights(), size, term,
                        la->GetPsiStride(), &bad);
  }
  printf("%s\n", loads && weightsMatch && psiKept ? "PASSED: loading an inference artifact restores W and b, keeps a fresh Psi"
                                                  : "FAILED: loading an inference artifact");
  ok &= loads && weightsMatch && psiKept;

  RewriteMeta(full, cut, [](auto&) {}, false, [](const std::string& n) { return io::IsTrainingOnlyTensor(n); });
  auto c = MakeNet(53);
  const std::string text = ErrorText([&] { ModelIO::Load(*c, cut); });
  const bool strict = text.find("psi") != std::string::npos && text.find("needs") != std::string::npos;
  printf("%s\n", strict ? "PASSED: a file missing Psi without the marker is an error naming the tensor"
                        : "FAILED: a file with a missing tensor loaded");
  ok &= strict;

  const bool contradiction = Throws([&] { ModelIO::SaveCheckpoint(*a, cut, {{io::kTrainingTensorsKey, "omitted"}}); });
  printf("%s\n", contradiction ? "PASSED: a checkpoint cannot omit training tensors" : "FAILED: omitting checkpoint accepted");
  ok &= contradiction;
  for (const auto& f : {slim, full, cut})
    std::filesystem::remove(f);
  return ok;
}

/// @brief The device-load fast path (no device-to-host copy first, straight from the mapped file when it
/// can) must give the same bytes as the general path, for mapped and streamed readers, and must leave the
/// padding of a padded tensor alone.
bool DirectDeviceLoad(const std::string& path)
{
  auto backend = CreateBackend(DeviceType::DEVICE_CPU);
  std::vector<float> want(15);
  for (size_t i = 0; i < want.size(); ++i)
    want[i] = 0.25f * static_cast<float>(i) - 2.f;
  {
    io::SafeTensorsWriter w;
    w.AddTensor("t", "F32", {3, 5}, want.data(), want.size() * sizeof(float));
    w.Write(path);
  }
  bool ok = true;
  for (const bool mapped : {true, false})
  {
    io::SafeTensorsReader reader(path, true, mapped);
    std::vector<float> dst(15, -9.f);
    TensorDescriptor tight{dst.data(), {3, 5}, 0};
    detail::ReadTensorIntoStaged(backend.get(), reader, *reader.Find("t"), {3, 5}, tight);
    ok &= dst == want && reader.IsMapped() == mapped;
  }
  // A padded destination (row stride 8) takes the general path and keeps its padding.
  // Readers are scoped: Windows cannot replace or delete a file while it is mapped.
  {
    io::SafeTensorsReader reader(path);
    std::vector<float> padded(3 * 8, -9.f);
    TensorDescriptor wide{padded.data(), {3, 5}, 8};
    detail::ReadTensorIntoStaged(backend.get(), reader, *reader.Find("t"), {3, 5}, wide);
    bool paddedOk = true;
    for (size_t r = 0; r < 3; ++r)
      for (size_t c = 0; c < 8; ++c)
        paddedOk &= padded[r * 8 + c] == (c < 5 ? want[r * 5 + c] : -9.f);
    ok &= paddedOk;
  }
  // A file tensor with fewer rows than the destination (old truncated Psi) fills only those rows.
  std::vector<float> longer(15, -9.f);
  TensorDescriptor taller{longer.data(), {5, 3}, 0};
  io::SafeTensorsWriter w2;
  w2.AddTensor("t", "F32", {3, 3}, want.data(), 9 * sizeof(float));
  w2.Write(path);
  {
    io::SafeTensorsReader smallFile(path);
    detail::ReadTensorIntoStaged(backend.get(), smallFile, *smallFile.Find("t"), {3, 3}, taller);
  }
  bool shortOk = true;
  for (size_t i = 0; i < 15; ++i)
    shortOk &= longer[i] == (i < 9 ? want[i] : -9.f);
  ok &= shortOk;
  printf("%s\n", ok ? "PASSED: device loads (direct, streamed, padded, fewer rows) match the general path"
                    : "FAILED: direct device load");
  std::filesystem::remove(path);
  return ok;
}

} // namespace

/// @brief Resuming from a checkpoint must continue training bit-for-bit like the uninterrupted run:
/// same moments, same step counters. A weights-only Load() must not.
template <typename Net, typename Make, typename S = float> bool CheckpointResumes(const char* name, Make make, const std::string& path, size_t inSize,
                      size_t outSize)
{
  std::vector<S> x(kBatch * inSize), y(kBatch * outSize, S(0.1));
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = S(0.1) * (static_cast<S>(i % 7) - S(3)) / S(3);
  auto setup = [&](unsigned seed) { return make(seed); };
  auto ref = setup(11), resumed = setup(12), weightsOnly = setup(13);
  for (int i = 0; i < 5; ++i)
    ref->TrainStep(x, y, 2);
  ModelIO::SaveCheckpoint(*ref, path);
  ModelIO::LoadCheckpoint(*resumed, path);
  ModelIO::Load(*weightsOnly, path);
  for (int i = 0; i < 3; ++i)
  {
    ref->TrainStep(x, y, 2);
    resumed->TrainStep(x, y, 2);
    weightsOnly->TrainStep(x, y, 2);
  }
  const bool same = SameState(*ref, *resumed);
  const bool differs = !SameState(*ref, *weightsOnly);
  const bool notCheckpoint = Throws([&] { ModelIO::Save(*ref, path); ModelIO::LoadCheckpoint(*resumed, path); });
  printf("%s\n", same ? "PASSED" : "FAILED");
  printf("  %s: training continues bit-identically after LoadCheckpoint: %s\n", name, same ? "yes" : "NO");
  printf("  %s: a weights-only Load() does not reproduce it (so the state really matters): %s\n", name,
         differs ? "yes" : "NO");
  printf("  %s: LoadCheckpoint rejects a non-checkpoint file: %s\n", name, notCheckpoint ? "yes" : "NO");
  return same && differs && notCheckpoint;
}

static int Run();

int main()
{
  setvbuf(stdout, nullptr, _IONBF, 0);
  try
  {
    return Run();
  }
  catch (const std::exception& e)
  {
    printf("FAILED: uncaught exception: %s\n", e.what());
    return 1;
  }
}

static int Run()
{
  const auto path = (std::filesystem::temp_directory_path() / "deepity_modelio_verify.safetensors").string();
  auto src = MakeNet(1), dst = MakeNet(2);

  if (!ModelIO::Save(*src, path))
  {
    printf("FAILED: Save() could not write %s\n", path.c_str());
    return 1;
  }
  ModelIO::Load(*dst, path);
  std::filesystem::remove(path);

  bool ok = true;
  const auto& a = src->GetLayers();
  const auto& b = dst->GetLayers();
  for (size_t i = 0; i + 1 < a.size(); ++i) // the terminal layer owns no weights
  {
    const size_t size = a[i]->GetInputSize(), next = a[i]->GetOutputSize();
    const size_t term = a[i]->GetTerminalSize(), stride = a[i]->GetPsiStride();
    size_t bad = 0;

    const bool wOk = SameRows(a[i]->GetWeights(), b[i]->GetWeights(), next, size, size, &bad);
    const bool bOk = SameRows(a[i]->GetBiases(), b[i]->GetBiases(), 1, next, next, &bad);
    const bool pOk =
        SameRows(a[i]->GetDirectFeedbackWeights(), b[i]->GetDirectFeedbackWeights(), size, term, stride, &bad);
    printf("  layer %zu: W %s  b %s  Psi[%zux%zu] %s", i, wOk ? "ok" : "MISMATCH", bOk ? "ok" : "MISMATCH",
           size, term, pOk ? "ok" : "MISMATCH");
    if (!pOk)
      printf(" (first differing row %zu)", bad);
    printf("\n");
    ok &= wOk && bOk && pOk;
  }
  printf("%s\n", ok ? "PASSED: DirectKPPCNetwork save/load round trip" : "FAILED: save/load round trip");

  // FullPCNetwork: same generic check over every tensor in the state dict.
  auto fsrc = MakeFullNet(3), fdst = MakeFullNet(4);
  const bool fullOk = ModelIO::Save(*fsrc, path) && ModelIO::Load(*fdst, path) && SameState(*fsrc, *fdst);
  printf("%s\n", fullOk ? "PASSED: FullPCNetwork save/load round trip" : "FAILED: FullPCNetwork round trip");
  ok &= fullOk;

  auto dsrc = MakeDoubleNet(5), ddst = MakeDoubleNet(6);
  const bool dblOk = ModelIO::Save(*dsrc, path) && ModelIO::Load(*ddst, path) && SameState(*dsrc, *ddst);
  printf("%s\n", dblOk ? "PASSED: FullPCNetworkT<double> save/load round trip" : "FAILED: double round trip");
  ok &= dblOk;
  // A float32 file must not load into a double network.
  ModelIO::Save(*fsrc, path);
  const bool dtypeOk = Throws([&] { ModelIO::Load(*ddst, path); });
  printf("%s\n", dtypeOk ? "PASSED: dtype mismatch is rejected" : "FAILED: dtype mismatch was loaded");
  ok &= dtypeOk;

  auto csrc = MakeConvNet(7), cdst = MakeConvNet(8);
  const bool convOk = ModelIO::Save(*csrc, path) && ModelIO::Load(*cdst, path) && SameState(*csrc, *cdst);
  printf("%s\n", convOk ? "PASSED: FullConvPCNetwork save/load round trip" : "FAILED: conv round trip");
  ok &= convOk;
  ModelIO::Save(*fsrc, path); // fresh float file for the corruption check below

  // A single flipped byte in the data region must be rejected by the content hash.
  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(-5, std::ios::end);
    char c = 0;
    f.get(c);
    f.seekp(-5, std::ios::end);
    f.put(static_cast<char>(c ^ 0x01));
  }
  const bool corruptOk = Throws([&] { ModelIO::Load(*fdst, path); });
  printf("%s\n", corruptOk ? "PASSED: corrupted file is rejected" : "FAILED: corrupted file was loaded");
  ok &= corruptOk;
  ok &= CheckpointResumes<DirectKPPCNetwork>("DirectKPPCNetwork", MakeNet, path, 8, 4);
  ok &= CheckpointResumes<FullPCNetwork>("FullPCNetwork", MakeFullNet, path, 8, 4);
  ok &= CheckpointResumes<FullConvPCNetwork>("FullConvPCNetwork", MakeConvNet, path, 64, 10);
  ok &= LoadWeightsWorks(path);
  ok &= ShardingWorks(path + ".shards");
  ok &= QuantRoundTrips<Fp4E2M1Value>("FullPCNetworkT<fp4_e2m1>", path);
  ok &= QuantRoundTrips<Posit4_1Value>("FullPCNetworkT<posit4_1>", path);
  ok &= CheckpointResumes<DirectKPPCNetworkT<double, double>, decltype(&MakeDkpDouble), double>(
      "DirectKPPCNetworkT<double>", MakeDkpDouble, path, 8, 4);
  ok &= CheckpointResumes<FullPCNetworkT<double, double>, decltype(&MakeDoubleNet), double>(
      "FullPCNetworkT<double>", MakeDoubleNet, path, 8, 4);
  ok &= CheckpointResumes<GaussSeidelPCNetwork>("GaussSeidelPCNetwork", MakeGaussSeidel, path, 8, 4);
  ok &= CheckpointResumes<GaussSeidelPCNetworkT<double, double>, decltype(&MakeGaussSeidelDouble), double>(
      "GaussSeidelPCNetworkT<double>", MakeGaussSeidelDouble, path, 8, 4);
  ok &= CheckpointResumes<DiscriminativePCNetwork>("DiscriminativePCNetwork", MakeDiscriminative, path, 8, 4);
  ok &= CheckpointResumes<ConvPCNetwork>("ConvPCNetwork", MakeConvPC, path, 64, 10);
  ok &= CheckpointResumes<ConvPCNetworkT<double, double>, decltype(&MakeConvPCDouble), double>(
      "ConvPCNetworkT<double>", MakeConvPCDouble, path, 64, 10);
  ok &= CheckpointResumes<FullConvPCNetworkT<double, double>, decltype(&MakeFullConvDouble), double>(
      "FullConvPCNetworkT<double>", MakeFullConvDouble, path, 64, 10);
  ok &= CheckpointResumes<SimplePCNetwork>("SimplePCNetwork", MakeSimple, path, 8, 4);
  ok &= CheckpointResumes<SimplePCNetworkT<double, double>, decltype(&MakeSimpleDouble), double>(
      "SimplePCNetworkT<double>", MakeSimpleDouble, path, 8, 4);
  ok &= CheckpointResumes<SimpleConvPCNetwork>("SimpleConvPCNetwork", MakeSimpleConv, path, 64, 10);
  ok &= CheckpointResumes<SimpleConvPCNetworkT<double, double>, decltype(&MakeSimpleConvDouble), double>(
      "SimpleConvPCNetworkT<double>", MakeSimpleConvDouble, path, 64, 10);

  // The same checks through the staging path a GPU network takes (host copy, backend copy, counters).
  ForceDeviceStaging() = true;
  ok &= CheckpointResumes<DirectKPPCNetwork>("DirectKPPCNetwork (staged)", MakeNet, path, 8, 4);
  ok &= CheckpointResumes<FullPCNetwork>("FullPCNetwork (staged)", MakeFullNet, path, 8, 4);
  ok &= CheckpointResumes<FullConvPCNetwork>("FullConvPCNetwork (staged)", MakeConvNet, path, 64, 10);
  ok &= CheckpointResumes<GaussSeidelPCNetwork>("GaussSeidelPCNetwork (staged)", MakeGaussSeidel, path, 8, 4);
  ok &= CheckpointResumes<DiscriminativePCNetwork>("DiscriminativePCNetwork (staged)", MakeDiscriminative, path, 8, 4);
  ok &= CheckpointResumes<ConvPCNetwork>("ConvPCNetwork (staged)", MakeConvPC, path, 64, 10);
  ok &= CheckpointResumes<SimplePCNetwork>("SimplePCNetwork (staged)", MakeSimple, path, 8, 4);
  ok &= CheckpointResumes<SimpleConvPCNetwork>("SimpleConvPCNetwork (staged)", MakeSimpleConv, path, 64, 10);
  ForceDeviceStaging() = false;

  // A network whose layers expose no state dict must refuse to write an empty file.
  NoStateNet noState;
  const bool emptyRefused = Throws([&] { ModelIO::Save(noState, path + ".nostate"); });
  printf("%s\n", emptyRefused ? "PASSED: a network with no state dict refuses to save an empty file"
                              : "FAILED: empty file was written");
  ok &= emptyRefused;
  const bool gpuFree = !Throws([&] { ModelIO::SaveCheckpoint(*fsrc, path); });
  ok &= gpuFree;

  const bool versionsOk = FormatVersionRules(path);
  ok &= versionsOk;
  ok &= InferenceArtifact(path);
  ok &= DirectDeviceLoad(path + ".direct");

  const bool missingOk = Throws([&] { ModelIO::Load(*fdst, path + ".does_not_exist"); }) &&
                         Throws([&] { ModelIO::Save(*fsrc, path + "_no_such_dir/x.safetensors"); });
  printf("%s\n", missingOk ? "PASSED: missing file and unwritable path throw" : "FAILED: no exception for bad paths");
  ok &= missingOk;
  std::filesystem::remove(path);
  return ok ? 0 : 1;
}
