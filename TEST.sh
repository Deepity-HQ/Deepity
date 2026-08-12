#!/bin/bash

echo "=== Baseline (unset) ==="
./build/Release/bin/DeepityTests

echo "=== OMP=1 BLAS=1 ==="
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./build/Release/bin/DeepityTests

echo "=== OMP=1 BLAS=4 ==="
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4 ./build/Release/bin/DeepityTests

echo "=== OMP=1 BLAS=8 ==="
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=8 ./build/Release/bin/DeepityTests

echo "=== OMP=1 BLAS=20 ==="
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=20 ./build/Release/bin/DeepityTests

echo "=== OMP=4 BLAS=4 ==="
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ./build/Release/bin/DeepityTests

echo "=== OMP=8 BLAS=8 ==="
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ./build/Release/bin/DeepityTests

echo "=== OMP=20 BLAS=20 ==="
OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 ./build/Release/bin/DeepityTests
