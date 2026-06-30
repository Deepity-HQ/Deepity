#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm> // For std::max
#include <cstring>   // For std::memcpy

#include "Activations.h"

void naive_tanh(float *arr, size_t n) {
    for (size_t i=0; i < n; i++) {
        float po = expf(arr[i]);
        float ne = expf(-arr[i]);
        arr[i] = (po - ne) / (po + ne);
    }
}

void naive_relu(float *arr, size_t n) {
    for (size_t i=0; i < n; i++) {
        // Safe ternary replacement for MAX macro
        arr[i] = (arr[i] > 0.0f) ? arr[i] : 0.0f;
    }
}

void naive_sigmoid(float *arr, size_t n) {
    for (size_t i=0; i < n; i++) {
        arr[i] = 1.0f / (1.0f + expf(-arr[i]));
    }
}

void std_tanh_bench(float *arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = std::tanh(arr[i]);
    }
}

void std_relu_bench(float *arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = std::max(0.0f, arr[i]);
    }
}

void do_not_optimize(float* arr, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
    // This tells the compiler that the memory pointed to by 'arr' 
    // has been read and modified, forcing it to execute all writes.
    asm volatile("" : : "g"(arr) : "memory");
#else
    volatile float sink1 = arr[0];
    volatile float sink2 = arr[n - 1];
    (void)sink1; (void)sink2;
#endif
}

template<typename Func>
void run_benchmark(const std::string& name, const float *baseline, size_t n, Func func) {
    // 1. Correctly allocate 64-bit (8-byte) aligned scratch memory for this specific run
    // Note: n * sizeof(float) must be a multiple of the alignment (8), which your sizes are.
    float *working_data = static_cast<float*>(std::aligned_alloc(64, n * sizeof(float)));
    
    // 2. Deep copy the original baseline data so the function gets fresh inputs
    std::memcpy(working_data, baseline, n * sizeof(float));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    func(working_data, n);
    do_not_optimize(working_data, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "  " << name << ": " << duration.count() << " ms\n";
    
    // 3. Clean up the scratch memory
    std::free(working_data);
}

int main(void)
{
    std::vector<size_t> sizes = {10'048, 1'000'064, 50'000'064};
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-3.0f, 3.0f);

    for (size_t n : sizes) {
        std::cout << "========================================\n";
        std::cout << "Testing Size N = " << n << " (" << (n * sizeof(float)) / (1024.0 * 1024.0) << " MB)\n";
        std::cout << "========================================\n";
        
        // Allocate baseline on an 8-byte boundary
        float *baseline = static_cast<float*>(std::aligned_alloc(8, n * sizeof(float)));
        for (size_t i = 0; i < n; ++i) {
            baseline[i] = dis(gen);
        }

        std::cout << "--- Tanh Performance ---\n";
        run_benchmark("Naive Tanh (2x expf)", baseline, n, naive_tanh);
        run_benchmark("Standard std::tanh  ", baseline, n, std_tanh_bench);
        run_benchmark("Custom Tanh         ", baseline, n, Deep::tanh);

        std::cout << "\n--- ReLU Performance ---\n";
        run_benchmark("Naive ReLU (MAX)    ", baseline, n, naive_relu);
        run_benchmark("Standard std::max   ", baseline, n, std_relu_bench);
        run_benchmark("Custom ReLU         ", baseline, n, Deep::relu);
        std::cout << "\n";

        std::cout << "\n--- Sigmoid Performance ---\n";
        run_benchmark("Naive Sigmoid    ", baseline, n, naive_sigmoid);
        run_benchmark("Custom Sigmoid   ", baseline, n, Deep::sigmoid);

        std::free(baseline);
    }

    return 0;
}