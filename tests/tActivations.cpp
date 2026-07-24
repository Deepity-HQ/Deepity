#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include <string>
#include "Activations.h"

void ref_tanh(float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = std::tanh(x[i]);
}
void ref_dTanh(float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float t = std::tanh(x[i]);
        x[i] = 1.0f - t * t;
    }
}
void ref_sigmoid(float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        x[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}
void ref_dSigmoid(float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float s = 1.0f / (1.0f + std::exp(-x[i]));
        x[i] = s * (1.0f - s);
    }
}
void ref_relu(float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::max(0.0f, x[i]);
    }
}
void ref_dRelu(float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        x[i] = x[i] > 0.0f ? 1.0f : 0.0f;
    }
}

// Calculates the maximum absolute error between two arrays
float calculate_max_error(const std::vector<float>& ref, const std::vector<float>& test) {
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float err = std::abs(ref[i] - test[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Benchmarks a function and returns the average execution time in microseconds
template <typename Func>
double benchmark_function(Func f, const std::vector<float>& data, int iterations) {
    std::vector<float> buffer = data;
    f(buffer.data(), buffer.size()); // Warm-up run

    double total_time_us = 0.0;
    
    for (int i = 0; i < iterations; ++i) {
        buffer = data; // Reset data to avoid state-dependent branch prediction (e.g., in ReLU)
        
        auto start = std::chrono::high_resolution_clock::now();
        f(buffer.data(), buffer.size());
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::micro> elapsed = end - start;
        total_time_us += elapsed.count();
    }
    
    return total_time_us / iterations;
}

template <typename RefFn, typename MyFn>
void run_test(const std::string& name,
              RefFn ref_func,
              MyFn my_func,
              const std::vector<float>& input_data,
              int iterations)
{
    size_t n = input_data.size();

    // 1. Test Accuracy
    std::vector<float> ref_out = input_data;
    std::vector<float> my_out = input_data;

    ref_func(ref_out.data(), n);
    my_func(my_out.data(), n);

    float max_error = calculate_max_error(ref_out, my_out);

    // 2. Test Speed
    double ref_time = benchmark_function(ref_func, input_data, iterations);
    double my_time = benchmark_function(my_func, input_data, iterations);

    // Calculate speedup
    double speedup = ref_time / my_time;

    // 3. Print Results
    std::cout << std::left << std::setw(15) << name
              << "| " << std::setw(12) << std::scientific << std::setprecision(4) << max_error
              << "| " << std::fixed << std::setw(10) << std::setprecision(2) << ref_time
              << "| " << std::setw(10) << my_time
              << "| " << std::setw(8) << speedup << "x\n";
}

int main(void) {
    const size_t ARRAY_SIZE = 1'000'000;
    const int ITERATIONS = 100;
    
    std::cout << "Generating " << ARRAY_SIZE << " random floats for testing...\n";
    std::vector<float> data(ARRAY_SIZE);
    std::mt19937 gen(42); // Fixed seed for reproducible benchmarks
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        data[i] = dist(gen);
    }
    
    std::cout << "\n----------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(15) << "Function" 
              << "| " << std::setw(12) << "Max Error" 
              << "| " << std::setw(10) << "Ref (us)" 
              << "| " << std::setw(10) << "Yours (us)" 
              << "| " << "Speedup\n";
    std::cout << "----------------------------------------------------------------------\n";

    run_test("Tanh", ref_tanh, Deep::tanh, data, ITERATIONS);
    run_test("dTanh", ref_dTanh, [](float* x, size_t n) { Deep::dTanh(x, n, false); }, data, ITERATIONS);
    run_test("Sigmoid", ref_sigmoid, Deep::sigmoid, data, ITERATIONS);
    run_test("dSigmoid", ref_dSigmoid, [](float* x, size_t n) { Deep::dSigmoid(x, n, false); }, data, ITERATIONS);
    run_test("ReLU", ref_relu, Deep::relu, data, ITERATIONS);
    run_test("dReLU", ref_dRelu, [](float* x, size_t n) { Deep::dRelu(x, n, false); }, data, ITERATIONS);
    
    std::cout << "----------------------------------------------------------------------\n";

    return 0;
}