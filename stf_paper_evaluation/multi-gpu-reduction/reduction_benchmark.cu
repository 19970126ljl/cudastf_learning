//===----------------------------------------------------------------------===//
//
// Benchmark program for comparing CUDA STF, CUB, and Thrust reduction performance
//
//===----------------------------------------------------------------------===//

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <vector_types.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <limits>
#include <string>
#include <algorithm>
#include <cctype>

#include "kernels/common.cuh"
#include "kernels/stf_launch_baseline.cuh"
#include "kernels/stf_pfor.cuh"
#include "kernels/cub.cuh"
#include "kernels/thrust.cuh"
#include "kernels/stf_launch_wide_load.cuh"

// Data generation functor for Thrust
struct sin_functor
{
    __host__ __device__
    double operator()(unsigned long long i) const
    {
        return sin((double)i);
    }
};

// Reference CPU implementation for verification
double cpu_reduce(const std::vector<double>& data) {
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i];
    }
    return sum;
}

// Benchmark function
void benchmark_reduction(unsigned long long N, std::ofstream& csv_file, const std::string& model, bool verify_with_cpu = false, int num_iterations = 10) {
    std::cout << "Benchmarking reduction with " << N << " elements (" 
              << N * sizeof(double) / 1024.0 / 1024.0 << " MB)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    // Generate data on the device using Thrust
    thrust::device_vector<double> d_X(N);
    thrust::transform(thrust::counting_iterator<unsigned long long>(0),
                      thrust::counting_iterator<unsigned long long>(N),
                      d_X.begin(),
                      sin_functor());
    
    // Compute reference sum on CPU if verification is enabled
    double ref_sum = 0.0;
    if (verify_with_cpu) {
        std::vector<double> X(N);
        thrust::copy(d_X.begin(), d_X.end(), X.begin());
        ref_sum = cpu_reduce(X);
        std::cout << "Reference sum (CPU): " << std::fixed << std::setprecision(15) << ref_sum << std::endl;
    }
    std::cout << std::string(80, '-') << std::endl;
    
    std::cout << "Testing on a single device:" << std::endl;
    std::cout << std::setw(15) << "Method" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "GB/s" 
              << std::setw(20) << "Error" 
              << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    if (model == "all" || model == "stf") {
        benchmark_stf_launch_baseline(N, d_X, ref_sum, verify_with_cpu, num_iterations, csv_file);
    }
        if (model == "all" || model == "stf_pfor") {
        benchmark_stf_pfor(N, d_X, ref_sum, verify_with_cpu, num_iterations, csv_file);
    }
    if (model == "all" || model == "cub") {
        benchmark_cub(N, d_X, ref_sum, verify_with_cpu, num_iterations, csv_file);
    }
    if (model == "all" || model == "thrust") {
        benchmark_thrust(N, d_X, ref_sum, verify_with_cpu, num_iterations, csv_file);
    }
    if (model == "all" || model == "stf_wide_load") {
        benchmark_stf_launch_wide_load(N, d_X, ref_sum, verify_with_cpu, num_iterations, csv_file);
    }
    
    std::cout << std::string(80, '-') << std::endl;
}

unsigned long long parse_size(std::string s) {
    s.erase(std::remove_if(s.begin(), s.end(), isspace), s.end());
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    unsigned long long multiplier = 1;
    size_t pos = std::string::npos;

    if ((pos = s.find("gb")) != std::string::npos) {
        multiplier = 1ULL * 1024 * 1024 * 1024;
        s = s.substr(0, pos);
    } else if ((pos = s.find("g")) != std::string::npos) {
        multiplier = 1ULL * 1024 * 1024 * 1024;
        s = s.substr(0, pos);
    } else if ((pos = s.find("mb")) != std::string::npos) {
        multiplier = 1024 * 1024;
        s = s.substr(0, pos);
    } else if ((pos = s.find("m")) != std::string::npos) {
        multiplier = 1024 * 1024;
        s = s.substr(0, pos);
    } else if ((pos = s.find("kb")) != std::string::npos) {
        multiplier = 1024;
        s = s.substr(0, pos);
    } else if ((pos = s.find("k")) != std::string::npos) {
        multiplier = 1024;
        s = s.substr(0, pos);
    }

    unsigned long long value = std::stoull(s);
    // The user specifies size in bytes, so we convert to number of elements
    return (value * multiplier) / sizeof(double);
}

int main(int argc, char** argv) {
    bool verify_with_cpu = false;
    std::string model_to_run = "all";
    unsigned long long problem_size_arg = 0;
    int num_iterations_arg = 10;
    int opt;
    
    // Parse command line arguments
    while ((opt = getopt(argc, argv, "cm:n:i:")) != -1) {
        switch (opt) {
            case 'c':
                verify_with_cpu = true;
                break;
            case 'm':
                model_to_run = optarg;
                break;
            case 'n':
                try {
                    problem_size_arg = parse_size(optarg);
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Invalid argument for -n: " << optarg << std::endl;
                    return 1;
                }
                break;
            case 'i':
                num_iterations_arg = std::stoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-c] [-m <model>] [-n <size>] [-i <iterations>]" << std::endl;
                std::cerr << "  -c: Enable CPU reduction verification" << std::endl;
                std::cerr << "  -m: Benchmark a specific model (stf, stf_pfor, stf_wide_load, cub, thrust, all). Default: all" << std::endl;
                std::cerr << "  -n: Problem size (e.g., 1024, 512MB, 2GB). Default: a range of sizes" << std::endl;
                std::cerr << "  -i: Number of iterations. Default: 10" << std::endl;
                return 1;
        }
    }
    
    if (verify_with_cpu) {
        std::cout << "CPU verification enabled - will verify GPU results against CPU reference" << std::endl;
    } else {
        std::cout << "CPU verification disabled - running benchmarks without validation" << std::endl;
    }
    std::cout << std::endl;
    // Get device properties
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s). Using device 0 for benchmarks." << std::endl;

    // Open CSV file for writing results
    std::ofstream csv_file("reduction_benchmark_results.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open reduction_benchmark_results.csv for writing." << std::endl;
        return 1;
    }

    // Write CSV header
    csv_file << "N,Method,Time_ms,GB_s,Error\n";
    
    // Test with different problem sizes
    std::vector<unsigned long long> problem_sizes;
    if (problem_size_arg > 0) {
        problem_sizes.push_back(problem_size_arg);
    } else {
        problem_sizes = {
            1024 * 1024,           // 8 MB
            16 * 1024 * 1024,      // 128 MB
            128 * 1024 * 1024,     // 1 GB
            256 * 1024 * 1024,     // 2 GB
            512 * 1024 * 1024,     // 4 GB
            1ULL * 1024 * 1024 * 1024,    // 8 GB
            2ULL * 1024 * 1024 * 1024,    // 16 GB
            4ULL * 1024 * 1024 * 1024,    // 32 GB
            8ULL * 1024 * 1024 * 1024     // 64 GB
        };
    }
    
    for (unsigned long long N : problem_sizes) {
        benchmark_reduction(N, csv_file, model_to_run, verify_with_cpu, num_iterations_arg);
        std::cout << std::endl;
    }

    csv_file.close();
    
    return 0;
}