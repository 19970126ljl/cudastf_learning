//===----------------------------------------------------------------------===//
//
// Benchmark program for comparing CUDA STF, CUB, and Thrust reduction performance
//
//===----------------------------------------------------------------------===//
#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/stf.cuh>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <nvtx3/nvtx3.hpp>
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

using namespace cuda::experimental::stf;

// Simple CUDA error check helper
static inline void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                  << " at line " << __LINE__ << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Timer class for measuring GPU execution time
class GPUTimer {
public:
    GPUTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start(const char* name) {
        nvtxRangePushA(name);
        cudaEventRecord(start_, 0);
    }
    
    void stop() {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        nvtxRangePop();
    }
    
    float elapsed_milliseconds() {
        float elapsed;
        cudaEventElapsedTime(&elapsed, start_, stop_);
        return elapsed;
    }
    
private:
    cudaEvent_t start_, stop_;
};

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

// CUDA STF reduction implementation
// Assumes lX and lsum are created and managed by the caller.
// This function only encapsulates the kernel launch.
template<typename T1, typename T2>
void stf_reduce_kernel_launch(context& ctx, T1 lX, T2 lsum) {
    auto where = exec_place::device(0);
    auto spec = par<4320>(con<256>(hw_scope::thread));
    
    ctx.launch(spec, where, lX.read(), lsum.rw())->*[] _CCCL_DEVICE(auto th, auto x, auto sum) {
        // Each thread computes the sum of elements assigned to it
        double local_sum = 0.0;
        
        // Get the original shape
        auto original_shape = shape(x);
        
        // Get the sizes from the original shape
        auto sizes = original_shape.get_sizes();
        
        // Calculate the vectorized size
        constexpr size_t vectorized_size = 4;
        // Get the last dimension's size directly
        size_t last_dim = original_shape.extent(original_shape.rank() - 1);
        size_t vectorized_last_dim = last_dim / vectorized_size;
            
        // Create a new box shape with the modified last dimension
        box<1> vectorized_shape(vectorized_last_dim);
        // Second loop: Handle remaining elements (if last_dim is not divisible by 4)
        size_t remaining_start = vectorized_last_dim * vectorized_size;
        if (remaining_start < last_dim) {
            box<1> remaining_shape(last_dim - remaining_start);
            for (auto i : th.apply_partition(remaining_shape)) {
                local_sum += x(remaining_start + i);
            }
        }
        
        
        // First loop: Process elements in vectorized chunks (no divergence)
        for (auto i : th.apply_partition(vectorized_shape)) {
            // Calculate the original index
            size_t original_idx = i * vectorized_size;
            
            // Use reinterpret_cast for vectorized loading
            const double4* vec_ptr = reinterpret_cast<const double4*>(&x(original_idx));
            double4 vec = *vec_ptr;
            
            // Sum all four components
            // local_sum += vec.x;
            // local_sum += vec.y;
            // local_sum += vec.z;
            // local_sum += vec.w;
            local_sum += vec.x + vec.y + vec.z + vec.w;
        }
        
        auto ti = th.inner();
        
        __shared__ double block_sum[th.static_width(1)];
        block_sum[ti.rank()] = local_sum;
        
        for (size_t s = ti.size() / 2; s > 0; s /= 2) {
            ti.sync();
            if (ti.rank() < s) {
                block_sum[ti.rank()] += block_sum[ti.rank() + s];
            }
        }
        
        if (ti.rank() == 0) {
            atomicAdd(&sum(0), block_sum[0]);
        }
    };
}

void report_results(
    const std::string& method_name,
    unsigned long long N,
    double avg_time,
    double result,
    double ref_sum,
    bool verify_with_cpu,
    std::ofstream& csv_file
) {
    if (verify_with_cpu) {
        double error = fabs(result - ref_sum);
        if (error > 1e-6) {
            std::cerr << "ERROR: " << method_name << " reduction failed for size " << N
                      << ". Error: " << std::scientific << error << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    const size_t data_size_bytes = N * sizeof(double);
    double bandwidth = (data_size_bytes) / (avg_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    double error = verify_with_cpu ? fabs(result - ref_sum) : 0.0;

    std::cout << std::setw(15) << method_name
              << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
              << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth
              << std::setw(20) << std::scientific << error
              << std::endl;
    csv_file << N << "," << method_name << "," << avg_time << "," << bandwidth << "," << error << "\n";
}

// Benchmark CUDA STF
void benchmark_stf(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
    double total_time = 0.0;
    double result = 0.0;
    double* d_input_stf = d_X.data().get();
    CUDA_CHECK(cudaSetDevice(0));

    // Warm-up
    {
        context ctx;
        double sum = 0.0;
        auto lX = ctx.logical_data(d_input_stf, {N}, data_place::device());
        auto lsum = ctx.logical_data(&sum, {1});
        stf_reduce_kernel_launch(ctx, lX, lsum);
        cudaStream_t stream = ctx.task_fence();
        cudaStreamSynchronize(stream);
        ctx.finalize();
    }

    GPUTimer timer;
    for (int i = 0; i < num_iterations; i++) {
        context ctx;
        double sum = 0.0;
        auto lX = ctx.logical_data(d_input_stf, {N}, data_place::device());
        auto lsum = ctx.logical_data(&sum, {1});

        timer.start("CUDA STF");
        stf_reduce_kernel_launch(ctx, lX, lsum);
        cudaStream_t stream = ctx.task_fence();
        cudaStreamSynchronize(stream);
        timer.stop();
        ctx.finalize();
        result = sum;
        total_time += timer.elapsed_milliseconds();
    }

    double avg_time = total_time / num_iterations;
    report_results("CUDA STF", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}



// Benchmark CUDA STF parallel_for
void benchmark_stf_pfor(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
    if (N > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        std::cout << std::setw(15) << "CUDA STF pfor"
                  << std::setw(15) << "N/A (skipped)"
                  << std::setw(15) << "N/A (skipped)"
                  << std::setw(20) << "N/A (skipped)"
                  << std::endl;
        return;
    }

    double total_time = 0.0;
    double result = 0.0;
    double* d_input_stf_pfor = d_X.data().get();
    CUDA_CHECK(cudaSetDevice(0));

    // Warm-up
    {
        context ctx;
        auto lX = ctx.logical_data(d_input_stf_pfor, {static_cast<size_t>(N)}, data_place::device());
        auto lsum = ctx.logical_data(shape_of<scalar_view<double>>());
        ctx.parallel_for(lX.shape(), lX.read(), lsum.reduce(reducer::sum<double>{}))
            ->*[] __device__(size_t i, auto x, auto& sum) {
                sum += x(i);
            };
        ctx.wait(lsum);
        ctx.finalize();
    }

    GPUTimer timer;
    for (int i = 0; i < num_iterations; i++) {
        context ctx;
        auto lX = ctx.logical_data(d_input_stf_pfor, {static_cast<size_t>(N)}, data_place::device());
        auto lsum = ctx.logical_data(shape_of<scalar_view<double>>());

        timer.start("CUDA STF pfor");
        ctx.parallel_for(lX.shape(), lX.read(), lsum.reduce(reducer::sum<double>{}))
            ->*[] __device__(size_t i, auto x, auto& sum) {
                sum += x(i);
            };
        CUDA_CHECK(cudaGetLastError());
        if (N > 1000000000) {
            cudaStream_t stream = ctx.task_fence();
            cudaStreamSynchronize(stream);
            result = ctx.wait(lsum);
        } else {
            result = ctx.wait(lsum);
        }
        timer.stop();
        ctx.finalize();
        total_time += timer.elapsed_milliseconds();
    }

    double avg_time = total_time / num_iterations;
    report_results("CUDA STF pfor", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

// Benchmark CUB
void benchmark_cub(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
    double total_time = 0.0;
    double result = 0.0;
    double* d_input_cub = d_X.data().get();
    double* d_output_cub = nullptr;
    void* d_temp_storage_cub = nullptr;
    size_t temp_storage_bytes_cub = 0;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d_output_cub, sizeof(double)));
    
    // Determine temporary storage requirements
    cub::DeviceReduce::Sum(d_temp_storage_cub, temp_storage_bytes_cub, d_input_cub, d_output_cub, N);
    CUDA_CHECK(cudaMalloc(&d_temp_storage_cub, temp_storage_bytes_cub));
    
    // Warm-up
    cub::DeviceReduce::Sum(d_temp_storage_cub, temp_storage_bytes_cub, d_input_cub, d_output_cub, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    GPUTimer timer;
    for (int i = 0; i < num_iterations; i++) {
        timer.start("CUB");
        cub::DeviceReduce::Sum(d_temp_storage_cub, temp_storage_bytes_cub, d_input_cub, d_output_cub, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        total_time += timer.elapsed_milliseconds();
    }

    CUDA_CHECK(cudaMemcpy(&result, d_output_cub, sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_output_cub));
    CUDA_CHECK(cudaFree(d_temp_storage_cub));
    
    double avg_time = total_time / num_iterations;
    report_results("CUB", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

// Benchmark Thrust
void benchmark_thrust(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
    double total_time = 0.0;
    double result = 0.0;
    CUDA_CHECK(cudaSetDevice(0));
    
    // Warm-up
    result = thrust::reduce(d_X.begin(), d_X.end(), 0.0, thrust::plus<double>());
    CUDA_CHECK(cudaDeviceSynchronize());

    GPUTimer timer;
    for (int i = 0; i < num_iterations; i++) {
        timer.start("Thrust");
        result = thrust::reduce(d_X.begin(), d_X.end(), 0.0, thrust::plus<double>());
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        total_time += timer.elapsed_milliseconds();
    }
    
    double avg_time = total_time / num_iterations;
    report_results("Thrust", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
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
        benchmark_stf(N, d_X, ref_sum, verify_with_cpu, num_iterations, csv_file);
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
                std::cerr << "  -m: Benchmark a specific model (stf, stf_pfor, cub, thrust, all). Default: all" << std::endl;
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
