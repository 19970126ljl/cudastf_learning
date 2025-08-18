//===----------------------------------------------------------------------===//
//
// Benchmark program for comparing CUDA STF, CUB, and Thrust reduction performance
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <unistd.h>

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
    
    void start() {
        cudaEventRecord(start_, 0);
    }
    
    void stop() {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
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
    auto spec = par(con<128>(hw_scope::thread));
    
    ctx.launch(spec, where, lX.read(), lsum.rw())->*[] _CCCL_DEVICE(auto th, auto x, auto sum) {
        // Each thread computes the sum of elements assigned to it
        double local_sum = 0.0;
        for (auto i : th.apply_partition(shape(x))) {
            local_sum += x(i);
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

// Benchmark function
void benchmark_reduction(unsigned long long N, std::ofstream& csv_file, bool verify_with_cpu = false, int num_iterations = 10) {
    std::cout << "Benchmarking reduction with " << N << " elements (\" " 
              << N * sizeof(double) / 1024.0 / 1024.0 << " MB)\"" << std::endl;
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
    
    
    
    // Benchmark CUDA STF
    {
        double total_time = 0.0;
        double result = 0.0;
        const size_t data_size_bytes = N * sizeof(double);

        // Device pointers for STF (following CUB's pattern)
        double* d_input_stf = d_X.data().get();
        double* d_output_stf = nullptr;

        CUDA_CHECK(cudaSetDevice(0)); // Assuming device 0 for STF
        CUDA_CHECK(cudaMalloc(&d_output_stf, sizeof(double)));
        
        // Data is already on device in d_X
        
        for (int i = 0; i < num_iterations; i++) {
            context ctx;
            double sum = 0.0;
            
            // Create logical data from device memory
            auto lX = ctx.logical_data(d_input_stf, {N}, data_place::device());
            auto lsum = ctx.logical_data(&sum, {1}); // Keep sum on host for result retrieval

            GPUTimer timer;
            timer.start();
            stf_reduce_kernel_launch(ctx, lX, lsum);
            cudaStream_t stream = ctx.task_fence();
            cudaStreamSynchronize(stream);
            timer.stop();
            ctx.finalize(); // STF finalize synchronizes, so no extra cudaDeviceSynchronize needed here for timing. 
            result = sum; // Get result after finalize ensures it's available. 
            total_time += timer.elapsed_milliseconds();
            
            // Verify result if CPU verification is enabled
            if (verify_with_cpu) {
                double error = fabs(result - ref_sum);
                if (error > 1e-6) {
                    std::cerr << "ERROR: CUDA STF reduction failed for size " << N
                              << ". Error: "
                              << std::scientific << error << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        // d_input_stf is managed by d_X
        CUDA_CHECK(cudaFree(d_output_stf));
        
        double avg_time = total_time / num_iterations;
        double bandwidth = (N * sizeof(double)) / (avg_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        double error = verify_with_cpu ? fabs(result - ref_sum) : 0.0;
        
        std::cout << std::setw(15) << "CUDA STF"
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth
                  << std::setw(20) << std::scientific << error
                  << std::endl;
        csv_file << N << ",CUDA STF," << avg_time << "," << bandwidth << "," << error << "\n";
    }

    // Benchmark CUDA STF parallel_for
    {
        double total_time = 0.0;
        double result = 0.0;
        const size_t data_size_bytes = N * sizeof(double);

        // Device pointers for STF
        double* d_input_stf_pfor = d_X.data().get();

        CUDA_CHECK(cudaSetDevice(0)); // Assuming device 0 for STF
        
        // Data is already on device in d_X
        
        for (int i = 0; i < num_iterations; i++) {
            context ctx;
            
            auto lX = ctx.logical_data(d_input_stf_pfor, {N}, data_place::device());
            auto lsum = ctx.logical_data(shape_of<scalar_view<double>>());

            GPUTimer timer;
            timer.start();
            
            ctx.parallel_for(lX.shape(), lX.read(), lsum.reduce(reducer::sum<double>{}))
                ->*[] __device__(size_t i, auto x, auto& sum) {
                    sum += x(i);
                };

            // Add CUDA error check after parallel_for
            CUDA_CHECK(cudaGetLastError());
            
            // Try task_fence instead of wait for large datasets
            if (N > 1000000000) { // For datasets > 1GB
                cudaStream_t stream = ctx.task_fence();
                cudaStreamSynchronize(stream);
                result = ctx.wait(lsum); // Now wait should be fast
            } else {
                result = ctx.wait(lsum);
            }
            timer.stop();
            ctx.finalize();
            total_time += timer.elapsed_milliseconds();
            
            // Verify result if CPU verification is enabled
            if (verify_with_cpu) {
                double error = fabs(result - ref_sum);
                if (error > 1e-6) {
                    std::cerr << "ERROR: CUDA STF pfor reduction failed for size " << N
                              << ". Error: "
                              << std::scientific << error << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        // d_input_stf_pfor is managed by d_X
        
        double avg_time = total_time / num_iterations;
        double bandwidth = (N * sizeof(double)) / (avg_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        double error = verify_with_cpu ? fabs(result - ref_sum) : 0.0;
        
        std::cout << std::setw(15) << "CUDA STF pfor"
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth
                  << std::setw(20) << std::scientific << error
                  << std::endl;
        csv_file << N << ",CUDA STF pfor," << avg_time << "," << bandwidth << "," << error << "\n";
    }
    
    // Benchmark CUB
    {
        double total_time = 0.0;
        double result = 0.0;
        const size_t data_size_bytes = N * sizeof(double);

        // Device pointers for CUB
        double* d_input_cub = d_X.data().get();
        double* d_output_cub = nullptr;
        void* d_temp_storage_cub = nullptr;
        size_t temp_storage_bytes_cub = 0;

        CUDA_CHECK(cudaSetDevice(0)); // Assuming device 0 for CUB
        CUDA_CHECK(cudaMalloc(&d_output_cub, sizeof(double)));
        
        // Data is already on device in d_X

        // Get temporary storage size
        cub::DeviceReduce::Sum(d_temp_storage_cub, temp_storage_bytes_cub, d_input_cub, d_output_cub, N);
        CUDA_CHECK(cudaMalloc(&d_temp_storage_cub, temp_storage_bytes_cub));
        
        for (int i = 0; i < num_iterations; i++) {
            GPUTimer timer;
            timer.start();
            cub::DeviceReduce::Sum(d_temp_storage_cub, temp_storage_bytes_cub, d_input_cub, d_output_cub, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop();
            total_time += timer.elapsed_milliseconds();
        }

        // Copy result back after all timed iterations
        CUDA_CHECK(cudaMemcpy(&result, d_output_cub, sizeof(double), cudaMemcpyDeviceToHost));
        
        // Verify result if CPU verification is enabled
        if (verify_with_cpu) {
            double error = fabs(result - ref_sum);
            if (error > 1e-6) {
                std::cerr << "ERROR: CUB reduction failed for size " << N
                          << ". Error: "
                          << std::scientific << error << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // d_input_cub is managed by d_X
        CUDA_CHECK(cudaFree(d_output_cub));
        CUDA_CHECK(cudaFree(d_temp_storage_cub));
        
        double avg_time = total_time / num_iterations;
        double bandwidth = (data_size_bytes) / (avg_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        double error = verify_with_cpu ? fabs(result - ref_sum) : 0.0;
        
        std::cout << std::setw(15) << "CUB"
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth
                  << std::setw(20) << std::scientific << error
                  << std::endl;
        csv_file << N << ",CUB," << avg_time << "," << bandwidth << "," << error << "\n";
    }
    
    // Benchmark Thrust
    {
        double total_time = 0.0;
        double result = 0.0;
        const size_t data_size_bytes = N * sizeof(double);

        // Data is already in a thrust device_vector d_X

        CUDA_CHECK(cudaSetDevice(0)); // Assuming device 0 for Thrust
        
        for (int i = 0; i < num_iterations; i++) {
            GPUTimer timer;
            timer.start();
            result = thrust::reduce(d_X.begin(), d_X.end(), 0.0, thrust::plus<double>());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop();
            total_time += timer.elapsed_milliseconds();
            
            // Verify result if CPU verification is enabled
            if (verify_with_cpu) {
                double error = fabs(result - ref_sum);
                if (error > 1e-6) {
                    std::cerr << "ERROR: Thrust reduction failed for size " << N
                              << ". Error: "
                              << std::scientific << error << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        // No need to free anything, d_X is managed by its scope
        
        double avg_time = total_time / num_iterations;
        double bandwidth = (data_size_bytes) / (avg_time / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        double error = verify_with_cpu ? fabs(result - ref_sum) : 0.0;
        
        std::cout << std::setw(15) << "Thrust"
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth
                  << std::setw(20) << std::scientific << error
                  << std::endl;
        csv_file << N << ",Thrust," << avg_time << "," << bandwidth << "," << error << "\n";
    }
    
    std::cout << std::string(80, '-') << std::endl;
}

int main(int argc, char** argv) {
    bool verify_with_cpu = false;
    int opt;
    
    // Parse command line arguments
    while ((opt = getopt(argc, argv, "c")) != -1) {
        switch (opt) {
            case 'c':
                verify_with_cpu = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-c]" << std::endl;
                std::cerr << "  -c: Enable CPU reduction verification" << std::endl;
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
    std::vector<unsigned long long> problem_sizes = {
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
    
    for (unsigned long long N : problem_sizes) {
        benchmark_reduction(N, csv_file, verify_with_cpu, 10);
        std::cout << std::endl;
    }

    csv_file.close();
    
    return 0;
}
