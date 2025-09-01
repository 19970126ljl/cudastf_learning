// stf_paper_evaluation/multi-gpu-reduction/kernels/common.cuh

#ifndef REDUCTION_KERNELS_COMMON_CUH
#define REDUCTION_KERNELS_COMMON_CUH

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <thrust/device_vector.h>

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

static inline void report_results(
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

#endif // REDUCTION_KERNELS_COMMON_CUH
