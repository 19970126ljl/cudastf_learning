// stf_paper_evaluation/multi-gpu-reduction/kernels/thrust.cuh

#ifndef REDUCTION_KERNELS_THRUST_CUH
#define REDUCTION_KERNELS_THRUST_CUH

#include "common.cuh"
#include <thrust/reduce.h>
#include <thrust/functional.h>

// Benchmark Thrust
static inline void benchmark_thrust(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
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

#endif // REDUCTION_KERNELS_THRUST_CUH
