// stf_paper_evaluation/multi-gpu-reduction/kernels/cub.cuh

#ifndef REDUCTION_KERNELS_CUB_CUH
#define REDUCTION_KERNELS_CUB_CUH

#include "common.cuh"
#include <cub/cub.cuh>

// Benchmark CUB
static inline void benchmark_cub(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
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

#endif // REDUCTION_KERNELS_CUB_CUH
