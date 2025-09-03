// stf_paper_evaluation/multi-gpu-reduction/mgpu-kernels/stf_pfor.cuh

#ifndef REDUCTION_MGPU_KERNELS_STF_PFOR_CUH
#define REDUCTION_MGPU_KERNELS_STF_PFOR_CUH

#include "../kernels/common.cuh"
#include <cuda/experimental/stf.cuh>
#include <limits>

using namespace cuda::experimental::stf;

static inline void benchmark_stf_pfor_mgpu(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, int num_gpus, std::ofstream& csv_file) {
    if (N > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        std::cout << std::setw(15) << ("STF pfor " + std::to_string(num_gpus) + "GPU")
                  << std::setw(15) << "N/A (skipped)"
                  << std::setw(15) << "N/A (skipped)"
                  << std::setw(20) << "N/A (skipped)"
                  << std::endl;
        return;
    }

    double total_time = 0.0;
    double result = 0.0;
    
    // Copy data to host first, then let STF manage distribution
    std::vector<double> h_X(N);
    thrust::copy(d_X.begin(), d_X.end(), h_X.begin());
    
    exec_place where;
    if (num_gpus == 1) {
        where = exec_place::current_device();
    } else {
        where = exec_place::all_devices();
    }

    GPUTimer timer;
    for (int i = 0; i < num_iterations; i++) {
        context ctx;
        auto lX = ctx.logical_data(&h_X[0], {N});
        
    double* sum;
    cudaMallocManaged(&sum, sizeof(double));
    *sum = 0.0;
    // Create scalar logical data for accumulation with managed memory
    auto lsum = ctx.logical_data(sum, {1}, data_place::managed());

        timer.start(("STF pfor " + std::to_string(num_gpus) + "GPU").c_str());
        
        ctx.launch(where, lX.read(), lsum.rw(data_place::managed()))->*[] __device__(auto t, auto dX, auto dsum) {
            double thread_sum = 0.0;
            // Each thread processes elements assigned to it by the partitioner
            for (auto ind : t.apply_partition(shape(dX))) {
                thread_sum += dX(ind);
            }
            // Use atomic operation to accumulate to global result
            if (thread_sum != 0.0) {
                atomicAdd(&dsum(0), thread_sum);
            }
        };
        
    cudaStreamSynchronize(ctx.fence());
    result = *sum;
    timer.stop();
    ctx.finalize();
    total_time += timer.elapsed_milliseconds();
    cudaFree(sum);
    }

    double avg_time = total_time / num_iterations;
    std::string method_name = "STF pfor " + std::to_string(num_gpus) + "GPU";
    report_results(method_name, N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

#endif // REDUCTION_MGPU_KERNELS_STF_PFOR_CUH
