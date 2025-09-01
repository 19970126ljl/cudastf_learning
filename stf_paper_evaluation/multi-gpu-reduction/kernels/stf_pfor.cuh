// stf_paper_evaluation/multi-gpu-reduction/kernels/stf_pfor.cuh

#ifndef REDUCTION_KERNELS_STF_PFOR_CUH
#define REDUCTION_KERNELS_STF_PFOR_CUH

#include "common.cuh"
#include <cuda/experimental/stf.cuh>
#include <limits>

using namespace cuda::experimental::stf;

// Benchmark CUDA STF parallel_for
static inline void benchmark_stf_pfor(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
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
            cudaStream_t stream = ctx.fence();
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

#endif // REDUCTION_KERNELS_STF_PFOR_CUH
