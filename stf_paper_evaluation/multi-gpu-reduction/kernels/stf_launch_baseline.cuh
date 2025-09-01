// stf_paper_evaluation/multi-gpu-reduction/kernels/stf.cuh

#ifndef REDUCTION_KERNELS_STF_CUH
#define REDUCTION_KERNELS_STF_CUH

#include "common.cuh"
#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// CUDA STF reduction implementation
// Assumes lX and lsum are created and managed by the caller.
// This function only encapsulates the kernel launch.
template<typename T1, typename T2>
void stf_baseline_launch(context& ctx, T1 lX, T2 lsum) {
    auto where = exec_place::device(0);
    auto spec = par(con<32>(hw_scope::thread));
    
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

// Benchmark CUDA STF
static inline void benchmark_stf_launch_baseline(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
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
        stf_baseline_launch(ctx, lX, lsum);
        cudaStream_t stream = ctx.fence();
        cudaStreamSynchronize(stream);
        ctx.finalize();
    }

    GPUTimer timer;
    for (int i = 0; i < num_iterations; i++) {
        context ctx;
        double sum = 0.0;
        auto lX = ctx.logical_data(d_input_stf, {N}, data_place::device());
        auto lsum = ctx.logical_data(&sum, {1});

        timer.start("CUDA STF BASELINE");
        stf_baseline_launch(ctx, lX, lsum);
        cudaStream_t stream = ctx.fence();
        cudaStreamSynchronize(stream);
        timer.stop();
        ctx.finalize();
        result = sum;
        total_time += timer.elapsed_milliseconds();
    }

    double avg_time = total_time / num_iterations;
    report_results("CUDA STF BASELINE", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

#endif // REDUCTION_KERNELS_STF_CUH
