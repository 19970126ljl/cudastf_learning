// stf_paper_evaluation/multi-gpu-reduction/kernels/stf_launch_wide_load.cuh

#ifndef REDUCTION_KERNELS_STF_WIDE_LOAD_CUH
#define REDUCTION_KERNELS_STF_WIDE_LOAD_CUH

#include "common.cuh"
#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// CUDA STF reduction implementation
// Assumes lX and lsum are created and managed by the caller.
// This function only encapsulates the kernel launch.
template<typename T1, typename T2>
void stf_launch_wide_load(context& ctx, T1 lX, T2 lsum) {
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

// Benchmark CUDA STF
static inline void benchmark_stf_launch_wide_load(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
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
        stf_launch_wide_load(ctx, lX, lsum);
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

        timer.start("CUDA STF WIDE");
        stf_launch_wide_load(ctx, lX, lsum);
        cudaStream_t stream = ctx.fence();
        cudaStreamSynchronize(stream);
        timer.stop();
        ctx.finalize();
        result = sum;
        total_time += timer.elapsed_milliseconds();
    }

    double avg_time = total_time / num_iterations;
    report_results("CUDA STF WIDE", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

#endif // REDUCTION_KERNELS_STF_CUH
