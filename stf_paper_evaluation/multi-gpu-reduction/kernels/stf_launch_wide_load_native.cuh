// stf_paper_evaluation/multi-gpu-reduction/kernels/stf_launch_wide_load_native.cuh

#ifndef REDUCTION_KERNELS_STF_WIDE_LOAD_NATIVE_CUH
#define REDUCTION_KERNELS_STF_WIDE_LOAD_NATIVE_CUH

#include "common.cuh"
#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// CUDA STF reduction implementation
// Assumes lX and lsum are created and managed by the caller.
// This function only encapsulates the kernel launch.
template<typename T1, typename T2>
void stf_launch_wide_load_native(context& ctx, T1 lX, T2 lsum) {
    auto where = exec_place::device(0);
    auto spec = par<4320>(con<256>(hw_scope::thread));
    
    ctx.launch(spec, where, lX.read(), lsum.rw())->*[] _CCCL_DEVICE(auto th, auto x, auto sum) {
        // Each thread computes the sum of elements assigned to it
        double local_sum = 0.0;

        // Get raw pointers and size
        const double* x_ptr = &x(0);
        size_t last_dim = x.extent(0);

        // --- Data partitioning and computation (native CUDA) ---
        constexpr size_t vectorized_size = 4;
        size_t vectorized_last_dim = last_dim / vectorized_size;
        size_t remaining_start = vectorized_last_dim * vectorized_size;

        // Handle remaining elements
        if (remaining_start < last_dim) {
            size_t total_remaining_elements = last_dim - remaining_start;
            size_t elements_per_block_rem = (total_remaining_elements + gridDim.x - 1) / gridDim.x;
            size_t block_start_idx_rem = blockIdx.x * elements_per_block_rem;
            size_t block_end_idx_rem = min((size_t)(blockIdx.x + 1) * elements_per_block_rem, total_remaining_elements);
            for (size_t i = block_start_idx_rem + threadIdx.x; i < block_end_idx_rem; i += blockDim.x) {
                local_sum += x_ptr[remaining_start + i];
            }
        }

        // Process vectorized chunks
        size_t total_vectorized_elements = vectorized_last_dim;
        size_t elements_per_block_vec = (total_vectorized_elements + gridDim.x - 1) / gridDim.x;
        size_t block_start_idx_vec = blockIdx.x * elements_per_block_vec;
        size_t block_end_idx_vec = min((size_t)(blockIdx.x + 1) * elements_per_block_vec, total_vectorized_elements);
        for (size_t i = block_start_idx_vec + threadIdx.x; i < block_end_idx_vec; i += blockDim.x) {
            size_t original_idx = i * vectorized_size;
            const double4* vec_ptr = reinterpret_cast<const double4*>(&x_ptr[original_idx]);
            double4 vec = *vec_ptr;
            local_sum += vec.x + vec.y + vec.z + vec.w;
        }
        // --- End of data partitioning and computation ---

        // --- In-block reduction using shared memory (native CUDA) ---
        __shared__ double block_sum[256]; // Block size is 256 as defined in spec

        block_sum[threadIdx.x] = local_sum;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(&sum(0), block_sum[0]);
        }
    };
}

// Benchmark CUDA STF
static inline void benchmark_stf_launch_wide_load_native(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
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
        stf_launch_wide_load_native(ctx, lX, lsum);
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

        timer.start("CUDA STF WIDE NATIVE");
        stf_launch_wide_load_native(ctx, lX, lsum);
        cudaStream_t stream = ctx.fence();
        cudaStreamSynchronize(stream);
        timer.stop();
        ctx.finalize();
        result = sum;
        total_time += timer.elapsed_milliseconds();
    }

    double avg_time = total_time / num_iterations;
    report_results("CUDA STF WIDE NATIVE", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

#endif // REDUCTION_KERNELS_STF_WIDE_LOAD_NATIVE_CUH
