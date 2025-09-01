// stf_paper_evaluation/multi-gpu-reduction/kernels/stf_launch_grid_stride_12.cuh

#ifndef REDUCTION_KERNELS_STF_GRID_STRIDE_12_CUH
#define REDUCTION_KERNELS_STF_GRID_STRIDE_12_CUH

#include "common.cuh"
#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// CUDA STF reduction implementation using grid-stride loops loading 12 doubles
template<typename T1, typename T2>
void stf_launch_grid_stride_12(context& ctx, T1 lX, T2 lsum) {
    auto where = exec_place::device(0);
    auto spec = par<4320>(con<256>(hw_scope::thread));
    
    ctx.launch(spec, where, lX.read(), lsum.rw())->*[] _CCCL_DEVICE(auto th, auto x, auto sum) {
        // Each thread computes the sum of elements assigned to it
        // --- Data partitioning and computation (CUB-style, main/tail separated) ---
        double local_sum = 0.0;
        const double* x_ptr = &x(0);
        size_t last_dim = x.extent(0);

        constexpr int BLOCK_THREADS = 256;
        constexpr int ITEMS_PER_THREAD = 12;
        constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

        size_t num_full_tiles = last_dim / TILE_SIZE;
        size_t grid_stride_tile = gridDim.x;

        // Main loop for full tiles (no branching)
        for (size_t tile_idx = blockIdx.x; tile_idx < num_full_tiles; tile_idx += grid_stride_tile) {
            size_t tile_start_idx = tile_idx * TILE_SIZE;
            
            const double4* d4_ptr = reinterpret_cast<const double4*>(x_ptr);
            size_t tile_start_d4 = tile_start_idx / 4;

            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                double4 item = d4_ptr[tile_start_d4 + threadIdx.x + i * BLOCK_THREADS];
                local_sum += (item.x + item.y + item.z + item.w);
            }
        }

        // Tail processing for remaining elements
        size_t tail_start_idx = num_full_tiles * TILE_SIZE;
        if (tail_start_idx < last_dim) {
            size_t grid_stride_elem = gridDim.x * blockDim.x;
            for (size_t i = tail_start_idx + blockIdx.x * blockDim.x + threadIdx.x; i < last_dim; i += grid_stride_elem) {
                local_sum += x_ptr[i];
            }
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
static inline void benchmark_stf_launch_grid_stride_12(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
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
        stf_launch_grid_stride_12(ctx, lX, lsum);
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

        timer.start("CUDA STF GRID STRIDE 12");
        stf_launch_grid_stride_12(ctx, lX, lsum);
        cudaStream_t stream = ctx.fence();
        cudaStreamSynchronize(stream);
        timer.stop();
        ctx.finalize();
        result = sum;
        total_time += timer.elapsed_milliseconds();
    }

    double avg_time = total_time / num_iterations;
    report_results("CUDA STF GRID STRIDE 12", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

#endif // REDUCTION_KERNELS_STF_GRID_STRIDE_12_CUH
