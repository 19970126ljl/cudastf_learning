// stf_paper_evaluation/multi-gpu-reduction/kernels/stf_launch_two_stage.cuh

#ifndef REDUCTION_KERNELS_STF_TWO_STAGE_CUH
#define REDUCTION_KERNELS_STF_TWO_STAGE_CUH

#include "common.cuh"
#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/stf.cuh>
#include <cub/block/block_reduce.cuh>

using namespace cuda::experimental::stf;

// Two-stage CUDA STF reduction implementation
// Stage 1: Block-level reduction to intermediate buffer
// Stage 2: Final reduction of intermediate results
template<typename T1, typename T2, typename T3>
void stf_launch_two_stage_cub(context& ctx, T1 lX, T2 lsum, T3 lintermediate) {
    auto where = exec_place::device(0);
    
    // Stage 1: Block-level reduction
    // Use same grid configuration as reference implementation
    auto spec = par<4320>(con<256>(hw_scope::thread));
    size_t num_blocks = 4320;
    
    ctx.launch(spec, where, lX.read(), lintermediate.write())->*[] _CCCL_DEVICE(auto th, auto x, auto intermediate) {
        // Each thread computes the sum of elements assigned to it
        double local_sum = 0.0;

        // Get raw pointers and size
        const double* x_ptr = &x(0);
        size_t last_dim = x.extent(0);

        // --- Data partitioning and computation (grid-stride) with cache optimization ---
        constexpr size_t vectorized_size = 4;
        size_t vectorized_last_dim = last_dim / vectorized_size;
        size_t remaining_start = vectorized_last_dim * vectorized_size;
        size_t grid_stride_loop_size = gridDim.x * blockDim.x;

        // Handle remaining elements with grid-stride loop (cache optimized)
        if (remaining_start < last_dim) {
            size_t total_remaining_elements = last_dim - remaining_start;
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_remaining_elements; i += grid_stride_loop_size) {
                // Use streaming load with L2 cache hint for better bandwidth
                local_sum += __ldcs(&x_ptr[remaining_start + i]);
            }
        }

        // Process vectorized chunks with grid-stride loop
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vectorized_last_dim; i += grid_stride_loop_size) {
            size_t original_idx = i * vectorized_size;
            
            const double4* vec_ptr = reinterpret_cast<const double4*>(&x_ptr[original_idx]);
            double4 vec = *vec_ptr;  
            local_sum += vec.x + vec.y + vec.z + vec.w;
        }

        // --- CUB block reduction ---
        using BlockReduce = cub::BlockReduce<double, 256>; // Block size is 256 as defined in spec
        __shared__ typename BlockReduce::TempStorage temp_storage;

        double block_sum = BlockReduce(temp_storage).Sum(local_sum);
        
        // Store block result in intermediate buffer (one result per block)
        if (threadIdx.x == 0) {
            intermediate(blockIdx.x) = block_sum;
        }
    };
    
    // Stage 2: Final reduction of intermediate results
    // Use a single block to reduce the intermediate results
    auto final_spec = par<1>(con<256>(hw_scope::thread));
    
    ctx.launch(final_spec, where, lintermediate.read(), lsum.rw())->*[num_blocks] _CCCL_DEVICE(auto th, auto intermediate, auto sum) {
        // Each thread processes part of the intermediate results
        double local_sum = 0.0;
        
        // Grid-stride loop over intermediate results
        for (size_t i = threadIdx.x; i < num_blocks; i += blockDim.x) {
            local_sum += intermediate(i);
        }
        
        // Final block reduction
        using BlockReduce = cub::BlockReduce<double, 256>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        
        double final_sum = BlockReduce(temp_storage).Sum(local_sum);
        
        // Store final result
        if (threadIdx.x == 0) {
            sum(0) = final_sum;
        }
    };
}

// Benchmark CUDA STF with two-stage reduction
static inline void benchmark_stf_launch_two_stage_cub(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
    double total_time = 0.0;
    double result = 0.0;
    double* d_input_stf = d_X.data().get();
    CUDA_CHECK(cudaSetDevice(0));

    // Number of blocks for intermediate results
    constexpr size_t num_blocks = 4320;

    // Warm-up
    {
        context ctx;
        double sum = 0.0;
        auto lX = ctx.logical_data(d_input_stf, {N}, data_place::device());
        auto lsum = ctx.logical_data(&sum, {1});
        auto lintermediate = ctx.logical_data<double>(num_blocks);
        stf_launch_two_stage_cub(ctx, lX, lsum, lintermediate);
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
        auto lintermediate = ctx.logical_data<double>(num_blocks);

        timer.start("CUDA STF TWO STAGE CUB");
        stf_launch_two_stage_cub(ctx, lX, lsum, lintermediate);
        cudaStream_t stream = ctx.fence();
        cudaStreamSynchronize(stream);
        timer.stop();
        ctx.finalize();
        result = sum;
        total_time += timer.elapsed_milliseconds();
    }

    double avg_time = total_time / num_iterations;
    report_results("CUDA STF TWO STAGE CUB", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

#endif // REDUCTION_KERNELS_STF_TWO_STAGE_CUH