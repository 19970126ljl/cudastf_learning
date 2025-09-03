// stf_paper_evaluation/multi-gpu-reduction/kernels/stf_launch_grid_stride.cuh

#ifndef REDUCTION_KERNELS_STF_GRID_STRIDE_CUH
#define REDUCTION_KERNELS_STF_GRID_STRIDE_CUH

#include "common.cuh"
#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/stf.cuh>
#include <cub/block/block_reduce.cuh>

using namespace cuda::experimental::stf;

// CUDA STF reduction implementation using grid-stride loops
template<typename T1, typename T2>
void stf_launch_grid_stride(context& ctx, T1 lX, T2 lsum) {
    auto where = exec_place::device(0);
    auto spec = par<4320>(con<256>(hw_scope::thread));
    
    ctx.launch(spec, where, lX.read(), lsum.rw())->*[] _CCCL_DEVICE(auto th, auto x, auto sum) {
        // Each thread computes the sum of elements assigned to it
        double local_sum = 0.0;

        // Get raw pointers and size
        const double* x_ptr = &x(0);
        size_t last_dim = x.extent(0);

        // --- Data partitioning and computation (grid-stride) ---
        constexpr size_t vectorized_size = 4;
        size_t vectorized_last_dim = last_dim / vectorized_size;
        size_t remaining_start = vectorized_last_dim * vectorized_size;
        size_t grid_stride_loop_size = gridDim.x * blockDim.x;

        // Handle remaining elements with grid-stride loop
        if (remaining_start < last_dim) {
            size_t total_remaining_elements = last_dim - remaining_start;
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_remaining_elements; i += grid_stride_loop_size) {
                local_sum += x_ptr[remaining_start + i];
            }
        }

        // Process vectorized chunks with grid-stride loop
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vectorized_last_dim; i += grid_stride_loop_size) {
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

// CUDA STF reduction implementation using grid-stride loops with CUB block reduction and cache optimization
template<typename T1, typename T2>
void stf_launch_grid_stride_cub(context& ctx, T1 lX, T2 lsum) {
    auto where = exec_place::device(0);
    auto spec = par<4320>(con<256>(hw_scope::thread));
    
    ctx.launch(spec, where, lX.read(), lsum.rw())->*[] _CCCL_DEVICE(auto th, auto x, auto sum) {
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
        // --- End of data partitioning and computation ---

        // --- CUB block reduction ---
        using BlockReduce = cub::BlockReduce<double, 256>; // Block size is 256 as defined in spec
        __shared__ typename BlockReduce::TempStorage temp_storage;

        double block_sum = BlockReduce(temp_storage).Sum(local_sum);
        if (threadIdx.x == 0) {
            atomicAdd(&sum(0), block_sum);
        }
    };
}

// Benchmark CUDA STF
static inline void benchmark_stf_launch_grid_stride(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
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
        stf_launch_grid_stride(ctx, lX, lsum);
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

        timer.start("CUDA STF GRID STRIDE");
        stf_launch_grid_stride(ctx, lX, lsum);
        cudaStream_t stream = ctx.fence();
        cudaStreamSynchronize(stream);
        timer.stop();
        ctx.finalize();
        result = sum;
        total_time += timer.elapsed_milliseconds();
    }

    double avg_time = total_time / num_iterations;
    report_results("CUDA STF GRID STRIDE", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

// Benchmark CUDA STF with CUB block reduction
static inline void benchmark_stf_launch_grid_stride_cub(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
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
        stf_launch_grid_stride_cub(ctx, lX, lsum);
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

        timer.start("CUDA STF GRID STRIDE CUB");
        stf_launch_grid_stride_cub(ctx, lX, lsum);
        cudaStream_t stream = ctx.fence();
        cudaStreamSynchronize(stream);
        timer.stop();
        ctx.finalize();
        result = sum;
        total_time += timer.elapsed_milliseconds();
    }

    double avg_time = total_time / num_iterations;
    report_results("CUDA STF GRID STRIDE CUB", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

// CUDA STF reduction implementation using striped items-per-thread=10 with CUB block reduction
template<typename T1, typename T2>
void stf_launch_grid_stride_cub_double2(context& ctx, T1 lX, T2 lsum) {
    auto where = exec_place::device(0);
    auto spec = par<4320>(con<256>(hw_scope::thread));
    
    ctx.launch(spec, where, lX.read(), lsum.rw())->*[] _CCCL_DEVICE(auto th, auto x, auto sum) {
        // Each thread processes exactly 10 elements in striped pattern
        constexpr int ITEMS_PER_THREAD = 8;
        
        // Get raw pointers and size
        const double* x_ptr = &x(0);
        size_t last_dim = x.extent(0);
        
        // Calculate striped access parameters
        size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_threads = gridDim.x * blockDim.x;
        size_t stride_per_iteration = total_threads * ITEMS_PER_THREAD;
        
        double local_sum = 0.0;
        
        // Outer loop: Process data blocks with stride = total_threads * ITEMS_PER_THREAD
        for (size_t block_start = 0; block_start < last_dim; block_start += stride_per_iteration) {
            
            // Inner loop: Each thread processes ITEMS_PER_THREAD striped elements
            #pragma unroll
            for (int item = 0; item < ITEMS_PER_THREAD; item++) {
                size_t global_idx = block_start + thread_id + item * total_threads;
                if (global_idx < last_dim) {
                    local_sum += __ldcs(&x_ptr[global_idx]);  // Cache optimized load
                }
            }
        }

        // Remainder loop: Handle leftover data that doesn't fit in complete iterations
        size_t processed_elements = (last_dim / stride_per_iteration) * stride_per_iteration;
        if (processed_elements < last_dim) {
            size_t remaining_elements = last_dim - processed_elements;
            
            // Each thread processes remaining elements in striped fashion
            for (size_t remainder_idx = thread_id; remainder_idx < remaining_elements; remainder_idx += total_threads) {
                size_t global_idx = processed_elements + remainder_idx;
                local_sum += __ldcs(&x_ptr[global_idx]);  // Cache optimized load
            }
        }

        // --- CUB block reduction ---
        using BlockReduce = cub::BlockReduce<double, 256>; // Block size is 256 as defined in spec
        __shared__ typename BlockReduce::TempStorage temp_storage;

        double block_sum = BlockReduce(temp_storage).Sum(local_sum);
        if (threadIdx.x == 0) {
            atomicAdd(&sum(0), block_sum);
        }
    };
}

// Benchmark CUDA STF with double2 vectorization and CUB block reduction
static inline void benchmark_stf_launch_grid_stride_cub_double2(unsigned long long N, thrust::device_vector<double>& d_X, double ref_sum, bool verify_with_cpu, int num_iterations, std::ofstream& csv_file) {
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
        stf_launch_grid_stride_cub_double2(ctx, lX, lsum);
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

        timer.start("CUDA STF GRID STRIDE CUB DOUBLE2 IPT10");
        stf_launch_grid_stride_cub_double2(ctx, lX, lsum);
        cudaStream_t stream = ctx.fence();
        cudaStreamSynchronize(stream);
        timer.stop();
        ctx.finalize();
        result = sum;
        total_time += timer.elapsed_milliseconds();
    }

    double avg_time = total_time / num_iterations;
    report_results("CUDA STF GRID STRIDE CUB DOUBLE2 IPT10", N, avg_time, result, ref_sum, verify_with_cpu, csv_file);
}

#endif // REDUCTION_KERNELS_STF_GRID_STRIDE_CUH
