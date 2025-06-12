#include <cuda/experimental/stf.cuh>
#include <vector>
#include <numeric> // for std::iota
#include <iostream>
#include <cmath> // for std::abs

// 简化版的核函数，用于并行计算元素乘积并累加到全局和
template<typename T>
__global__ void accumulate_product(cuda::experimental::stf::slice<const T> x, cuda::experimental::stf::slice<const T> y, T* global_sum_output) {
    extern __shared__ T sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    T local_sum = 0;
    if (gid < x.size()) { // 确保不越界
        local_sum = x(gid) * y(gid);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Shared memory reduction (simplified for one block, assumes blockDim.x is power of 2)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(global_sum_output, sdata[0]);
    }
}

int main() {
    using namespace cuda::experimental::stf;
    context ctx;

    const size_t N = 1024;
    std::vector<double> h_x(N), h_y(N);
    std::iota(h_x.begin(), h_x.end(), 1.0); // 1.0, 2.0, ..., N
    std::iota(h_y.begin(), h_y.end(), 1.0); // 1.0, 2.0, ..., N

    auto lX = ctx.logical_data(&h_x[0], {N});
    auto lY = ctx.logical_data(&h_y[0], {N});
    auto lsum = ctx.logical_data(shape_of<scalar_view<double>>()); // Create scalar logical data for sum

    lX.set_symbol("X_dot");
    lY.set_symbol("Y_dot");
    lsum.set_symbol("Sum_dot");

    // Task to compute dot product and store in lsum
    ctx.task(exec_place::current_device(),
             lX.read(),
             lY.read(),
             lsum.write() // Initialize lsum (or use reduce with no_init_tag for first access)
            )
        ->*[&](cudaStream_t s, slice<const double> sX, slice<const double> sY, scalar_view<double> sSum_scalar) {
            cudaMemsetAsync(sSum_scalar.addr, 0, sizeof(double), s); // Initialize sum to 0 on device

            dim3 threads_per_block(256);
            dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x);
            size_t shared_mem_size = threads_per_block.x * sizeof(double);

            std::cout << "Submitting accumulate_product kernel..." << std::endl;
            accumulate_product<<<num_blocks, threads_per_block, shared_mem_size, s>>>(sX, sY, sSum_scalar.addr);
        };

    // Block and wait for lsum's computation to complete and get its value
    std::cout << "Calling ctx.wait(lsum)..." << std::endl;
    double result = ctx.wait(lsum); // Host waits here until lsum is ready

    // finalize() ensures all other operations (like potential write-backs for lX, lY if they were modified) are also complete.
    // In this specific case, since lX and lY are read-only for the task, and lsum's value is already fetched by wait(), 
    // finalize() might not do much more for these specific handles but is good practice for overall context cleanup.
    ctx.finalize(); 

    std::cout << "Dot product result (obtained via ctx.wait): " << result << std::endl;

    double expected_sum = 0;
    for(size_t i = 0; i < N; ++i) expected_sum += h_x[i] * h_y[i];
    std::cout << "Expected dot product: " << expected_sum << std::endl;

    if (std::abs(result - expected_sum) < 1e-5) {
        std::cout << "Result is correct!" << std::endl;
    } else {
        std::cout << "Result is INCORRECT! Difference: " << std::abs(result - expected_sum) << std::endl;
    }

    return 0;
}
