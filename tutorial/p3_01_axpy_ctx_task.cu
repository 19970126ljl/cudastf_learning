#include <cuda/experimental/stf.cuh>  
#include <vector>  
#include <cmath>  
#include <iostream>
#include <numeric> // For std::iota if needed, or remove if not

// 假设我们有这样一个 AXPY 核函数，它直接操作 slice  
template <typename T>  
__global__ void axpy_slice_kernel(T alpha, cuda::experimental::stf::slice<const T> x, cuda::experimental::stf::slice<T> y) {  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  
    int nthreads = gridDim.x * blockDim.x;  
    for (size_t ind = tid; ind < x.size(); ind += nthreads) {  
        y(ind) += alpha * x(ind);  
    }  
}

int main() {  
    using namespace cuda::experimental::stf; // 为简洁起见  
    context ctx; // 使用通用上下文

    const size_t N = 1024;  
    double alpha = 2.0;

    // 1. 创建主机数据  
    std::vector<double> h_x_vec(N);  
    std::vector<double> h_y_vec(N);  
    for(size_t i = 0; i < N; ++i) {  
        h_x_vec[i] = (double)i;  
        h_y_vec[i] = (double)(N - i);  
    }

    // 2. 创建逻辑数据 (STF将负责H2D拷贝)  
    auto lX = ctx.logical_data(h_x_vec.data(), h_x_vec.size());  
    auto lY = ctx.logical_data(h_y_vec.data(), h_y_vec.size());  
    lX.set_symbol("X_vec");  
    lY.set_symbol("Y_vec");

    // 3. 定义核函数启动配置  
    dim3 num_blocks( (N + 255) / 256 );  
    dim3 threads_per_block(256);

    // 4. 创建并提交任务  
    ctx.task(exec_place::current_device(), 
             lX.read(),      
             lY.rw()         
            )  
        ->*[&](cudaStream_t s,                         
               slice<const double> sX_kernel_arg,     
               slice<double>       sY_kernel_arg      
              ) {  
        std::cout << "AXPY Task Body: Submitting kernel to stream " << s << std::endl;  
        axpy_slice_kernel<<<num_blocks, threads_per_block, 0, s>>>(alpha, sX_kernel_arg, sY_kernel_arg);  
    }; 

    // 5. 等待所有任务完成  
    ctx.finalize();

    std::cout << "First element of Y after AXPY: " << h_y_vec[0] << std::endl;  
    double expected_y0 = alpha * 0.0 + (double)N;
    std::cout << "Expected: " << expected_y0 << std::endl;
    if (std::abs(h_y_vec[0] - expected_y0) < 1e-5) {
        std::cout << "Result is correct!" << std::endl;
    } else {
        std::cout << "Result is INCORRECT!" << std::endl;
    }

    return 0;  
}
