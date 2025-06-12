#include <cuda/experimental/stf.cuh>
#include <cassert>
#include <iostream>
#include <vector> // Required for STF logical_data from host data usually

// 假设的简单核函数
__global__ void inc_kernel(cuda::experimental::stf::slice<int> sX_slice) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sX_slice(0) += 1;
    }
}

int main() {
    using namespace cuda::experimental::stf;
    context ctx;

    int host_X_val_arr[1] = {42}; // Use an array for STF logical_data
    auto lX = ctx.logical_data(host_X_val_arr); // STF manages this array now
    lX.set_symbol("X_scalar");

    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (num_devices < 1) {
        std::cout << "Requires at least 1 CUDA device, found " << num_devices << ". Skipping device tasks." << std::endl;
    } else {
        std::cout << "Found " << num_devices << " CUDA device(s). Proceeding with device tasks." << std::endl;
        // 任务1: 在设备0上执行
        ctx.task(exec_place::device(0), lX.rw())
            ->*[&](cudaStream_t stream, slice<int> sX_kernel_arg) {
                int current_device;
                cudaGetDevice(&current_device);
                std::cout << "Task 1 (exec_place::device(0)): Current CUDA device is " << current_device << ", incrementing X. Stream: " << stream << std::endl;
                inc_kernel<<<1, 1, 0, stream>>>(sX_kernel_arg);
        };

        if (num_devices > 1) {
            // 任务2: 在设备1上执行 (如果存在)
            ctx.task(exec_place::device(1), lX.rw())
                ->*[&](cudaStream_t stream, slice<int> sX_kernel_arg) {
                    int current_device;
                    cudaGetDevice(&current_device);
                    std::cout << "Task 2 (exec_place::device(1)): Current CUDA device is " << current_device << ", incrementing X. Stream: " << stream << std::endl;
                    inc_kernel<<<1, 1, 0, stream>>>(sX_kernel_arg);
            };
        }
    }

    // 任务3: 在主机上执行，读取 lX (使用 host_launch)
    ctx.host_launch(lX.read()) // exec_place::host() is implicit and optional for host_launch
        .set_symbol("HostReadTask")
        ->*[&](slice<const int> sX_host_arg) { // lambda 参数是主机上的数据实例
            std::cout << "Host Task: reading X." << std::endl;
            int increments = 0;
            if (num_devices >= 1) increments++; // From Task 1
            if (num_devices > 1) increments++;  // From Task 2
            int expected_val = 42 + increments;
            std::cout << "Host Task: X(0) = " << sX_host_arg(0) << ", Expected = " << expected_val << std::endl;
            assert(sX_host_arg(0) == expected_val);
    };

    ctx.finalize();
    std::cout << "Finalized. Original host_X_val_arr[0] after STF (due to write-back): " << host_X_val_arr[0] << std::endl;

    return 0;
}
