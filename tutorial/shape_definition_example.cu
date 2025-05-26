#include <cuda/experimental/stf.cuh>
#include <cuda_runtime.h> // For cudaMemsetAsync
#include <iostream> // For std::cout, std::endl

using namespace cuda::experimental::stf;

int main() {  
    context ctx;

    // 定义一个包含10个整数的向量的逻辑数据，仅通过形状定义  
    auto lX_from_shape = ctx.logical_data(shape_of<slice<int>>(10));  
    lX_from_shape.set_symbol("X_from_shape");

    // 首次访问必须是 write() (或无初始值的 reduce())
    ctx.task(exec_place::current_device(), lX_from_shape.write())  
        ->*[&](cudaStream_t stream, slice<int> sX_instance) {  
            cudaMemsetAsync(sX_instance.data_handle(), 0, sX_instance.size() * sizeof(int), stream);  
            std::cout << "Task: Initialized X_from_shape on device." << std::endl;  
        };

    // 定义多维逻辑数据
    auto lY_2D_from_shape = ctx.logical_data(shape_of<slice<double, 2>>(16, 24)); // 16x24 double 矩阵
    lY_2D_from_shape.set_symbol("Y_2D_from_shape");

    ctx.task(exec_place::current_device(), lY_2D_from_shape.write())  
        ->*[&](cudaStream_t stream, slice<double, 2> sY_instance) {  
            cudaMemsetAsync(sY_instance.data_handle(), 1, sY_instance.size() * sizeof(double), stream); // Init with 1.0 pattern
            std::cout << "Task: Initialized Y_2D_from_shape on device. Extent(0): "  
                      << sY_instance.extent(0) << ", Extent(1): " << sY_instance.extent(1) << std::endl;  
        };

    ctx.finalize();  
    std::cout << "Finalized shape_definition_example." << std::endl;  
    return 0;  
}
