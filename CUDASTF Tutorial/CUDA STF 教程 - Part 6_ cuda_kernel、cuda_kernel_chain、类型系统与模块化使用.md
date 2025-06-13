## **CUDA STF 教程 - Part 6: cuda_kernel、cuda_kernel_chain、类型系统与模块化使用**

在 Part 5 中，我们学习了强大的 parallel_for 和 launch 构造。现在，我们将回顾并深入探讨另外两种直接与 CUDA 核函数交互的构造：cuda_kernel 和 cuda_kernel_chain。之后，我们会讨论 CUDA STF 如何利用 C++ 的类型系统来增强代码的健壮性和清晰度，最后介绍一些模块化使用 STF 的高级技巧，如数据冻结和令牌。

本部分主要依据您提供的文档中 "cuda_kernel construct" (Page 33-34)、"cuda_kernel_chain construct" (Page 34-35)、"C++ Types of logical data and tasks" (Page 35-38) 和 "Modular use of CUDASTF" (Page 38-40) 章节的内容。

### **10. cuda_kernel 构造 (文档 Page 33-34)**

我们之前在分析 01-axpy.cu 等示例时已经接触过 ctx.cuda_kernel (或 stf_ctx.cuda_kernel)。这个构造提供了一种直接的方式来将单个预定义的 CUDA 核函数作为 STF 任务执行。

**目的与优势：**
`cuda_kernel` 构造对于执行已有的 CUDA 核函数特别有用。当使用 CUDA Graph 后端 (`graph_ctx`) 时，`ctx.task()` 依赖于图捕获机制，这可能会带来一些开销。而 `cuda_kernel` 构造直接转换成 CUDA 核函数启动 API，从而避免了这种开销，可能更高效。

**语法回顾：**
```cpp
// ctx.cuda_kernel([execution_place], logicalData1.accessMode(), ...)
//     ->*[capture_list] () { // Lambda 不接受流或数据实例作为参数
//         // Lambda 的任务是返回一个 cuda_kernel_desc 对象
//         return cuda_kernel_desc{
//             kernel_function_ptr,
//             gridDim,
//             blockDim,
//             sharedMemBytes,
//             kernel_arg1, // 可以是标量值
//             logical_data_handle_for_arg2, // STF 会处理为设备上的 slice
//             ...
//         };  
// };
```
*   `cuda_kernel` 接受与 `ctx.task` 类似的参数，包括可选的执行位置和一系列数据依赖。
*   其 `->*` 操作符接受的 lambda 函数**不接收** CUDA 流或数据实例作为参数。
*   这个 lambda 函数的职责是**返回一个 `cuda_kernel_desc` 对象**。
*   `cuda_kernel_desc` 的构造函数参数包括：
    1.  `Fun func`: 指向 `__global__` CUDA 核函数的指针。
    2.  `dim3 gridDim_`: 网格维度。
    3.  `dim3 blockDim_`: 块维度。
    4.  `size_t sharedMem_`: 动态分配的共享内存大小。
    5.  `Args... args`: 传递给 CUDA 核函数的参数。这些参数可以是普通值，也可以是逻辑数据句柄（STF 会自动将它们解析为设备上的数据实例，通常是 `slice`）。

**示例 (来自文档 Page 34，稍作调整和解释):**
```cpp
#include <cuda/experimental/stf.cuh>  
#include <vector>  
#include <cmath> 
#include <iostream>

// 假设的 AXPY 核函数，接收 slice 参数  
template <typename T>  
__global__ void axpy_kernel_for_desc(T alpha, cuda::experimental::stf::slice<const T> x, cuda::experimental::stf::slice<T> y) {  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx < x.size()) { 
        y(idx) += alpha * x(idx); 
    }  
}

int main() {  
    using namespace cuda::experimental::stf;  
    context ctx;

    const size_t N = 256;  
    double alpha_val = 3.14;

    std::vector<double> h_x(N), h_y(N);
    for(size_t i=0; i<N; ++i) { h_x[i] = (double)i; h_y[i] = (double)i*2; }

    auto lX = ctx.logical_data(h_x);  
    auto lY = ctx.logical_data(h_y);  
    lX.set_symbol("X_cudakernel");  
    lY.set_symbol("Y_cudakernel");

    dim3 grid(1);  
    dim3 block(N < 256 ? N : 256);

    ctx.cuda_kernel(exec_place::current_device(), lX.read(), lY.rw())  
        .set_symbol("axpy_via_cudakernel")  
        ->*[&]() { 
             return cuda_kernel_desc{  
                 axpy_kernel_for_desc<double>, 
                 grid,        
                 block,       
                 0,           
                 alpha_val,   
                 lX,          
                 lY           
             };  
    };

    ctx.finalize();  

    if (std::abs(h_y[1] - (1.0*2.0 + alpha_val * 1.0)) < 1e-9) {  
        std::cout << "cuda_kernel example: Correct!" << std::endl;  
    } else {  
        std::cout << "cuda_kernel example: Incorrect! Y[1] is " << h_y[1] << " but expected " << (1.0*2.0 + alpha_val * 1.0) << std::endl;  
    }

    return 0;  
}
```
**请打开您本地的 stf/01-axpy-cuda_kernel.cu。** 这个示例应该与上述结构非常相似。

### **11. cuda_kernel_chain 构造 (文档 Page 34-35)**

除了 `cuda_kernel`，CUDASTF 还提供了 `cuda_kernel_chain` 构造，用于在一个 STF 任务中**顺序执行一系列 CUDA 核函数**。

**语法：**
与 `cuda_kernel` 类似，但其 lambda 函数应返回一个 `std::vector<cuda_kernel_desc>`。向量中的每个 `cuda_kernel_desc` 对象描述一个核函数启动，它们将按照在向量中出现的顺序依次执行。
```cpp
// ctx.cuda_kernel_chain([execution_place], logicalData1.accessMode(), ...)
//     ->*[capture_list] () {
//         return std::vector<cuda_kernel_desc>{
//             {kernel1_ptr, grid1, block1, shmem1, args1...}, // 第一个核函数
//             {kernel2_ptr, grid2, block2, shmem2, args2...}, // 第二个核函数
//             // ...
//         };
// };
```
**示例 (来自文档 Page 35，概念性):**
假设我们想顺序执行三次 AXPY 操作：`Y=Y+αX`, `Y=Y+βX`, `Y=Y+γX`。
```cpp
#include <cuda/experimental/stf.cuh>
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>

using namespace cuda::experimental::stf;

__global__ void axpy(double a, slice<const double> x, slice<double> y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;
    for (int i = tid; i < x.size(); i += nthreads) {
        y(i) += a * x(i);
    }
}

double X0(int i) { return sin((double)i); }
double Y0(int i) { return cos((double)i); }

int main() {
    context ctx;
    const size_t N = 16;
    std::vector<double> h_x(N), h_y(N);
    for (size_t i = 0; i < N; i++) { h_x[i] = X0(i); h_y[i] = Y0(i); }

    double alpha = 3.14, beta = 4.5, gamma = -4.1;

    auto lX = ctx.logical_data(h_x);
    auto lY = ctx.logical_data(h_y);

    dim3 grid(1), block(128);

    ctx.cuda_kernel_chain(lX.read(), lY.rw())
        .set_symbol("axpy_chain_task")
        ->*[&]() {
            return std::vector<cuda_kernel_desc>{
                {axpy, grid, block, 0, alpha, lX, lY},
                {axpy, grid, block, 0, beta, lX, lY},
                {axpy, grid, block, 0, gamma, lX, lY}
            };
        };

    ctx.finalize();

    for (size_t i = 0; i < N; i++) {
        assert(fabs(h_y[i] - (Y0(i) + (alpha + beta + gamma) * X0(i))) < 1e-4);
    }
    std::cout << "cuda_kernel_chain example: Correct!" << std::endl;
    return 0;
}
```
**请打开您本地的 stf/01-axpy-cuda_kernel_chain.cu。** 这个示例将具体展示 `cuda_kernel_chain` 的用法。

### **12. C++ 类型与逻辑数据和任务 (文档 Page 35-38)**

为了防止常见错误，CUDASTF 努力使其处理语义与 C++ 类型尽可能紧密地对齐。如各种示例所示，通常建议使用 `auto` 关键字来创建可读的代码，同时仍然强制执行类型安全。

#### **12.1 逻辑数据的类型**
调用 `ctx.logical_data()` 的结果是一个对象，其类型包含了用于操作该逻辑数据对象的底层数据接口的信息。例如，一个连续的 `double` 数组在内部表示为 `slice<double>`。

#### **12.2 任务的类型**
任务的类型 (`stream_task`, `graph_task`, `unified_task`) 及其模板参数都与数据依赖的类型相对应。这使得编译器可以在编译时捕捉到 lambda 参数类型不匹配等错误。

#### **12.3 动态类型任务**
在某些无法静态确定依赖的情况下，STF 提供了动态类型任务 (`stream_task<>`)，允许使用 `add_deps()` 动态添加依赖。但这会牺牲部分编译时检查的优势。

### **13. 模块化使用 CUDASTF (文档 Page 38-40)**

#### **13.1 冻结逻辑数据 (Freezing logical data)**
当一块数据被频繁读取时，为了避免每次访问都强制执行数据依赖关系的开销，可以“冻结”逻辑数据。
*   `auto frozen_ld = ctx.freeze(logical_data_handle, [access_mode, data_place]);`
*   默认情况下，返回的 `frozen_ld` 是只读的。
*   `frozen_ld.get(data_place, stream)` 返回底层数据在指定位置上的视图（如 `slice`），可在指定流上异步使用。
*   `frozen_ld.unfreeze(stream)` 解冻数据，必须确保所有在 `get()` 中使用的流上的工作都已完成。

**请参考示例 `stf/frozen_data_init.cu`。**

#### **13.2 令牌 (Tokens)**
令牌是一种特殊类型的逻辑数据，其唯一目的是**自动化同步**，而让应用程序管理实际的数据。当用户有自己的缓冲区（例如在单个设备上，不需要分配或传输）但可能发生并发访问时，令牌非常有用。
*   `auto token = ctx.token();`
*   令牌内部依赖于 `void_interface` 数据接口，该接口经过优化，可以跳过缓存一致性协议中不必要的数据分配或复制阶段，从而最大限度地减少运行时开销。

**请参考示例 `stf/void_data_interface.cu`。**

### **动手试试:**

1.  **编译并运行本部分提供的 `p6_01_cuda_kernel.cu` 和 `p6_02_cuda_kernel_chain.cu` 示例。**
2.  **研究 `stf/01-axpy-cuda_kernel_chain.cu`**: 与为每个核函数创建一个单独的 `ctx.cuda_kernel` 任务相比，`cuda_kernel_chain` 有什么潜在的好处？ (提示: 性能，尤其是在 `graph_ctx` 后端)
3.  **思考类型安全**: 在您看过的 STF 示例中，`auto` 关键字是如何帮助简化代码同时保持类型安全的？尝试在一个示例中，将 `auto` 替换为显式的 `logical_data<...>` 或任务参数的 `slice<...>` 类型，以加深理解。
4.  **研究 `stf/frozen_data_init.cu`**: 数据是如何被冻结和解冻的？`get()` 方法在其中扮演什么角色？
5.  **(概念思考)** 在什么情况下，使用 `ctx.token()` 比使用常规的 `logical_data` 进行同步更有优势？

我们已经完成了对 `cuda_kernel`、`cuda_kernel_chain`、STF 的类型系统以及模块化使用技巧（如冻结数据和令牌）的学习。这些工具和概念为构建复杂、高效且可维护的 CUDA 应用程序提供了坚实的基础。

在教程的最后一部分 (Part 7)，我们将简要介绍 CUDA STF 提供的**工具**，主要是任务图的可视化和使用 ncu 进行核函数性能分析。