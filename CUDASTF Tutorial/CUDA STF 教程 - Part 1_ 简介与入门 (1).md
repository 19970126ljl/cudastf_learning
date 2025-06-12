## **CUDA STF 教程 \- Part 1: 简介与入门**

### **1\. CUDASTF 简介 (文档 Page 1\)**

CUDASTF 是一个为 CUDA 实现的 **顺序任务流 (Sequential Task Flow, STF)** 模型。在现代硬件中，并行性大幅增加，尤其是在包含多个加速器的大型节点上。因此，以可扩展的方式最大化应用程序级别的并发性已成为关键。为了有效地隐藏延迟，实现尽可能高级别的异步性至关重要。

CUDASTF 引入了一种 **任务化模型 (tasking model)**，该模型可以 **自动化数据传输**，同时强制执行 **隐式的数据驱动依赖关系**。

它是一个 **仅头文件的 C++ 库**，构建在 CUDA API 之上，旨在简化多 GPU 应用程序的开发。CUDASTF 目前能够使用 CUDA 流 API 或 CUDA 图 API 生成并行应用程序。

#### **1.1 The Sequential Task Flow (STF) 编程模型 (文档 Page 2\)**

CUDASTF 编程模型涉及定义 **逻辑数据 (logical data)** 并提交操作这些数据的 **任务 (tasks)**。CUDASTF 会自动推断不同任务之间的依赖关系，并协调计算和数据移动，以确保尽可能多的并发性，从而实现高效执行。

STF 模型的核心思想是：通过为一系列任务标注适当的数据访问模式（只读、只写或读写），从中提取并发性。

**依赖规则示例：**

* **写后读 (Read-after-Write, RAW):** 一个任务必须等待所有先前的修改完成后才能读取数据。  
* **读后写 (Write-after-Read, WAR):** 一个需要修改数据的任务只有在所有先前的读取完成后才能进行。  
* **写后写 (Write-after-Write, WAW):** 两个修改相同数据的任务将被序列化。  
* **并发读 (Concurrent Reads):** 两个仅读取相同数据而不修改的任务可以并发执行。

将这些简单规则应用于一个（最初串行表示的）复杂算法，会产生一个任务的 **有向无环图 (Directed Acyclic Graph, DAG)**，CUDASTF 利用这个 DAG 来设计算法的并发执行。

通过向 CUDASTF 提供数据使用标注，程序员可以从 **自动并行化** 和 **透明数据管理** 中受益。CUDASTF 通过专门的缓存一致性协议自动化数据分配和传输。

**一个简单的依赖图示例 (文档 Page 2-3):**

考虑三个逻辑数据 X, Y, Z 和四个任务 T1, T2, T3, T4：

* T1\[X(rw)\]  
* T2\[X(read), Y(rw)\]  
* T3\[X(read), Z(rw)\]  
* T4\[Y(read), Z(rw)\]

依赖关系分析：

* T2 和 T3 读取 X，而 X 被 T1 修改。因此，T1 与 T2 之间、T1 与 T3 之间存在 RAW 依赖。  
* T2 和 T3 仅对 X 进行并发读取，因此它们可以并发执行。  
* T4 读取 Y 和 Z，而 Y 和 Z 分别被 T2 和 T3 修改。因此，T2 与 T4 之间、T3 与 T4 之间存在 WAR 依赖。

\[文档 Page 3 上的依赖图\]

### **2\. Getting started with CUDASTF (文档 Page 3-6)**

#### **2.1 Getting CUDASTF (文档 Page 3\)**

CUDASTF 是 CCCL 项目中 CUDA Experimental 库的一部分。它不随 CUDA Toolkit 分发，仅在 [CCCL GitHub repository](https://github.com/NVIDIA/cccl) 上提供。您已经拥有了示例代码，这意味着您已经有了 STF 的源码。

#### **2.2 Using CUDASTF (文档 Page 3\)**

CUDASTF 是一个仅头文件的 C++ 库。

Includes:  
主要包含的头文件是：  
\#include \<cuda/experimental/stf.cuh\>

Namespace:  
CUDASTF API 位于 cuda::experimental::stf 命名空间下。为了简洁，文档和本教程通常会假设使用了该命名空间：  
using namespace cuda::experimental::stf;  
// 或者更推荐具体使用  
// using cuda::experimental::stf::context;  
// using cuda::experimental::stf::logical\_data;  
// 等

Compiling (文档 Page 3-4):  
CUDASTF 要求编译器符合 C++17 或更高标准。

* **使用 nvcc 编译 (推荐用于包含自定义核函数的代码):**  
  \# 编译标志  
  nvcc \-std=c++17 \-expt-relaxed-constexpr \--extended-lambda \-I$(cudastf\_path) your\_file.cu \-o your\_executable  
  \# 链接标志 (通常 nvcc 会自动处理)  
  \# \-lcuda (如果需要显式链接 CUDA runtime)

  您的 stf/CMakeLists.txt 文件中应该已经配置了这些。  
* 不使用 nvcc 编译 (例如，仅调用现有 CUDA 库如 CUBLAS):  
  当不编写自定义 CUDA 核函数时，可以使用如 g++ 的主机编译器。此时，像 parallel\_for 或 launch 这样自动生成 CUDA 核函数的 STF API 将被禁用。  
  \# 编译标志  
  g++ \-std=c++17 \-I$(cudastf\_path) your\_file.cpp \-o your\_executable  
  \# 链接标志  
  g++ \-lcuda \-lcudart

* 在 CMake 项目中使用 CUDASTF (推荐方式):  
  CCCL 项目使用 CMake。您的 stf/CMakeLists.txt 就是一个很好的例子。它会找到 CUDASTF 并将其包含到目标中。

#### **2.3 A simple example: AXPY (文档 Page 4-5)**

文档中提供了一个 AXPY (Y=Y+α⋅X) 的例子。我们来看一下您代码库中的类似示例。

**请打开您本地的 stf/01-axpy.cu 文件。**

// 内容来自 stf/01-axpy.cu (或类似文件)  
// 我会在这里高亮关键部分，请您对照您的文件阅读

\#include \<cuda/experimental/stf.cuh\> // 主要 STF 头文件  
\#include \<iostream\>  
\#include \<vector\>  
\#include \<cmath\> // For std::sin, std::cos

// 通常我们会把 using namespace 放在函数内部或更小的作用域  
// using namespace cuda::experimental::stf;

// CUDA Kernel for AXPY  
template \<typename T\>  
\_\_global\_\_ void axpy\_kernel(T alpha, const T\* x, T\* y, size\_t n) {  
    int idx \= blockIdx.x \* blockDim.x \+ threadIdx.x;  
    if (idx \< n) {  
        y\[idx\] \= alpha \* x\[idx\] \+ y\[idx\];  
    }  
}

// 一个更符合 STF slice 风格的 kernel (如文档 Page 5 所示)  
template \<typename T\>  
\_\_global\_\_ void axpy\_stf\_kernel(T alpha, cuda::experimental::stf::slice\<const T\> x, cuda::experimental::stf::slice\<T\> y) {  
    int tid \= blockIdx.x \* blockDim.x \+ threadIdx.x;  
    int nthreads \= gridDim.x \* blockDim.x;

    for (int ind \= tid; ind \< x.size(); ind \+= nthreads) {  
        y(ind) \+= alpha \* x(ind); // 使用 slice 的 operator()  
    }  
}

int main(int argc, char\*\* argv) {  
    // 使用特定命名空间，避免污染全局  
    using cuda::experimental::stf::context;  
    using cuda::experimental::stf::logical\_data;  
    using cuda::experimental::stf::slice;  
    using cuda::experimental::stf::exec\_place; // 如果用到

    const size\_t N \= 1 \<\< 16; // 示例大小  
    double alpha \= 3.14;

    // 1\. 初始化主机数据  
    std::vector\<double\> h\_x(N);  
    std::vector\<double\> h\_y(N);  
    std::vector\<double\> h\_y\_expected(N);

    for (size\_t i \= 0; i \< N; \++i) {  
        h\_x\[i\] \= std::sin((double)i);  
        h\_y\[i\] \= std::cos((double)i);  
        h\_y\_expected\[i\] \= alpha \* h\_x\[i\] \+ h\_y\[i\];  
    }

    // 分配设备内存并拷贝数据  
    double \*d\_x\_ptr, \*d\_y\_ptr;  
    cudaMalloc(\&d\_x\_ptr, N \* sizeof(double));  
    cudaMalloc(\&d\_y\_ptr, N \* sizeof(double));  
    cudaMemcpy(d\_x\_ptr, h\_x.data(), N \* sizeof(double), cudaMemcpyHostToDevice);  
    cudaMemcpy(d\_y\_ptr, h\_y.data(), N \* sizeof(double), cudaMemcpyHostToDevice);

    // 2\. 声明一个 CUDASTF 上下文 (context)  
    // 文档 Page 5 使用 context ctx;  
    // 您的 01-axpy.cu 可能直接使用 scheduler 和 execute  
    cuda::experimental::stf::scheduler sched;

    // 3\. 创建逻辑数据 (logical data)  
    // STF 将管理这些数据的不同实例 (主机/设备)  
    // 注意：文档 Page 5 的 ctx.logical\_data(X) 是一个更简洁的 API  
    // 您的 01-axpy.cu 可能使用 make\_data\_handle  
    // 我们将遵循您示例代码的风格，如果它与文档最新版本略有不同

    // 使用原始指针和大小创建 slice  
    slice\<double\> s\_x\_device(d\_x\_ptr, {N});  
    slice\<double\> s\_y\_device(d\_y\_ptr, {N});

    // 将设备上的 slice 包装成 STF 的 logical\_data  
    // 这里假设我们直接操作设备数据，STF 负责同步  
    // 这种方式更接近于直接控制，而 ctx.logical\_data(host\_array) 则让STF做更多拷贝工作  
    auto ld\_x \= cuda::experimental::stf::make\_data\_handle\<cuda::experimental::stf::device\_memory\>(s\_x\_device);  
    auto ld\_y \= cuda::experimental::stf::make\_data\_handle\<cuda::experimental::stf::device\_memory\>(s\_y\_device);

    // 为逻辑数据设置符号名称，便于调试和可视化 (文档 Page 41\)  
    ld\_x.set\_symbol("X");  
    ld\_y.set\_symbol("Y");

    // 定义kernel启动参数  
    dim3 threads\_per\_block(256);  
    dim3 num\_blocks((N \+ threads\_per\_block.x \- 1\) / threads\_per\_block.x);

    // 4\. 提交任务 (submit task)  
    // 文档 Page 5: ctx.task(lX.read(), lY.rw())-\>\*\[&\] (...) { ... };  
    // 您的 01-axpy.cu 可能使用 sched.execute 和 ctx.launch 或 ctx.cuda\_kernel  
    // 我们这里模拟 ctx.cuda\_kernel 的方式，因为它更直接对应一个kernel launch

    // 使用 sched.execute 来定义一个任务批次  
    sched.execute(\[&\](auto& stf\_ctx) { // stf\_ctx 通常是 stf::context& 或 stf::multi\_gpu\_context&  
        stf\_ctx.cuda\_kernel(exec\_place::current\_device(), // 执行位置  
                           ld\_x.reads(),      // X 是只读  
                           ld\_y.reads\_writes() // Y 是读写  
                           )  
            .set\_symbol("axpy\_task") // 给任务命名 (文档 Page 42\)  
            \-\>\*\[&\]() { // 定义任务体，返回一个 kernel\_desc  
                // axpy\_kernel\<\<\<num\_blocks, threads\_per\_block, 0, stream\>\>\>(alpha, dX.data\_handle(), dY.data\_handle());  
                // 返回 cuda\_kernel\_desc 对象  
                return cuda::experimental::stf::cuda\_kernel\_desc{  
                    axpy\_stf\_kernel\<double\>, // 内核函数指针  
                    num\_blocks,  
                    threads\_per\_block,  
                    0, // 共享内存大小  
                    // 内核参数:  
                    alpha,  
                    // STF 会将 ld\_x 和 ld\_y 的设备实例传递给 axpy\_stf\_kernel  
                    // 这里我们不需要显式传递 stf\_ctx.get\<slice\<const double\>\>(ld\_x)  
                    // 因为 cuda\_kernel 的 lambda 参数就是数据实例  
                    // 但为了清晰，我们通常会接收它们：  
                    // \-\>\*\[&\](cuda::experimental::stf::slice\<const double\> sX\_kernel, cuda::experimental::stf::slice\<double\> sY\_kernel) {  
                    // return cuda::experimental::stf::cuda\_kernel\_desc{axpy\_stf\_kernel\<double\>, ..., alpha, sX\_kernel, sY\_kernel};  
                    // }  
                    // 然而，cuda\_kernel 的 lambda 签名是 () \-\> cuda\_kernel\_desc  
                    // 数据实例需要通过 task\_object.get\<Type\>(index) 获取，或者如果直接传递给 kernel\_desc，STF会处理  
                    // 最简单的形式是直接将 data\_handle 传递给 kernel\_desc，STF 会处理  
                    ld\_x, // STF 会处理为 slice\<const double\>  
                    ld\_y  // STF 会处理为 slice\<double\>  
                };  
            };  
    });

    // 5\. 等待所有挂起的工作完成 (finalize)  
    // 文档 Page 5: ctx.finalize();  
    // 在基于 scheduler 的模型中，通常 execute() 是阻塞的，或者需要显式等待  
    // 如果 sched.execute 是异步的，则可能需要 sched.wait\_for\_all\_tasks();  
    // 在很多示例中，execute() 本身就包含了同步，或者 finalize 是通过 scheduler 的析构隐式完成的。  
    // 为了确保，显式调用 finalize (如果 scheduler 有此方法) 或等待。  
    // 对于这里的 scheduler 模型，通常 execute() 之后，如果需要确保完成，  
    // 可以依赖 scheduler 的析构或者特定的 wait 方法。  
    // 简单的例子中，main函数结束前，scheduler析构会自动处理。

    // 将结果从设备拷贝回主机  
    std::vector\<double\> h\_y\_result(N);  
    cudaMemcpy(h\_y\_result.data(), d\_y\_ptr, N \* sizeof(double), cudaMemcpyDeviceToHost);

    // 验证结果  
    int errors \= 0;  
    for (size\_t i \= 0; i \< N; \++i) {  
        if (std::abs(h\_y\_result\[i\] \- h\_y\_expected\[i\]) \> 1e-9) {  
            errors++;  
        }  
    }  
    std::cout \<\< "AXPY example with STF." \<\< std::endl;  
    if (errors \== 0\) {  
        std::cout \<\< "Result is correct\!" \<\< std::endl;  
    } else {  
        std::cout \<\< "Result is INCORRECT\! Errors: " \<\< errors \<\< std::endl;  
    }

    // 释放设备内存  
    cudaFree(d\_x\_ptr);  
    cudaFree(d\_y\_ptr);

    return 0;  
}

**代码步骤分解 (对照文档 Page 5):**

1. **包含 STF 头文件**: \#include \<cuda/experimental/stf.cuh\>  
2. **声明 CUDASTF 上下文/调度器**: stf::scheduler sched; (文档中使用 context ctx;，两者角色类似，scheduler 更侧重于任务的提交和执行流程，而 context 可能包含更多状态。在较新的示例中，scheduler 和 execute lambda 结合 stf::context& 参数是常见模式。)  
3. **创建逻辑数据**:  
   * 首先准备原始数据 (主机数组 h\_x, h\_y，设备指针 d\_x\_ptr, d\_y\_ptr)。  
   * 使用 stf::make\_data\_handle 将设备上的数据（通过 slice 描述）包装成 STF 的 logical\_data 对象 (ld\_x, ld\_y)。  
   * ld\_x.set\_symbol("X") 为逻辑数据命名，便于调试。  
4. **提交任务**:  
   * 使用 sched.execute(\[&\](auto& stf\_ctx){...}) 来定义一个执行块。  
   * 在 lambda 内部，使用 stf\_ctx.cuda\_kernel(...) 来定义一个 CUDA 核函数任务。  
     * exec\_place::current\_device(): 指定任务在当前选定的 CUDA 设备上执行。  
     * ld\_x.reads(): 声明逻辑数据 ld\_x 在此任务中是只读的。  
     * ld\_y.reads\_writes(): 声明逻辑数据 ld\_y 在此任务中是可读写的。  
     * .set\_symbol("axpy\_task"): 为任务本身命名。  
     * \-\>\*\[&\]() { return cuda\_kernel\_desc{...}; }: 任务体。这个 lambda **返回一个 cuda\_kernel\_desc 对象**，它描述了要启动的实际 CUDA 核函数、启动配置（网格和块维度、共享内存）以及传递给核函数的参数。STF 会根据 ld\_x 和 ld\_y 的访问模式和它们在 cuda\_kernel\_desc 中的使用，自动将正确的设备内存地址（通常是 slice 对象）传递给 axpy\_stf\_kernel。  
5. **等待完成**:  
   * 在这个基于 scheduler 的模型中，execute() 调用通常会处理任务的调度和执行。对于简单的同步场景，当 execute() 返回后，任务图中的操作就已经被安排或完成了。  
   * 结果拷贝回主机 (cudaMemcpy) 和验证。  
   * cudaFree 释放显式分配的设备内存。STF 的 logical\_data 本身不拥有通过 make\_data\_handle 传入的外部指针的生命周期，除非它是从 shape\_of 创建的（STF 会负责分配）。

**动手试试:**

1. 编译并运行 01-axpy.cu:  
   进入您的 stf/build 目录，执行 make cudax.cpp17.example.stf.01-axpy (或类似的目标名，具体看您的 CMake 输出)，然后运行 ./bin/cudax.cpp17.example.stf.01-axpy。确认输出结果是否正确。  
2. 修改数据大小或 alpha 值:  
   在 01-axpy.cu 中，尝试修改 N 的值（例如，改为 1 \<\< 10）或 alpha 的值。重新编译并运行，观察结果。  
3. **可视化任务图 (文档 Page 41\)**:  
   * 确保您的系统中安装了 Graphviz (dot 命令)。  
   * 通过设置环境变量来运行示例以生成 .dot 文件：  
     CUDASTF\_DOT\_FILE=axpy\_graph.dot ./bin/cudax.cpp17.example.stf.01-axpy

   * 将 .dot 文件转换为图像：  
     dot \-Tpng axpy\_graph.dot \-o axpy\_graph.png

   * 打开 axpy\_graph.png 查看生成的任务图。由于这个例子很简单，图可能只包含一个标记为 "axpy\_task" 的节点，以及它对 "X" (read) 和 "Y" (rw) 的依赖。

我们已经完成了入门部分！您现在对 STF 的基本工作流程、上下文/调度器、逻辑数据和任务提交有了一个初步的印象。

接下来，我们将根据文档深入探讨 **"Backends and contexts"** 和 **"Logical data"** 的更多细节。您想继续吗？