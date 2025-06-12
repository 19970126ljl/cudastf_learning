## **CUDA STF 教程 \- Part 3: 任务 (Tasks)**

在 CUDA STF 中，**任务 (Task)** 是计算的基本单元。它可以是一个 CUDA 核函数、一个主机端函数，甚至是调用一个库函数。STF 的强大之处在于它能够根据任务对逻辑数据的访问模式自动推断依赖关系，并构建一个高效的执行图。

本部分将主要依据您提供的文档中 "Tasks" 章节 (Page 12-16) 的内容。

### **5\. 任务创建与数据依赖 (文档 Page 12-13)**

任务是通过上下文对象（例如 ctx 或在 scheduler.execute lambda 中的 stf\_ctx）的 task() 成员函数（或其变体如 cuda\_kernel()、host\_launch() 等）创建的。

#### **5.1 基本任务创建语法**

一个典型的任务创建涉及以下几个方面：

1. **执行位置 (Execution Place)**: 可选参数，指定任务在哪里执行（例如，特定 GPU 或主机）。如果未提供，则默认为当前 CUDA 设备 (exec\_place::current\_device())。我们将在后续的 "Places" 部分详细讨论。  
2. **数据依赖列表 (Data Dependencies)**: 这是至关重要的一步。您需要列出此任务将访问的所有逻辑数据，并为每个逻辑数据指定其**访问模式 (access mode)**。  
   * logical\_data\_handle.read(): 只读访问。  
   * logical\_data\_handle.write(): 只写访问（会覆盖数据，不保证读取旧值）。  
   * logical\_data\_handle.rw(): 读写访问。  
   * logical\_data\_handle.reduce(reducer, \[no\_init\_tag\]): 归约访问（例如求和、最大值等）。我们将在 parallel\_for 部分详细介绍。  
3. **任务体 (Task Body)**: 通常是一个 C++ lambda 函数，它定义了任务实际执行的工作。

**通用 ctx.task() 语法示例 (来自文档 Page 12):**

\#include \<cuda/experimental/stf.cuh\>  
\#include \<vector\>  
\#include \<cmath\>  
\#include \<iostream\>

// 假设我们有这样一个 AXPY 核函数，它直接操作 slice  
template \<typename T\>  
\_\_global\_\_ void axpy\_slice\_kernel(T alpha, cuda::experimental::stf::slice\<const T\> x, cuda::experimental::stf::slice\<T\> y) {  
    int tid \= blockIdx.x \* blockDim.x \+ threadIdx.x;  
    int nthreads \= gridDim.x \* blockDim.x;  
    for (size\_t ind \= tid; ind \< x.size(); ind \+= nthreads) {  
        y(ind) \+= alpha \* x(ind);  
    }  
}

int main() {  
    using namespace cuda::experimental::stf; // 为简洁起见  
    context ctx; // 使用通用上下文

    const size\_t N \= 1024;  
    double alpha \= 2.0;

    // 1\. 创建主机数据  
    std::vector\<double\> h\_x\_vec(N);  
    std::vector\<double\> h\_y\_vec(N);  
    for(size\_t i \= 0; i \< N; \++i) {  
        h\_x\_vec\[i\] \= (double)i;  
        h\_y\_vec\[i\] \= (double)(N \- i);  
    }

    // 2\. 创建逻辑数据 (STF将负责H2D拷贝)  
    auto lX \= ctx.logical\_data(h\_x\_vec);  
    auto lY \= ctx.logical\_data(h\_y\_vec);  
    lX.set\_symbol("X\_vec");  
    lY.set\_symbol("Y\_vec");

    // 3\. 定义核函数启动配置  
    dim3 num\_blocks( (N \+ 255\) / 256 );  
    dim3 threads\_per\_block(256);

    // 4\. 创建并提交任务  
    // ctx.task(...) 返回一个任务对象，该对象重载了 operator-\>\*()  
    // operator-\>\*() 接受右侧的 lambda 函数作为任务体  
    ctx.task(exec\_place::current\_device(), // 可选的执行位置  
             lX.read(),      // X 以只读模式访问  
             lY.rw()         // Y 以读写模式访问  
            )  
        \-\>\*\[&\](cudaStream\_t s,                         // 1\. CUDA 流，用于提交异步工作  
               slice\<const double\> sX\_kernel\_arg,     // 2\. lX 的设备数据实例 (只读)  
               slice\<double\>       sY\_kernel\_arg      // 3\. lY 的设备数据实例 (读写)  
              ) {  
        // 这是任务体 lambda  
        // 注意：这个 lambda 本身是在主机上立即执行的，它的作用是 \*提交\* 异步工作到流 s  
        // 它不是 CUDA 核函数本身！

        std::cout \<\< "AXPY Task Body: Submitting kernel to stream " \<\< s \<\< std::endl;  
        axpy\_slice\_kernel\<\<\<num\_blocks, threads\_per\_block, 0, s\>\>\>(alpha, sX\_kernel\_arg, sY\_kernel\_arg);  
    }; // 任务定义结束

    // 5\. 等待所有任务完成  
    ctx.finalize();

    // (可选) 验证结果，需要将 lY 的数据拷回主机  
    // STF 的写回策略 (write-back policy) 默认会在 finalize() 时将修改后的数据写回原始的主机内存 h\_y\_vec  
    std::cout \<\< "First element of Y after AXPY: " \<\< h\_y\_vec\[0\] \<\< std::endl;  
    std::cout \<\< "Expected: " \<\< alpha \* 0.0 \+ (double)N \<\< std::endl;

    return 0;  
}

**lambda 函数的参数:**

* 第一个参数通常是 cudaStream\_t stream (或简写为 s)。这是 STF 为此任务提供的 CUDA 流。您应该将所有异步 CUDA 操作（如核函数启动、异步内存拷贝）提交到这个流中。  
* 后续参数对应于您在 ctx.task(...) 中声明的每个逻辑数据依赖。STF 会将相应逻辑数据在任务执行位置的有效**数据实例 (data instance)** 传递给 lambda。  
  * 如果逻辑数据以 .read() 模式访问，则对应的数据实例类型通常是 slice\<const T\> (或类似的基础类型)。  
  * 如果以 .rw() 或 .write() 模式访问，则是 slice\<T\>。  
  * 使用 auto 关键字可以简化类型声明：auto sX, auto sY。

重要说明 (文档 Page 13):  
任务构造的主体 (lambda 函数) 是在任务提交时立即在主机上执行的，而不是在任务实际准备好执行时（即其依赖满足时）。因此，任务主体的作用是将异步工作（例如 CUDA 核函数）提交到传递给它的 CUDA 流中，而不是它本身就是 CUDA 核函数。试图在 lambda 函数中立即直接使用传递进来的 slice 数据（例如在 CPU 上访问其内容）是错误的，除非这些数据明确位于主机并且您已确保同步。正确的方式是将这些 slice 作为参数传递给与该流同步的 CUDA 核函数。CUDA 的执行语义将确保当核函数实际运行时，这些 slice 是有效的。

#### **5.2 cuda\_kernel 和 cuda\_kernel\_chain (回顾 Part 1\)**

我们在 Part 1 的 01-axpy.cu 示例分析中已经接触过 stf\_ctx.cuda\_kernel(...)。这是一种更直接的方式来提交单个 CUDA 核函数任务。

* **ctx.cuda\_kernel(dependencies...) \-\>\* \[\](){ return cuda\_kernel\_desc{...}; }**:  
  * lambda 函数体**返回一个 cuda\_kernel\_desc 对象**。  
  * cuda\_kernel\_desc 包含了核函数指针、启动配置 (grid/block dims, shared memory) 和核函数参数。  
  * 这种方式对于 CUDA Graph 后端可能更高效，因为它避免了 ctx.task() 可能涉及的图捕获开销。  
* **ctx.cuda\_kernel\_chain(dependencies...) \-\>\* \[\](){ return std::vector\<cuda\_kernel\_desc\>{...}; }**:  
  * 类似于 cuda\_kernel，但 lambda 返回一个 std::vector\<cuda\_kernel\_desc\>。  
  * 允许在一个 STF 任务中按顺序执行多个 CUDA 核函数。

**请打开您本地的 stf/01-axpy-cuda\_kernel.cu 和 stf/01-axpy-cuda\_kernel\_chain.cu (如果存在) 或回顾 01-axpy.cu 中 cuda\_kernel 的用法。**

// 概念性回顾 stf/01-axpy.cu 中 cuda\_kernel 的用法  
// sched.execute(\[&\](auto& stf\_ctx) {  
//     stf\_ctx.cuda\_kernel(exec\_place::current\_device(),  
//                        ld\_x.reads(),  
//                        ld\_y.reads\_writes())  
//         .set\_symbol("axpy\_task\_direct\_kernel")  
//         \-\>\*\[&\]() { // 这个 lambda 返回一个 cuda\_kernel\_desc  
//             return cuda::experimental::stf::cuda\_kernel\_desc{  
//                 axpy\_stf\_kernel\<double\>,  
//                 num\_blocks,  
//                 threads\_per\_block,  
//                 0, // sharedMem  
//                 alpha,  
//                 // STF 会自动从 ld\_x, ld\_y 获取设备上的 slice 实例  
//                 // 并传递给 axpy\_stf\_kernel  
//                 // 在 cuda\_kernel\_desc 中直接使用逻辑数据句柄即可  
//                 ld\_x,  
//                 ld\_y  
//             };  
//         };  
// });

#### **5.3 多任务与依赖推断 (Example of creating and using multiple tasks \- 文档 Page 14-16)**

当您提交多个任务时，CUDASTF 会根据它们声明的数据依赖关系自动推断执行顺序，构建任务图。

**文档 Page 14 的示例逻辑：**

// 伪代码，说明依赖关系  
auto lX \= ctx.logical\_data(X\_host\_data);  
auto lY \= ctx.logical\_data(Y\_host\_data);

// Task 1: 读取 lX 和 lY，执行 K1, K2  
ctx.task(lX.read(), lY.read())-\>\*\[\](cudaStream\_t s, auto sX, auto sY) {  
    K1\<\<\<..., s\>\>\>(sX, sY);  
    K2\<\<\<..., s\>\>\>(sX, sY);  
};

// Task 2: 读写 lX，执行 K3  
ctx.task(lX.rw())-\>\*\[\](cudaStream\_t s, auto sX) {  
    K3\<\<\<..., s\>\>\>(sX);  
};

// Task 3: 读写 lY，执行 K4  
ctx.task(lY.rw())-\>\*\[\](cudaStream\_t s, auto sY) {  
    K4\<\<\<..., s\>\>\>(sY);  
};

// Task 4 (Host Task): 读取 lX 和 lY，执行 callback  
ctx.host\_launch(lX.read(), lY.read())-\>\*\[\](auto sX, auto sY) {  
    callback(sX, sY); // 假设 callback 是一个主机函数  
};

ctx.finalize();

**依赖分析 (文档 Page 14):**

* **Task 2 和 Task 3 依赖于 Task 1**:  
  * Task 2 修改 (rw) lX，而 Task 1 读取 (read) lX (WAR 依赖)。  
  * Task 3 修改 (rw) lY，而 Task 1 读取 (read) lY (WAR 依赖)。  
  * 因此，Task 2 和 Task 3 必须在 Task 1 完成后执行。  
  * 由于 Task 2 和 Task 3 操作不同的逻辑数据 (lX vs lY)，它们之间没有直接依赖，**可以并发执行** (在 Task 1 完成后)。  
* **Task 4 依赖于 Task 2 和 Task 3**:  
  * Task 4 在主机上读取 (read) lX，而 lX 被 Task 2 修改 (rw) (RAW 依赖)。  
  * Task 4 在主机上读取 (read) lY，而 lY 被 Task 3 修改 (rw) (RAW 依赖)。  
  * 因此，Task 4 必须在 Task 2 和 Task 3 都完成后执行。

文档 Page 15 展示了由此产生的任务图，Page 16 进一步展示了包含自动数据分配和传输的更详细的图。这突出了 STF 如何减轻程序员管理复杂异步操作和数据移动的负担。

请打开您本地的 stf/01-axpy-cuda\_kernel\_chain.cu。  
这个示例应该演示了如何顺序执行多个（可能是两个）AXPY 操作，例如：

1. Z \= a\*X \+ Y  
2. W \= b\*Z \+ Q

观察它是如何定义多个 cuda\_kernel (或 cuda\_kernel\_chain 中的多个 cuda\_kernel\_desc)，以及如何通过共享的逻辑数据 lZ 来建立它们之间的依赖关系。lZ 在第一个任务中是 writes() 或 rw()，在第二个任务中是 reads()。

#### **5.4 主机端任务 (Host-Side Task Execution with host\_launch) (文档 Page 14, 19\)**

除了在 GPU 上执行计算，STF 还允许您将主机 (CPU) 上的函数作为任务集成到图中。这是通过 ctx.host\_launch() (或 stf\_ctx.host\_task()，取决于上下文对象) 实现的。

**语法 (文档 Page 19, 55):**

// ctx.host\_launch(logicalData1.accessMode(), logicalData2.accessMode()...)  
//     \-\>\*\[capture list\] (auto data1, auto data2...) {  
//     // 主机任务的实现代码  
// };

* 与 GPU 任务类似，您需要声明对逻辑数据的访问模式。  
* lambda 函数的参数是相应逻辑数据在主机上的有效实例。  
* host\_launch 的一个重要特性是，它通过将 lambda 函数作为 CUDA 回调来调用，从而保持了整个工作负载的最佳异步语义。这意味着 lambda 函数的执行会正确地排队，等待其 GPU 数据依赖项准备就绪（包括必要的设备到主机的数据传输，这些由 STF 自动处理）。  
* 这与文档 Page 19 早期提到的 ctx.task(exec\_place::host(), ...) 有所不同。ctx.task(exec\_place::host(), ...) 仍然会立即执行 lambda（该 lambda 接收一个 cudaStream\_t），并且需要用户在该 lambda 内部显式处理与 CUDA 流的同步（例如 cudaStreamSynchronize），这在 graph\_ctx 后端是不允许的。**因此，ctx.host\_launch 是推荐的执行主机任务的方式，因为它与所有后端兼容。**

请打开您本地的 stf/02-axpy-host\_launch.cu。  
分析这个示例：

1. 它可能首先在 GPU 上执行一个 AXPY 操作。  
2. 然后，它使用 host\_launch 提交一个主机任务。  
3. 这个主机任务可能依赖于 GPU AXPY 的结果（例如，读取修改后的 lY）。观察 lY 是如何以 read() 模式传递给 host\_launch 的。  
4. 主机任务 lambda 内部的代码会在 CPU 上执行，并且可以安全地访问 lY 的主机数据实例。

**示例片段 (概念性，请对照 02-axpy-host\_launch.cu):**

// ... (GPU AXPY 任务提交，修改了 lY) ...

// 主机任务，读取 lY 的结果  
stf\_ctx.host\_launch(exec\_place::host(), // 明确指定主机执行 (host\_launch 隐含了主机执行)  
                    lY.read()           // 依赖于 lY 的最新值  
                   )  
    .set\_symbol("verify\_on\_host")  
    \-\>\*\[&\](cuda::experimental::stf::slice\<const double\> sY\_host\_instance) {  
        // 此代码在主机上执行，sY\_host\_instance 是 lY 在主机内存中的有效副本  
        std::cout \<\< "Host Task: Verifying Y\[0\] \= " \<\< sY\_host\_instance(0) \<\< std::endl;  
        // ... 进行验证 ...  
    };

**动手试试:**

1. **编译并运行 01-axpy-cuda\_kernel\_chain.cu 和 02-axpy-host\_launch.cu。**  
2. 在 01-axpy-cuda\_kernel\_chain.cu 中，尝试修改第二个 AXPY 操作，使其不依赖于第一个操作的结果（例如，让它操作一组全新的数据）。然后生成任务图 (CUDASTF\_DOT\_FILE=...)，观察图结构是否反映了这种并行性（即两个任务是否可以独立执行）。  
3. 在 02-axpy-host\_launch.cu 的主机任务 lambda 中添加一些计算或打印语句，以确认它确实在主机上执行，并且可以访问从 GPU 同步过来的数据。  
4. **思考 ctx.task(exec\_place::host(), ...) 和 ctx.host\_launch(...) 的区别。** 为什么 host\_launch 通常是更推荐的方式？(提示：异步性，与 graph 后端的兼容性)。

我们已经深入探讨了 STF 中任务的定义、数据依赖的声明、多种任务提交方式 (task, cuda\_kernel, host\_launch) 以及 STF 如何自动推断任务间的依赖关系。

在 Part 4 中，我们将学习 **"Synchronization" (同步)**、**"Places" (位置)** 的概念，以及更高级的任务构造原语如 **parallel\_for** 和 **launch**。