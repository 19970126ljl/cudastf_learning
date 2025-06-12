## **CUDA STF 教程 \- Part 4: 同步 (Synchronization) 与位置 (Places)**

在前面的部分中，我们学习了如何定义逻辑数据和提交任务。现在，我们将探讨如何在 STF 中管理这些异步操作的完成，以及如何控制任务的执行位置和数据的存放位置。

本部分主要依据您提供的文档中 "Synchronization" (Page 17-18) 和 "Places" (Page 18-24) 章节的内容。

### **6\. 同步 (Synchronization) (文档 Page 17-18)**

如前所述，提交给 STF 上下文的任务体（lambda 函数）通常是立即在主机上执行的，其作用是将异步工作（如 CUDA 核函数）提交到 CUDA 流中。CUDASTF 确保在任务体内的操作可以一致地访问指定的数据，并遵循请求的访问模式。

由于任务并行执行的异步性，必须确保所有操作都正确调度和执行。因为 CUDASTF 透明地处理数据管理（分配、传输等），可能存在一些用户未显式提交的悬而未决的异步操作。因此，**仅使用原生的 CUDA 同步操作（如 cudaStreamSynchronize() 或 cudaDeviceSynchronize()）是不够的**，因为它们不知道 CUDASTF 的内部状态。

#### **6.1 ctx.submit() 和 ctx.finalize()**

* **ctx.submit()**:  
  * 启动序列中所有异步任务的提交过程。  
  * 通常，创建任务和调用 ctx.finalize() 就足够了。  
  * 手动调用 ctx.submit() 在以下情况可能有用：  
    1. 允许在提交和同步之间在 CPU（或其他 GPU）上执行额外的无关工作。  
    2. 当需要两个上下文并发运行时，使用 ctx1.submit(); ctx2.submit(); ctx1.finalize(); ctx2.finalize(); 可以实现此目标（而直接调用 ctx1.finalize(); ctx2.finalize(); 会等待第一个任务完成才开始第二个）。  
* **ctx.finalize()**:  
  * 等待上下文中所有未完成的异步操作（包括任务、传输等）结束。  
  * 如果用户代码未在此之前调用 ctx.submit()，finalize() 会自动调用它。  
  * 这是确保所有 STF 管理的工作都已完成的主要机制。

**示例 (来自文档 Page 17):**

\#include \<cuda/experimental/stf.cuh\>  
// ... 其他必要的 includes ...

int main() {  
    using namespace cuda::experimental::stf;  
    context ctx;

    // ... 定义逻辑数据 lX, lY ...  
    // ... 提交任务 ctx.task(lX.read(), lY.rw())-\>\*\[\]{ ... }; ...

    // 可选：提交所有挂起的任务，但不立即等待它们完成  
    ctx.submit();

    // 此时可以在 CPU 上执行一些与 STF 任务无关的工作  
    // ... Unrelated CPU-based code might go here...

    // 等待 STF 上下文中所有操作完成  
    ctx.finalize();

    return 0;  
}

在您提供的 stf/ 示例中，通常 main 函数结束时，scheduler 或 context 对象的析构函数会隐式地处理 finalize 的逻辑，或者 scheduler.execute(...) 调用本身就是阻塞的，直到该批次任务完成。对于需要显式控制的复杂场景，submit() 和 finalize() 提供了更细致的控制。

#### **6.2 任务栅栏 (ctx.task\_fence())**

这是一种等待所有挂起操作（任务、传输等）完成的异步栅栏机制。

cudaStream\_t stream\_for\_fence \= ctx.task\_fence();  
// 现在，任何提交到 stream\_for\_fence 的后续 CUDA 工作  
// 都会在其之前的 STF 任务完成后才开始。  
// 如果需要主机等待，则可以同步这个特定的流：  
cudaStreamSynchronize(stream\_for\_fence);  
\`\`\`ctx.task\_fence()\` 返回一个 CUDA 流。STF 保证所有在调用 \`task\_fence()\` 之前提交到上下文的任务和数据操作，都会在这个返回的流上的任何后续操作开始之前完成。这允许您将 STF 工作流与其他基于 CUDA 流的异步操作同步。

\#\#\#\# 6.3 \`ctx.wait(logical\_data\_handle)\`

这是一种\*\*阻塞调用\*\*，用于等待特定的逻辑数据准备就绪，并返回其内容。返回值的类型由逻辑数据接口的 \`owning\_container\_of\<interface\>\` trait 类定义。

\* 此方法通常与 \`reduce()\` 访问模式结合使用，以实现动态控制流（例如，根据归约结果决定下一步操作）。  
\* 不能在没有重载此 trait 类的接口的逻辑数据上调用 \`wait()\`。

\*\*示例 (参考文档 Page 28 的点积示例):\*\*  
\`\`\`cpp  
\#include \<cuda/experimental/stf.cuh\>  
\#include \<vector\>  
\#include \<numeric\> // for std::iota  
\#include \<iostream\>

// 假设的核函数，用于并行计算并累加到 sum  
template\<typename T\>  
\_\_global\_\_ void accumulate\_product(cuda::experimental::stf::slice\<const T\> x, cuda::experimental::stf::slice\<const T\> y, T\* global\_sum\_output) {  
    extern \_\_shared\_\_ T sdata\[\];  
    int tid \= threadIdx.x;  
    int gid \= blockIdx.x \* blockDim.x \+ threadIdx.x;

    T local\_sum \= 0;  
    if (gid \< x.size()) { // 确保不越界  
        local\_sum \= x(gid) \* y(gid);  
    }  
    sdata\[tid\] \= local\_sum;  
    \_\_syncthreads();

    // Shared memory reduction (simplified for one block)  
    for (unsigned int s \= blockDim.x / 2; s \> 0; s \>\>= 1\) {  
        if (tid \< s) {  
            sdata\[tid\] \+= sdata\[tid \+ s\];  
        }  
        \_\_syncthreads();  
    }

    if (tid \== 0\) {  
        atomicAdd(global\_sum\_output, sdata\[0\]);  
    }  
}

int main() {  
    using namespace cuda::experimental::stf;  
    context ctx;

    const size\_t N \= 1024;  
    std::vector\<double\> h\_x(N), h\_y(N);  
    std::iota(h\_x.begin(), h\_x.end(), 1.0); // 1, 2, ..., N  
    std::iota(h\_y.begin(), h\_y.end(), 1.0); // 1, 2, ..., N

    auto lX \= ctx.logical\_data(h\_x);  
    auto lY \= ctx.logical\_data(h\_y);  
    // 创建一个用于存储标量结果的逻辑数据，从形状定义  
    auto lsum \= ctx.logical\_data(shape\_of\<scalar\_view\<double\>\>());

    lX.set\_symbol("X\_dot");  
    lY.set\_symbol("Y\_dot");  
    lsum.set\_symbol("Sum\_dot");

    // 任务：计算点积的一部分，并将结果归约到 lsum  
    // 注意：真正的点积归约通常用 parallel\_for \+ reduce 更优雅  
    // 这里仅为演示 ctx.wait() 的概念，用一个简化的核函数  
    // 这个核函数只是把每个元素的乘积加到一个全局的 lsum 中  
    // 更好的方式是使用 ctx.parallel\_for 和 lsum.reduce(reducer::sum\<double\>{})  
    // 我们将在 parallel\_for 部分看到更合适的点积实现

    ctx.task(exec\_place::current\_device(),  
             lX.read(),  
             lY.read(),  
             lsum.write() // 初始化 lsum (或者用 reduce)  
            )  
        \-\>\*\[&\](cudaStream\_t s, slice\<const double\> sX, slice\<const double\> sY, scalar\_view\<double\> sSum\_scalar) {  
            // sSum\_scalar.data() 返回 double\*  
            // 这里需要一个设备上的 double\* 来让 kernel 写入  
            // scalar\_view 本身可能代表主机或设备上的单个值  
            // 为了简单起见，我们假设 sSum\_scalar.data() 可以被核函数安全写入  
            // 实际上，对于从 shape\_of\<scalar\_view\> 创建的逻辑数据，  
            // STF 会在设备上为其分配内存，sSum\_scalar.data() 将指向那里。

            // 首先将设备上的 lsum 初始化为0  
            cudaMemsetAsync(sSum\_scalar.data(), 0, sizeof(double), s);

            dim3 threads\_per\_block(256);  
            dim3 num\_blocks((N \+ threads\_per\_block.x \- 1\) / threads\_per\_block.x);  
            size\_t shared\_mem\_size \= threads\_per\_block.x \* sizeof(double);

            accumulate\_product\<\<\<num\_blocks, threads\_per\_block, shared\_mem\_size, s\>\>\>(sX, sY, sSum\_scalar.data());  
    };

    // ctx.finalize(); // 如果在 wait 之前 finalize，结果就已经可用了

    // 阻塞等待 lsum 的计算结果，并获取其值  
    double result \= ctx.wait(lsum); // 主机将等待 lsum 准备就绪

    ctx.finalize(); // 确保所有其他操作也完成

    std::cout \<\< "Dot product result (obtained via ctx.wait): " \<\< result \<\< std::endl;

    double expected\_sum \= 0;  
    for(size\_t i=0; i\<N; \++i) expected\_sum \+= h\_x\[i\] \* h\_y\[i\];  
    std::cout \<\< "Expected dot product: " \<\< expected\_sum \<\< std::endl;

    return 0;  
}

**请打开您本地的 stf/09-dot-reduce.cu。** 这个示例应该更规范地展示了如何使用 parallel\_for 和 reduce() 访问模式来计算点积，并可能使用 ctx.wait() 来获取最终的标量结果。

### **7\. 位置 (Places) (文档 Page 18-24)**

为了帮助用户管理数据和执行的亲和性 (affinity)，CUDASTF 提供了 **位置 (place)** 的概念。位置可以表示：

* **执行位置 (Execution Places)**: 决定代码在哪里执行。  
* **数据位置 (Data Places)**: 指定数据在机器非均匀内存中的位置。

CUDASTF 的目标之一是默认确保数据根据执行位置进行高效放置，同时也为用户提供了在必要时轻松自定义放置的选项。

#### **7.1 执行位置 (Execution Places) (文档 Page 18-19)**

任务的构造函数（或 task(), cuda\_kernel() 等方法的第一个参数）允许选择一个执行位置。

* exec\_place::current\_device(): (默认) 在当前活动的 CUDA 设备上运行。  
* exec\_place::device(int device\_id): 在指定的 device\_id 的 CUDA 设备上运行。  
* exec\_place::host(): 在主机 CPU 上运行。

重要说明：  
无论执行位置如何，任务体（lambda 函数）本身是在主机上执行的 CPU 代码，其目的是异步地启动计算。

* 当使用 exec\_place::device(id) 时，CUDASTF 会在任务开始时自动将当前 CUDA 设备设置为 id，并在任务结束时恢复之前的设备。  
* exec\_place::host() 不会影响当前 CUDA 设备。  
* 对于 exec\_place::host()，如果使用 ctx.task(exec\_place::host(), ...)，lambda 会接收一个 cudaStream\_t。如前所述，用户需要小心处理同步，并且这种方式与 graph\_ctx 不兼容。**推荐使用 ctx.host\_launch(...) 来执行主机任务**，因为它通过 CUDA 回调机制维护了更好的异步语义，并与所有后端兼容。

**示例 (来自文档 Page 19，稍作修改以便运行):**

\#include \<cuda/experimental/stf.cuh\>  
\#include \<cassert\>  
\#include \<iostream\>

// 假设的简单核函数  
\_\_global\_\_ void inc\_kernel(cuda::experimental::stf::slice\<int\> sX\_slice) {  
    if (threadIdx.x \== 0 && blockIdx.x \== 0\) {  
        sX\_slice(0) \+= 1;  
    }  
}

int main() {  
    using namespace cuda::experimental::stf;  
    context ctx;

    int host\_X\_val \= 42;  
    // 创建一个描述单个整数的逻辑数据  
    // 注意：直接传递 \&host\_X\_val 给 slice 构造函数，然后给 logical\_data  
    // STF 会处理到设备的数据传输  
    auto lX \= ctx.logical\_data(slice\<int\>(\&host\_X\_val, {1}));  
    lX.set\_symbol("X\_scalar");

    int num\_devices \= 0;  
    cudaGetDeviceCount(\&num\_devices);

    if (num\_devices \< 1\) {  
        std::cerr \<\< "Requires at least 1 CUDA device\!" \<\< std::endl;  
        return 1;  
    }

    // 任务1: 在设备0上执行  
    ctx.task(exec\_place::device(0), lX.rw())  
        \-\>\*\[&\](cudaStream\_t stream, slice\<int\> sX\_kernel\_arg) {  
            std::cout \<\< "Task 1 (on device 0): incrementing X. Stream: " \<\< stream \<\< std::endl;  
            inc\_kernel\<\<\<1, 1, 0, stream\>\>\>(sX\_kernel\_arg);  
    };

    if (num\_devices \> 1\) {  
        // 任务2: 在设备1上执行 (如果存在)  
        ctx.task(exec\_place::device(1), lX.rw())  
            \-\>\*\[&\](cudaStream\_t stream, slice\<int\> sX\_kernel\_arg) {  
                std::cout \<\< "Task 2 (on device 1): incrementing X. Stream: " \<\< stream \<\< std::endl;  
                inc\_kernel\<\<\<1, 1, 0, stream\>\>\>(sX\_kernel\_arg);  
        };  
    }

    // 任务3: 在主机上执行，读取 lX (使用 host\_launch)  
    // host\_launch 会确保在 lambda 执行前，lX 的值从设备同步回主机  
    ctx.host\_launch(exec\_place::host(), // exec\_place::host() 是可选的，host\_launch 隐含了它  
                    lX.read())  
        \-\>\*\[&\](slice\<const int\> sX\_host\_arg) { // lambda 参数是主机上的数据实例  
            std::cout \<\< "Host Task: reading X." \<\< std::endl;  
            // 在 host\_launch 的 lambda 中，可以直接访问 sX\_host\_arg(0)  
            // 因为 STF 保证了数据此时在主机上是有效的  
            int expected\_val \= 42 \+ (num\_devices \> 1 ? 2 : 1);  
            std::cout \<\< "Host Task: X(0) \= " \<\< sX\_host\_arg(0) \<\< ", Expected \= " \<\< expected\_val \<\< std::endl;  
            assert(sX\_host\_arg(0) \== expected\_val);  
    };

    ctx.finalize();  
    std::cout \<\< "Finalized. Original host\_X\_val after STF (due to write-back): " \<\< host\_X\_val \<\< std::endl;

    return 0;  
}

请打开您本地的 stf/explicit\_data\_places.cu 或 stf/heat\_mgpu.cu / stf/fdtd\_mgpu.cu。  
这些示例（尤其是多 GPU 示例）会清晰地展示如何使用 exec\_place::device(id) 来将计算任务分配到不同的 GPU 上。

#### **7.2 数据位置 (Data Places) (文档 Page 19-21)**

默认情况下，逻辑数据与其当前处理它的任务的**执行位置的仿射数据位置 (affine data place)** 相关联。

* 在设备上启动的任务，其数据默认加载到该设备的全局内存中。  
* 在主机上执行的任务，其数据默认访问主机内存 (RAM)。

您可以在声明数据依赖时，为逻辑数据显式指定一个数据位置：  
logical\_data\_handle.accessMode(data\_place\_specifier)

* data\_place::affine(): (默认) 将数据定位在与执行位置仿射的数据位置。  
* data\_place::managed(): 使用统一内存 (managed memory)。  
* data\_place::device(int device\_id): 将数据放在指定 device\_id 的 CUDA 设备的内存中（这可能与当前设备或任务的执行设备不同）。  
* data\_place::host(): 将数据放在主机内存中。

**示例 (来自文档 Page 20):**

// context ctx;  
// auto lA \= ctx.logical\_data(...);

// 任务在 device 0 执行，数据 lA 也默认在 device 0 的内存中  
// ctx.task(exec\_place::device(0), lA.rw()) \-\>\* ...

// 等同于显式指定仿射数据位置  
// ctx.task(exec\_place::device(0), lA.rw(data\_place::affine())) \-\>\* ...  
// ctx.task(exec\_place::device(0), lA.rw(data\_place::device(0))) \-\>\* ...

// 覆盖亲和性：任务在 device 0 执行，但访问位于主机内存的 lA 实例  
// (假设系统支持从设备访问主机内存，例如通过统一虚拟内存 UVM)  
// ctx.task(exec\_place::device(0), lA.rw(data\_place::host())) \-\>\* ...

// 任务在主机执行，但访问位于 device 0 内存的 lA 实例  
// (假设系统支持从主机访问设备内存)  
// ctx.task(exec\_place::host(), lA.rw(data\_place::device(0))) \-\>\* ...

覆盖数据亲和性在某些情况下可能有利，例如当任务仅稀疏访问大块逻辑数据时，可以避免传输大量数据（CUDA 统一内存的分页系统会自动调入实际使用的部分）。然而，这依赖于系统硬件（如 NVLINK、UVM）和操作系统（例如 WSL 对从 CUDA 核函数访问主机内存的支持可能有限且性能较低）。

**请再次查看 stf/explicit\_data\_places.cu。** 这个示例应该专门演示了如何显式控制数据位置，以及它与执行位置的交互。

#### **7.3 位置网格 (Grid of Places) 与分区策略 (文档 Page 21-24)**

这部分内容较为高级，主要用于多 GPU 或更复杂的分布式内存场景，允许将多个位置组织成网格，并定义数据如何在这些位置之间分区。

* **exec\_place\_grid**: 描述执行位置的网格。可以从一个位置向量创建，或定义为多维网格。  
  * exec\_place::all\_devices(): 一个创建包含所有可用设备网格的辅助函数。  
* **分区策略 (Partitioning policies)**: 定义数据如何在网格中的不同位置分派，或者并行循环的索引空间如何在位置间分布。  
  * tiled\_partition\<TILE\_SIZE\>: 使用瓦片式布局分派。  
  * blocked\_partition: 使用块式布局分派，网格中的每个条目大致接收形状的相同连续部分。

这些高级功能对于编写高度可扩展的多 GPU 应用非常重要。如果您计划进行此类开发，建议详细阅读文档的这部分内容，并研究 stf/heat\_mgpu.cu 或 stf/linear\_algebra/06-pdgemm.cu (分布式 GEMM) 等示例。

**动手试试:**

1. **编译并运行上面修改过的 exec\_place 示例代码。** 如果您有多个 GPU，观察任务是否确实在不同的设备上调度（可以通过 nvidia-smi 或在任务体中打印设备 ID 来粗略判断，但 STF 抽象了流，直接打印设备 ID 可能不总能反映 STF 的调度意图，除非您在任务体中显式设置设备）。  
2. **研究 stf/explicit\_data\_places.cu**: 理解它是如何显式地将数据放置在与执行位置不同的地方，以及这可能带来的影响。  
3. **(可选，如果有多 GPU 环境)** 尝试理解 stf/heat\_mgpu.cu 是如何使用 exec\_place::device(id) 来划分工作的。注意查找是否有显式的拷贝任务（例如 stf\_ctx.copy\_task(...)）用于在不同 GPU 的数据实例之间同步数据（例如 halo 交换）。

我们现在对 STF 中的同步机制和位置管理有了更深入的了解。这些概念对于编写正确且高效的异构并行程序至关重要。

在教程的 Part 5，我们将开始学习 STF 提供的高级并行构造原语：parallel\_for 和 launch，它们使得在逻辑数据上直接编写并行计算变得更加简洁。