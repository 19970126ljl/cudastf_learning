## **CUDA STF 教程 \- Part 2: 后端、上下文与逻辑数据**

在 Part 1 中，我们对 CUDA STF 有了初步了解，并通过一个简单的 AXPY 示例看到了 scheduler、logical\_data 和 cuda\_kernel 的基本用法。现在，我们将更详细地探讨 STF 的核心组件：后端与上下文，以及 STF 如何抽象和管理数据。

### **3\. 后端与上下文 (Backends and contexts) (文档 Page 6-7)**

在 CUDASTF 中，**上下文 (context)** 是所有 API 调用的入口点。它存储了 CUDASTF 库的状态，并跟踪所有资源和依赖关系。上下文对象最终必须通过调用其 finalize() 方法来销毁（或者在基于 scheduler 的模型中，通常由 scheduler 的析构函数隐式处理）。

如文档 Page 5 的 AXPY 示例所示：

// 文档中的示例风格  
\#include \<cuda/experimental/stf.cuh\>  
using namespace cuda::experimental::stf; // 文档为简洁通常如此声明

int main() {  
    context ctx; // \<--- 创建一个上下文对象  
    // ... 创建逻辑数据 lX, lY ...  
    // ... 提交任务 ctx.task(...) ...  
    ctx.finalize(); // \<--- 结束时销毁上下文并等待所有操作完成  
    return 0;  
}

在您提供的 01-axpy.cu 示例中，我们使用的是 stf::scheduler sched;，并通过 sched.execute(\[&\](auto& stf\_ctx){ ... }); 来提交任务。这里的 stf\_ctx 就是在 execute 作用域内有效的上下文引用。scheduler 可以看作是管理一个或多个上下文并协调任务执行的更高级别的封装。

CUDASTF 目前提供了三种上下文后端，它们共享一个通用 API，但内部实现可能不同，并且可能有一些特定的扩展：

1. **context (通用上下文类型)**:  
   * 这是一个通用的上下文实现，通常应优先选择它来编写通用代码。  
   * 默认情况下，它使用 stream\_ctx 作为其后端。  
   * 使用通用上下文类型可能会有轻微的运行时开销和增加的编译时间，但灵活性更高。  
2. **stream\_ctx (基于流的上下文)**:  
   * 依赖 CUDA 流 (streams) 和 CUDA 事件 (events) 来实现同步。  
   * 任务是 **即时 (eagerly)** 启动的。  
   * 在此后端中，每个任务 (stream\_task) 都关联一个输入的 CUDA 流。异步工作可以在任务体内提交到这个流中。  
   * 可以使用任务对象的 get\_stream() 方法查询关联的流。  
3. **graph\_ctx (基于图的上下文)**:  
   * 通过 CUDA 图 (graphs) 实现任务并行性。  
   * 任务（及所有相关操作）被放入 CUDA 图中。描述任务的 lambda 函数会立即被捕获（在 ctx.task() 调用期间），即使执行被推迟。  
   * 底层的 CUDA 图在需要与主机同步时，或上下文被 finalize() 时启动。其他情况（如任务栅栏 task\_fence）也可能刷新所有挂起的操作并导致图的启动。后续操作将被放入新的 CUDA 图中。  
   * 选择此后端是采用 CUDA 图的一种简单方法，对于重复的任务模式可能在性能上有所裨益。  
   * **重要限制**: 与其他上下文类型不同，graph\_ctx **不允许** 任务在其内部与 CUDA 流同步（例如，调用 cudaStreamSynchronize）。  
   * 可以使用任务对象的 get\_graph() 方法检索关联的图。

**选择和切换上下文类型 (文档 Page 7):**

\#include \<cuda/experimental/stf.cuh\>  
using namespace cuda::experimental::stf;

int main() {  
    // 1\. 使用通用的 \`context\`，并将其赋值为一个 graph\_ctx 实例  
    context generic\_ctx\_as\_graph \= graph\_ctx();  
    // ... 使用 generic\_ctx\_as\_graph ...  
    generic\_ctx\_as\_graph.finalize();

    // 2\. 静态选择基于 CUDA 流和事件的上下文  
    stream\_ctx stream\_context;  
    // ... 使用 stream\_context ...  
    stream\_context.finalize();

    // 3\. 静态选择基于 CUDA 图的上下文  
    graph\_ctx graph\_context;  
    // ... 使用 graph\_context ...  
    graph\_context.finalize();

    // 大多数情况下，这些上下文类型可以互换使用。  
    // 差异在于内部用于实现同步和执行计算的机制。  
    return 0;  
}

**思考:**

* 对于大多数入门示例（如 01-axpy.cu），默认的上下文行为（可能是 stream\_ctx）已经足够。  
* 当您处理具有许多重复任务模式的复杂应用，或者希望利用 CUDA Graphs 的低启动开销特性时，显式选择 graph\_ctx 可能会带来性能优势。

### **4\. 逻辑数据 (Logical data) (文档 Page 8-12)**

在传统的计算中，“数据”（例如描述神经网络层的矩阵）通常指内存中具有确定地址的位置。然而，在混合 CPU/GPU 系统中，同一个概念上的数据可能同时存在于多个位置并拥有多个地址（通常是 CPU 控制的 RAM，以及一个或多个 GPU 使用的高带宽内存中的副本）。

CUDASTF 将这种概念上的数据称为 **逻辑数据 (logical data)**。它是一个抽象的句柄，代表可能被透明地传输到或复制到 CUDASTF 任务所使用的不同 **位置 (places)** 的数据。

#### **4.1 创建逻辑数据 (文档 Page 8\)**

当用户代码从用户提供的对象（例如，一个 double 数组）创建一个逻辑数据对象时，它们实际上是将原始数据的所有权转移给了 CUDASTF。因此，对原始数据的任何访问都应通过逻辑数据接口进行，因为 CUDASTF 可能会将逻辑数据传输到 CUDA 设备并在那里修改它，从而使原始数据失效。通过这样做，用户代码从所有内存分配的苦差事以及跟踪在计算的不同阶段哪个物理位置持有正确数据的任务中解脱出来。

逻辑数据是通过调用上下文对象的 logical\_data() 成员函数创建的。生成的对象将用于在任务中指定数据访问。

**示例1 (来自文档 Page 8): 从主机数组创建逻辑数据**

\#include \<cuda/experimental/stf.cuh\>  
using namespace cuda::experimental::stf;

int main() {  
    context ctx; // 或者 stream\_ctx ctx; graph\_ctx ctx;  
    const size\_t N \= 16;  
    double host\_array\_X\[N\]; // 一个栈上的主机数组

    // 用主机数组 host\_array\_X 定义一个新的逻辑数据对象 lX  
    // 此后应通过 lX 而不是直接通过 host\_array\_X 来操作这份数据  
    auto lX \= ctx.logical\_data(host\_array\_X);  
    lX.set\_symbol("MyHostDataX"); // 为逻辑数据命名

    // ... 任务可能会在设备上使用 lX ...  
    // 如果一个设备任务修改了 lX，STF 会:  
    // 1\. 在设备内存中创建 lX 的一个新实例。  
    // 2\. 异步地将数据从主机传输到设备。  
    // 3\. 使主机上的原始实例失效。  
    // 4\. 如果后续有主机任务需要从CPU访问 lX，STF 会将数据拷回主机。

    ctx.finalize();  
    return 0;  
}

示例2 (回顾 01-axpy.cu): 从已在设备上的数据创建逻辑数据  
在 01-axpy.cu 中，我们首先手动分配了设备内存 (d\_x\_ptr, d\_y\_ptr) 并将数据从主机拷贝到设备。然后，我们使用这些设备指针（通过 slice 封装）来创建逻辑数据句柄：  
// (在 01-axpy.cu 的 main 函数中)  
// double \*d\_x\_ptr, \*d\_y\_ptr;  
// cudaMalloc(\&d\_x\_ptr, N \* sizeof(double));  
// ... cudaMemcpy ...

// 使用原始指针和大小创建 slice (slice 代表了设备上的一块内存区域)  
slice\<double\> s\_x\_device(d\_x\_ptr, {N});  
slice\<double\> s\_y\_device(d\_y\_ptr, {N});

// 将设备上的 slice 包装成 STF 的 logical\_data  
// cuda::experimental::stf::device\_memory 是一个标记类型，指明数据位于设备内存  
auto ld\_x \= cuda::experimental::stf::make\_data\_handle\<cuda::experimental::stf::device\_memory\>(s\_x\_device);  
auto ld\_y \= cuda::experimental::stf::make\_data\_handle\<cuda::experimental::stf::device\_memory\>(s\_y\_device);

这种方式下，由于数据已经存在于设备上，make\_data\_handle 主要是告诉 STF 这块内存的存在及其描述（通过 slice）。STF 仍然会管理它的一致性，但初始的 HtoD 拷贝是我们手动完成的。

数据实例 (Data Instances) (文档 Page 8):  
每个逻辑数据对象内部维护着多个数据实例 (data instances)，这些实例是逻辑数据在不同数据位置 (data places) 的副本。例如，可能有一个主机内存中的实例，以及 CUDA 设备0 和 CUDA 设备1 的板载内存中的实例。CUDASTF 确保任务在其执行位置能够访问到有效的数据实例，并可能动态地创建新实例或销毁现有实例。

#### **4.2 数据接口 (Data interfaces) (文档 Page 8-9)**

CUDASTF 实现了一个通用接口来操作机器上不同类型的数据格式。  
每种数据格式类型由三个独立的类型描述：

1. **形状 (shape)**: 存储所有实例共有的参数。例如，对于固定大小的向量，形状将包含向量的长度。  
2. **每实例类型 (per-instance type)**: 描述一个特定的数据实例。例如，对于固定大小的向量，此类型将包含向量的地址。  
3. **数据接口类 (data interface class)**: 实现诸如基于其形状分配数据实例，或将一个实例复制到另一个实例等操作。

CUDASTF API 设计为可扩展的，高级用户可以定义自己的数据接口。这对于操作非规则多维数组的数据格式，或为领域特定/应用特定的数据格式提供直接访问非常有用。  
(高级主题 "Defining custom data interfaces" 及其示例链接在文档 Page 9，我们暂时跳过细节。)

#### **4.3 写回策略 (Write-back policy) (文档 Page 9\)**

当一个逻辑数据对象被销毁时，其原始数据实例会被更新（除非该逻辑数据是在没有引用值的情况下创建的，例如从形状创建）。只有在上下文对象上调用了 finalize() 方法后，结果才保证在相应的数据位置上可用。同样，当调用 finalize() 时，如果上下文中关联的所有逻辑数据尚未被销毁，会自动对它们执行写回机制。

* **默认启用写回。**  
* 可以通过在逻辑数据对象上调用 set\_write\_back(bool flag) 来为特定的逻辑数据禁用写回。  
* 对从形状定义且没有引用数据实例的逻辑数据启用写回将导致错误。

// ...  
auto lX \= ctx.logical\_data(host\_array\_X);  
// lX.set\_write\_back(false); // 如果我们不希望在 lX 销毁或 ctx.finalize() 时  
                           // 将设备上可能修改过的数据写回到 host\_array\_X  
// ...

#### **4.4 切片 (Slices) (文档 Page 9-11)**

为了方便使用潜在的非连续多维数组，CUDASTF 引入了一个名为 slice 的 C++ 数据结构类。slice 是 C++ std::mdspan（或 std::experimental::mdspan，取决于 C++ 版本）的部分特化。

template \<typename T, size\_t dimensions \= 1\>  
using slice \= cuda::std::mdspan\<T, cuda::std::dextents\<size\_t, dimensions\>, cuda::std::layout\_stride\>;

(注意：文档中可能直接写作 mdspan，但 STF 内部实现和示例中通常是 cuda::std::mdspan 或其别名 stf::slice)

当从 C++ 数组创建 logical\_data 时，CUDASTF 会自动将其描述为一个 slice，其实例化了元素类型和数组的维度。

**示例 (1D slice):**

double A\[128\];  
context ctx;  
auto lA \= ctx.logical\_data(A); // 内部 lA 的实例被描述为 slice\<double, 1\>  
                               // slice\<double\> 等同于 slice\<double, 1\>  
\`\`\`mdspan\` (因此 \`slice\` 也一样) 提供了多种方法：  
\* \`T\* data\_handle()\`: 返回第一个元素的地址。  
\* \`operator()\`: 例如 \`A(i)\` 是1D slice 的第 i 个元素，\`A(i,j)\` 是2D slice 在坐标 (i,j) 的元素。  
\* \`size\_t size()\`: 返回 slice 中的总元素数量。  
\* \`size\_t extent(size\_t dim)\`: 返回给定维度的 slice 大小。  
\* \`size\_t stride(size\_t dim)\`: 返回给定维度上两个连续元素之间的内存距离（以元素数量表示）。

Slices 可以按值传递、复制或移动。\*\*复制一个 slice 不会复制底层数据\*\*，它只复制描述符。Slices 可以作为参数传递给 CUDA 核函数。

\*\*在核函数中使用 \`slice\` (回顾 \`01-axpy.cu\` 中的 \`axpy\_stf\_kernel\`):\*\*  
\`\`\`cpp  
// template \<typename T\> // 已在 Part 1 中展示  
// \_\_global\_\_ void axpy\_stf\_kernel(T alpha, cuda::experimental::stf::slice\<const T\> x, cuda::experimental::stf::slice\<T\> y) {  
//     int tid \= blockIdx.x \* blockDim.x \+ threadIdx.x;  
//     int nthreads \= gridDim.x \* blockDim.x;  
//     for (int ind \= tid; ind \< x.size(); ind \+= nthreads) {  
//         y(ind) \+= alpha \* x(ind); // 直接使用 slice 的 operator() 和 size()  
//     }  
// }

定义多维切片 (Defining slices with multiple dimensions) (文档 Page 10-11):  
可以使用 make\_slice 方法来定义多维（可能非连续）的 slice。它需要一个基指针、一个包含所有维度的元组以及各个维度的步长。  
**示例 (2D slice):**

// (假设在主机代码中)  
\#include \<cuda/std/mdspan\> // 为了 make\_slice 和 dextents  
// ...  
double A\_host\_2D\[5 \* 2\]; // 一个 10 个元素的数组

// 创建一个连续的 2D slice (5行2列)，行主序（C风格）  
// 步长参数：对于N维数组，需要N-1个步长。  
// 对于2D，第二个维度（列）的步长是1（隐式），第一个维度（行）的步长是列数。  
// make\_slice(pointer, {dim0, dim1, ...}, stride1, stride2, ...)  
// stride\_k is the stride for dimension k.  
// For layout\_stride, strides are provided explicitly.  
// For a 5x2 matrix (row-major), to get from A(i,j) to A(i+1,j), you move '2' elements (stride for dim 0).  
auto s\_2d\_contiguous \= cuda::experimental::stf::make\_slice(A\_host\_2D, std::tuple{5, 2}, std::array\<std::size\_t, 2\>{2, 1});  
// 文档 Page 11 的例子: slice\<double, 2\> s \= make\_slice(A, std::tuple{5,2}, 5);  
// 这里的 '5' 似乎是指第一维的步长，假设第二维的步长是1 (即元素是紧密排列的)。  
// 如果是列主序，或者有不同的布局，步长会不同。  
// 一般来说，对于C风格的二维数组 arr\[H\]\[W\], slice(ptr, {H,W}, {W,1})

// 创建一个非连续的 2D slice (例如，取一个更大矩阵的子块，或者有padding)  
// 假设 A\_host\_2D 是一个 5x2 的数据块，但我们想把它看作一个 4x2 的 slice，  
// 并且行之间的步长仍然是按照原始的 '5' (如果原始数据是 5xSomething\_else)  
// slice\<double, 2\> s2 \= make\_slice(A\_host\_2D, std::tuple{4, 2}, 5); // 文档示例  
// 这意味着 s2(r, c) 访问的是 A\_host\_2D \+ r\*5 \+ c  
// 如果 A\_host\_2D 本身就是 5x2，那么 stride 应该是 2 (列数)  
auto s\_2d\_non\_contiguous\_example \= cuda::experimental::stf::make\_slice(A\_host\_2D, std::tuple{4, 2}, std::array\<std::size\_t, 2\>{2 /\*stride for rows\*/, 1 /\*stride for columns\*/});

// 创建逻辑数据从多维 slice  
// auto lX\_2D\_contiguous \= ctx.logical\_data(s\_2d\_contiguous);  
// auto lX\_2D\_non\_contiguous \= ctx.logical\_data(s\_2d\_non\_contiguous\_example);

请打开 stf/parallel\_for\_2D.cu 或 stf/linear\_algebra/cg\_dense\_2D.cu (如果存在)。  
在这些示例中，您可能会找到创建和使用二维 slice 及相应的 logical\_data 的代码。观察它们是如何使用 make\_slice (如果直接创建slice对象) 或者 ctx.logical\_data 如何处理多维数组的。  
例如，在 parallel\_for\_2D.cu 中，您可能会看到类似这样的逻辑数据创建：

// (概念性代码，请对照您的 parallel\_for\_2D.cu)  
// const size\_t N \= ...;  
// double host\_matrix\_X\[N \* N\];  
// double host\_matrix\_Y\[N \* N\];  
// ... 初始化 ...  
// auto lX \= ctx.logical\_data(make\_slice(\&host\_matrix\_X\[0\], std::tuple{N, N}, N));  
// auto lY \= ctx.logical\_data(make\_slice(\&host\_matrix\_Y\[0\], std::tuple{N, N}, N));

#### **4.5 从形状定义逻辑数据 (Defining logical data from a shape) (文档 Page 11-12)**

有时，我们希望定义一个逻辑数据，但其初始内容将由一个任务来填充，而不是从现有的内存中拷贝。在这种情况下，我们不需要为逻辑数据关联一个初始的引用实例，因为 CUDASTF 会在其首次使用时自动分配一个实例。

**创建方法:**

\#include \<cuda/experimental/stf.cuh\>  
\#include \<cuda\_runtime.h\> // For cudaMemsetAsync  
using namespace cuda::experimental::stf;

int main() {  
    context ctx;

    // 定义一个包含10个整数的向量的逻辑数据，仅通过形状定义  
    // shape\_of\<slice\<int\>\>(10) 创建了一个描述符，表示“一个大小为10的一维int切片”  
    auto lX\_from\_shape \= ctx.logical\_data(shape\_of\<slice\<int\>\>(10));  
    lX\_from\_shape.set\_symbol("X\_from\_shape");

    // 由于 lX\_from\_shape 没有有效的初始数据实例，  
    // 第一个访问它的任务必须是只写访问 (write()) 或不带累加的归约 (reduce())  
    ctx.task(exec\_place::current\_device(), lX\_from\_shape.write())  
        \-\>\*\[&\](cudaStream\_t stream, slice\<int\> sX\_instance) {  
            // sX\_instance 是 STF 在设备上为 lX\_from\_shape 分配的内存实例  
            // 我们可以在这里用 cudaMemsetAsync 或一个核函数来初始化它  
            cudaMemsetAsync(sX\_instance.data\_handle(), 0, sX\_instance.size\_bytes(), stream);  
            // 或者一个kernel: my\_init\_kernel\<\<\<...\>\>\>(sX\_instance);  
            std::cout \<\< "Task: Initializing X\_from\_shape on device." \<\< std::endl;  
        };

    // 也可以定义多维的逻辑数据  
    auto lY\_2D\_from\_shape \= ctx.logical\_data(shape\_of\<slice\<double, 2\>\>(16, 24)); // 16x24 的 double 矩阵  
    lY\_2D\_from\_shape.set\_symbol("Y\_2D\_from\_shape");

    ctx.task(exec\_place::current\_device(), lY\_2D\_from\_shape.write())  
        \-\>\*\[&\](cudaStream\_t stream, slice\<double, 2\> sY\_instance) {  
            // 初始化 sY\_instance  
            // my\_2D\_init\_kernel\<\<\<...\>\>\>(sY\_instance);  
            std::cout \<\< "Task: Initializing Y\_2D\_from\_shape on device. Extent(0): "  
                      \<\< sY\_instance.extent(0) \<\< ", Extent(1): " \<\< sY\_instance.extent(1) \<\< std::endl;  
        };

    ctx.finalize();  
    std::cout \<\< "Finalized." \<\< std::endl;  
    return 0;  
}

**关键点:**

* 使用 shape\_of\<slice\<ElementType, Dimensions\>\>(dim1\_size, dim2\_size, ...) 来创建形状描述符。  
* 对此类逻辑数据的首次访问 **必须是 write()** （或不带初始值的 reduce()）。使用 read() 或 rw() 会导致错误，因为没有有效的初始数据可读。  
* STF 会在任务执行前，在适当的位置为该逻辑数据分配内存。

**动手试试:**

1. **编译并运行上述示例代码片段** (您可以将其整合到一个完整的 .cu 文件中，包含必要的 includes 和 CUDA 运行时调用)。观察输出。  
2. 尝试错误的访问模式:  
   修改上述代码，将 lX\_from\_shape.write() 改为 lX\_from\_shape.read() 或 lX\_from\_shape.rw()。编译并运行。STF 应该会抛出一个运行时错误，指出尝试从未初始化的逻辑数据中读取。  
3. 查看 stf/03-temporary-data.cu:  
   这个示例很可能演示了如何使用从形状创建的逻辑数据作为计算中的临时缓冲区。分析它是如何创建、写入、读取，以及 STF 如何管理其生命周期的。  
4. 查看 stf/09-dot-reduce.cu:  
   文档 Page 28 的点积示例中，lsum 就是通过 ctx.logical\_data(shape\_of\<scalar\_view\<double\>\>()) 创建的。它用于累积归约结果。

我们已经详细学习了 STF 的后端、上下文以及逻辑数据的各种创建和使用方式。您现在应该对 STF 如何抽象数据、管理数据副本和处理数据生命周期有了更清晰的理解。

接下来，我们将进入教程的 Part 3，重点讨论 **"Tasks"** 的创建、数据依赖声明以及不同类型的任务。准备好了吗？