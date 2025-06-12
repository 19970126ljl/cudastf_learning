## **CUDA STF 教程 \- Part 6: cuda\_kernel、cuda\_kernel\_chain、类型系统与模块化使用**

在 Part 5 中，我们学习了强大的 parallel\_for 和 launch 构造。现在，我们将回顾并深入探讨另外两种直接与 CUDA 核函数交互的构造：cuda\_kernel 和 cuda\_kernel\_chain。之后，我们会讨论 CUDA STF 如何利用 C++ 的类型系统来增强代码的健壮性和清晰度，最后介绍一些模块化使用 STF 的高级技巧，如数据冻结和令牌。

本部分主要依据您提供的文档中 "cuda\_kernel construct" (Page 33-34)、"cuda\_kernel\_chain construct" (Page 34-35)、"C++ Types of logical data and tasks" (Page 35-38) 和 "Modular use of CUDASTF" (Page 38-40) 章节的内容。

### **10\. cuda\_kernel 构造 (文档 Page 33-34)**

我们之前在分析 01-axpy.cu 等示例时已经接触过 ctx.cuda\_kernel (或 stf\_ctx.cuda\_kernel)。这个构造提供了一种直接的方式来将单个预定义的 CUDA 核函数作为 STF 任务执行。

目的与优势：  
cuda\_kernel 构造对于执行已有的 CUDA 核函数特别有用。当使用 CUDA Graph 后端 (graph\_ctx) 时，ctx.task() 依赖于图捕获机制，这可能会带来一些开销。而 cuda\_kernel 构造直接转换成 CUDA 核函数启动 API，从而避免了这种开销，可能更高效。  
**语法回顾：**

// ctx.cuda\_kernel(\[execution\_place\], logicalData1.accessMode(), ...)  
//     \-\>\*\[capture\_list\] () { // Lambda 不接受流或数据实例作为参数  
//         // Lambda 的任务是返回一个 cuda\_kernel\_desc 对象  
//         // auto dX \= task\_object.template get\<slice\_type\>(0); // 如果需要显式获取数据实例  
//         // auto dY \= task\_object.template get\<slice\_type\>(1);  
//         return cuda\_kernel\_desc{  
//             kernel\_function\_ptr,  
//             gridDim,  
//             blockDim,  
//             sharedMemBytes,  
//             kernel\_arg1, // 可以是标量值  
//             logical\_data\_handle\_for\_arg2, // STF 会处理为设备上的 slice  
//             logical\_data\_handle\_for\_arg3  
//             // ... 其他核函数参数  
//         };  
// };

* cuda\_kernel 接受与 ctx.task 类似的参数，包括可选的执行位置和一系列数据依赖。  
* 其 \-\>\* 操作符接受的 lambda 函数**不接收** CUDA 流或数据实例作为参数。  
* 这个 lambda 函数的职责是**返回一个 cuda\_kernel\_desc 对象**。  
* cuda\_kernel\_desc 的构造函数参数包括：  
  1. Fun func: 指向 \_\_global\_\_ CUDA 核函数的指针。  
  2. dim3 gridDim\_: 网格维度。  
  3. dim3 blockDim\_: 块维度。  
  4. size\_t sharedMem\_: 动态分配的共享内存大小。  
  5. Args... args: 传递给 CUDA 核函数的参数。这些参数可以是普通值，也可以是逻辑数据句柄（STF 会自动将它们解析为设备上的数据实例，通常是 slice）。

**示例 (来自文档 Page 34，稍作调整和解释):**

\#include \<cuda/experimental/stf.cuh\>  
\#include \<vector\>  
\#include \<cmath\> // For std::sin, std::cos  
\#include \<iostream\>

// 假设的 AXPY 核函数，接收 slice 参数  
template \<typename T\>  
\_\_global\_\_ void axpy\_kernel\_for\_desc(T alpha, cuda::experimental::stf::slice\<const T\> x, cuda::experimental::stf::slice\<T\> y) {  
    int idx \= blockIdx.x \* blockDim.x \+ threadIdx.x;  
    if (idx \< x.size()) { // 假设 x 和 y 大小相同  
        y(ind) \+= alpha \* x(ind); // 注意：文档示例中 y(ind) 应为 y(idx)  
    }  
}

int main() {  
    using namespace cuda::experimental::stf;  
    context ctx;

    const size\_t N \= 256;  
    double alpha\_val \= 3.14;

    std::vector\<double\> h\_x(N), h\_y(N);  
    for(size\_t i=0; i\<N; \++i) { h\_x\[i\] \= (double)i; h\_y\[i\] \= (double)i\*2; }

    auto lX \= ctx.logical\_data(h\_x);  
    auto lY \= ctx.logical\_data(h\_y);  
    lX.set\_symbol("X\_cudakernel");  
    lY.set\_symbol("Y\_cudakernel");

    dim3 grid(1);  
    dim3 block(N \< 256 ? N : 256);

    // 创建一个任务，启动 axpy\_kernel\_for\_desc  
    ctx.cuda\_kernel(exec\_place::current\_device(), lX.read(), lY.rw())  
        .set\_symbol("axpy\_via\_cudakernel")  
        \-\>\*\[&\]() { // 这个 lambda 返回 cuda\_kernel\_desc  
            // 在这个 lambda 内部，如果需要显式获取数据实例的指针或slice (虽然通常不需要，  
            // 因为可以直接将 logical\_data 句柄传给 cuda\_kernel\_desc)，  
            // 需要通过任务对象本身来获取，例如：  
            // auto task\_obj \= ctx.cuda\_kernel(...); /\* 先不链接 lambda \*/  
            // task\_obj.add\_deps(...);  
            // task\_obj-\>\*\[&\](){  
            //    auto sX\_instance \= task\_obj.template get\<slice\<const double\>\>(0); // 0 是依赖索引  
            //    auto sY\_instance \= task\_obj.template get\<slice\<double\>\>(1);  
            //    return cuda\_kernel\_desc{..., sX\_instance, sY\_instance};  
            // };  
            // 但更简洁的方式是直接在 cuda\_kernel\_desc 中使用逻辑数据句柄：  
            return cuda\_kernel\_desc{  
                axpy\_kernel\_for\_desc\<double\>, // 核函数指针  
                grid,                         // 网格维度  
                block,                        // 块维度  
                0,                            // 共享内存大小  
                alpha\_val,                    // 核函数参数1 (标量)  
                lX,                           // 核函数参数2 (逻辑数据 \-\> slice\<const double\>)  
                lY                            // 核函数参数3 (逻辑数据 \-\> slice\<double\>)  
            };  
    };

    ctx.finalize();  
    // 验证 h\_y  
    std::cout \<\< "cuda\_kernel example: Y\[0\] after axpy \= " \<\< h\_y\[0\] \<\< std::endl;  
    // Expected: (0\*2) \+ 3.14 \* 0 \= 0  
    // Expected: Y\[1\] \= (1\*2) \+ 3.14 \* 1 \= 5.14  
    if (std::abs(h\_y\[1\] \- (1.0\*2.0 \+ alpha\_val \* 1.0)) \< 1e-9) {  
        std::cout \<\< "cuda\_kernel example: Correct\!" \<\< std::endl;  
    } else {  
        std::cout \<\< "cuda\_kernel example: Incorrect\! Y\[1\] is " \<\< h\_y\[1\] \<\< " but expected " \<\< (1.0\*2.0 \+ alpha\_val \* 1.0) \<\< std::endl;  
    }

    return 0;  
}

动态添加依赖 (文档 Page 34):  
也可以先创建一个 cuda\_kernel 对象，然后动态地添加依赖，并使用 get\<Type\>(index) 在 lambda 内部获取数据实例。  
**请打开您本地的 stf/01-axpy-cuda\_kernel.cu。** 这个示例应该与上述结构非常相似。

### **11\. cuda\_kernel\_chain 构造 (文档 Page 34-35)**

除了 cuda\_kernel，CUDASTF 还提供了 cuda\_kernel\_chain 构造，用于在一个 STF 任务中**顺序执行一系列 CUDA 核函数**。

语法：  
与 cuda\_kernel 类似，但其 lambda 函数应返回一个 std::vector\<cuda\_kernel\_desc\>。向量中的每个 cuda\_kernel\_desc 对象描述一个核函数启动，它们将按照在向量中出现的顺序依次执行。  
// ctx.cuda\_kernel\_chain(\[execution\_place\], logicalData1.accessMode(), ...)  
//     \-\>\*\[capture\_list\] () {  
//         return std::vector\<cuda\_kernel\_desc\>{  
//             {kernel1\_ptr, grid1, block1, shmem1, args1...}, // 第一个核函数  
//             {kernel2\_ptr, grid2, block2, shmem2, args2...}, // 第二个核函数  
//             // ...  
//         };  
// };

示例 (来自文档 Page 35，概念性):  
假设我们想顺序执行三次 AXPY 操作：Y=Y+αX, Y=Y+βX, Y=Y+γX。  
// (在 main 函数中)  
// ... (lX, lY, alpha, beta, gamma, grid, block 已定义) ...

// ctx.cuda\_kernel\_chain(exec\_place::current\_device(), lX.read(), lY.rw())  
//     .set\_symbol("axpy\_chain\_task")  
//     \-\>\*\[&\]() {  
//         return std::vector\<cuda\_kernel\_desc\>{  
//             {axpy\_kernel\_for\_desc\<double\>, grid, block, 0, alpha, lX, lY},  
//             {axpy\_kernel\_for\_desc\<double\>, grid, block, 0, beta,  lX, lY},  
//             {axpy\_kernel\_for\_desc\<double\>, grid, block, 0, gamma, lX, lY}  
//         };  
// };

// ctx.finalize();

这比使用 ctx.task() 并在其 lambda 中多次调用 kernel\<\<\<...\>\>\>() 可能更高效，尤其是在使用 CUDA Graph 后端时，因为它避免了多次图捕获的开销。

**请打开您本地的 stf/01-axpy-cuda\_kernel\_chain.cu。** 这个示例将具体展示 cuda\_kernel\_chain 的用法。

### **12\. C++ 类型与逻辑数据和任务 (文档 Page 35-38)**

为了防止常见错误，CUDASTF 努力使其处理语义与 C++ 类型尽可能紧密地对齐。如各种示例所示，通常建议使用 auto 关键字来创建可读的代码，同时仍然强制执行类型安全。

#### **12.1 逻辑数据的类型**

调用 ctx.logical\_data() 的结果是一个对象，其类型包含了用于操作该逻辑数据对象的底层数据接口的信息。  
例如，一个连续的 double 数组在内部表示为 slice\<double\> (它是 std::mdspan 的别名)。  
// double host\_X\_arr\[16\];  
// context ctx;

// 显式类型  
// logical\_data\<slice\<double\>\> lX\_typed \= ctx.logical\_data(host\_X\_arr);

// 使用 auto 更简洁  
// auto lX\_auto \= ctx.logical\_data(host\_X\_arr); // lX\_auto 的类型仍是 logical\_data\<slice\<double\>\>

在类或结构体中存储逻辑数据时，可能需要使用 mutable 限定符，因为即使对包含该逻辑数据的 const 对象进行只读访问的任务，也可能会修改逻辑数据对象的内部状态（例如，更新版本号、缓存状态等），尽管从用户角度看这应该是一个 const 操作。

#### **12.2 任务的类型**

使用 stream\_ctx 后端时，ctx.task(lX.read(), lY.rw()) 返回一个类型为 stream\_task\<TX, TY\> 的对象，其中 TX 和 TY 是与逻辑数据 lX 和 lY 的数据接口相关联的类型。

* 如果 lX 和 lY 是 double 数组 (内部为 slice\<double\>)：  
  * lX.read() 对应 slice\<const double\>  
  * lY.rw() 对应 slice\<double\>  
  * 任务类型将是 stream\_task\<slice\<const double\>, slice\<double\>\>。  
* 这种类型信息会从任务对象传播到通过 operator-\>\* 调用的 lambda，从而在编译时检测类型错误。  
* 使用 graph\_ctx 后端时，对应的类型是 graph\_task\<...\>。  
* 使用通用 context 类型时，对应的类型是 unified\_task\<...\>。

**类型安全示例 (文档 Page 37):**

// double X\[16\], Y\[16\];  
// auto lX \= ctx.logical\_data(X);  
// auto lY \= ctx.logical\_data(Y);

// 如果 lambda 参数类型不匹配，会导致编译错误  
// ctx.task(lX.read(), lY.rw())-\>\*\[\](cudaStream\_t s, slice\<int\> x, slice\<int\> y) { /\* ... \*/ }; // 错误！期望 slice\<const double\>, slice\<double\>

// 使用 auto 可以避免这种手动类型声明的麻烦  
// ctx.task(lX.read(), lY.rw())-\>\*\[\](cudaStream\_t s, auto x, auto y) { /\* ... \*/ }; // 正确

#### **12.3 动态类型任务 (Dynamically-typed tasks) (文档 Page 37-38)**

在某些情况下，任务访问的确切数据（以及因此任务的类型）可能无法静态确定（例如，访问计算域中某个部分的最近邻居，而邻居是动态确定的）。

对于这种情况，CUDASTF 提供了动态类型的任务（例如，在 stream\_ctx 后端中称为 stream\_task\<\>，注意没有模板参数），其 add\_deps() 成员函数允许动态添加依赖。

// double X\[16\], Y\[16\];  
// auto lX \= ctx.logical\_data(X);  
// auto lY \= ctx.logical\_data(Y);

// stream\_ctx s\_ctx; // 假设使用 stream\_ctx  
// stream\_task\<\> dyn\_task \= s\_ctx.task(); // 创建一个无类型的任务  
// dyn\_task.add\_deps(lX.read(), lY.rw()); // 动态添加依赖

// 这种动态方法导致了表达能力的损失。基于 \-\>\* 的 API 仅与静态类型任务兼容。  
// 因此，动态类型任务需要使用较低级的 API 进行操作，例如使用 get\<Type\>(index) 获取数据实例。  
// dyn\_task-\>\*\[&\]() { /\* ... lambda ... \*/ }; // 这通常不直接与完全动态的任务一起使用  
// 通常是：  
// dyn\_task.set\_body(\[&\](cudaStream\_t s){  
//    auto sX\_instance \= dyn\_task.template get\<slice\<const double\>\>(0);  
//    auto sY\_instance \= dyn\_task.template get\<slice\<double\>\>(1);  
//    // ... 使用 sX\_instance 和 sY\_instance ...  
// });

也可以将依赖动态添加到静态类型的任务中，但这不会改变任务的原始静态类型。访问动态添加的依赖需要使用 task\_object.template get\<Type\>(index)，并且会在运行时进行检查。

### **13\. 模块化使用 CUDASTF (文档 Page 38-40)**

CUDASTF 维护整个系统的数据一致性，并根据数据访问推断并发机会。然而，在某些用例中，用户可能已经自己管理了一致性或强制执行了依赖关系。STF 提供了机制来适应这些情况。

#### **13.1 冻结逻辑数据 (Freezing logical data) (文档 Page 38-39)**

当一块数据被非常频繁地使用时（例如，一次写入后多次读取），为了避免每次访问都强制执行数据依赖关系（这会带来开销），可以“冻结”逻辑数据。

* auto frozen\_ld \= ctx.freeze(logical\_data\_handle, \[access\_mode, data\_place\]);  
  * 默认情况下，返回一个冻结的逻辑数据对象，可以在**只读模式**下无额外同步地访问。  
  * frozen\_ld.get(data\_place, stream): 返回底层数据在指定数据位置上的视图（例如 slice）。此视图可以在传递给 get 的流上异步使用，直到调用 unfreeze()。  
  * 可以多次调用 get() 获取不同位置或不同流上的视图。  
  * 修改只读冻结视图会导致未定义行为。  
  * 如有必要，调用 get() 时会异步执行隐式数据传输或分配。  
* frozen\_ld.unfreeze(stream);: 解冻逻辑数据。必须确保传递给 unfreeze 的流依赖于先前传递给 freeze 和所有 get 调用的流中的工作完成。  
* 也可以创建**可修改的冻结逻辑数据** (access\_mode::rw)，这允许应用程序临时将逻辑数据的所有权转移给不使用任务的代码。在数据冻结为可修改期间，任何访问此逻辑数据的 STF 任务都将被推迟，直到调用 unfreeze。  
* 不能并发冻结同一个逻辑数据。

**示例 (概念性，来自文档 Page 39):**

// context ctx;  
// auto ld \= ctx.logical\_data(...);  
// cudaStream\_t stream1, stream2; /\* ... 创建流 ... \*/

// // 冻结 ld 为只读  
// auto frozen\_ld\_readonly \= ctx.freeze(ld);  
//  
// // 获取设备上的只读视图  
// auto dX\_view \= frozen\_ld\_readonly.get(data\_place::current\_device(), stream1);  
// kernel\_read\<\<\<..., stream1\>\>\>(dX\_view);  
//  
// // 获取主机上的只读视图  
// auto hX\_view \= frozen\_ld\_readonly.get(data\_place::host(), stream2);  
// // ... 在 stream2 同步后，在主机上使用 hX\_view ...  
//  
// // 必须确保 stream1 和 stream2 上的工作完成后再 unfreeze  
// // 例如: cudaStreamSynchronize(stream1); cudaStreamSynchronize(stream2);  
// cudaStream\_t unfreeze\_stream; /\* ... \*/  
// // 假设 unfreeze\_stream 依赖于 stream1 和 stream2  
// frozen\_ld\_readonly.unfreeze(unfreeze\_stream);

// // 冻结 ld 为可修改 (在当前设备)  
// auto frozen\_ld\_rw \= ctx.freeze(ld, access\_mode::rw, data\_place::current\_device());  
// auto dX\_modifiable\_view \= frozen\_ld\_rw.get(data\_place::current\_device(), stream1);  
// kernel\_modify\<\<\<..., stream1\>\>\>(dX\_modifiable\_view); // 这个核函数可以修改 dX\_modifiable\_view  
// // ... 确保 stream1 完成 ...  
// frozen\_ld\_rw.unfreeze(unfreeze\_stream); // 假设 unfreeze\_stream 依赖 stream1

**请打开您本地的 stf/frozen\_data\_init.cu。** 这个示例应该演示了如何使用冻结数据机制。

#### **13.2 令牌 (Tokens) (文档 Page 40\)**

令牌是一种特殊类型的逻辑数据，其唯一目的是**自动化同步**，而让应用程序管理实际的数据。当用户有自己的缓冲区（例如在单个设备上，不需要分配或传输），但可能发生并发访问时，令牌非常有用。

* auto token \= ctx.token();  
* 令牌内部依赖于 void\_interface 数据接口，该接口经过优化，可以跳过缓存一致性协议中不必要的阶段（如数据分配或复制）。  
* 使用令牌而不是具有完整数据接口的逻辑数据可以最大限度地减少运行时开销。

**示例 (来自文档 Page 40):**

// context ctx;  
// auto lA \= ctx.logical\_data(...);  
// auto lB \= ctx.logical\_data(...);

// auto sync\_token \= ctx.token();  
// sync\_token.set\_symbol("MySyncToken");

// Task 1: 修改 lB，并“接触”token (rw)  
// ctx.task(sync\_token.rw(), lA.read(), lB.rw())  
//     \-\>\*\[&\](cudaStream\_t s, /\* void\_interface dummy, 可选 \*/ auto sA, auto sB) {  
//         // ... kernel\_using\_A\_modifying\_B\<\<\<...\>\>\>(sA, sB);  
//         // lambda 中可以省略对应 token 的 void\_interface 参数  
//     };

// Task 2: 依赖于 Task 1 对 token 的“接触”  
// ctx.task(sync\_token.read(), lB.read()) /\* 或者 sync\_token.rw() 如果 Task 2 也想“拥有”它一会儿 \*/  
//     \-\>\*\[&\](cudaStream\_t s, /\* void\_interface dummy, 可选 \*/ auto sB\_after\_task1) {  
//         // ... kernel\_reading\_B\_after\_task1\<\<\<...\>\>\>(sB\_after\_task1);  
//     };

由于令牌仅用于同步目的，其在任务 lambda 中的对应参数（类型为 void\_interface）可以被省略。  
ctx.token() 创建的令牌已经是“有效的”，意味着第一次访问可以是 read() 或 rw()，不需要像从形状创建的逻辑数据那样先进行 write()。  
**请打开您本地的 stf/void\_data\_interface.cu (如果它演示了令牌的用法) 或其他可能使用令牌进行细粒度同步的复杂示例。**

**动手试试:**

1. **编译并运行本部分提供的 cuda\_kernel 示例代码。**  
2. **研究 stf/01-axpy-cuda\_kernel\_chain.cu**: 理解它是如何将多个核函数调用捆绑到单个 STF 任务中的。与为每个核函数创建一个单独的 ctx.cuda\_kernel 任务相比，这样做有什么潜在的好处？  
3. **思考类型安全**: 在您看过的 STF 示例中，auto 关键字是如何帮助简化代码同时保持类型安全的？尝试在一个示例中，将 auto 替换为显式的 logical\_data\<...\> 或任务参数的 slice\<...\> 类型，以加深理解。  
4. **研究 stf/frozen\_data\_init.cu**:  
   * 数据是如何被冻结和解冻的？  
   * get() 方法在其中扮演什么角色？  
   * 冻结数据对于性能或编程模型有什么影响？  
5. **(概念思考)** 在什么情况下，使用 ctx.token() 比使用常规的 logical\_data 进行同步更有优势？

我们已经完成了对 cuda\_kernel、cuda\_kernel\_chain、STF 的类型系统以及模块化使用技巧（如冻结数据和令牌）的学习。这些工具和概念为构建复杂、高效且可维护的 CUDA 应用程序提供了坚实的基础。

在教程的最后一部分 (Part 7)，我们将简要介绍 CUDA STF 提供的**工具**，主要是任务图的可视化和使用 ncu 进行核函数性能分析。