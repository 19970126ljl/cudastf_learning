## **CUDA STF 教程 \- Part 5: 并行构造 (parallel\_for 与 launch)**

在前面的部分中，我们学习了如何创建基本任务、管理数据依赖、同步以及控制执行和数据位置。CUDA STF 还提供了更高级的构造原语，使得直接在逻辑数据上编写并行计算内核变得更加简洁和强大。本部分将重点介绍 parallel\_for 和 launch 这两个构造。

本部分主要依据您提供的文档中 "parallel\_for construct" (Page 24-29) 和 "launch construct" (Page 30-33) 章节的内容。

### **8\. parallel\_for 构造 (文档 Page 24-29)**

parallel\_for 是 CUDASTF 提供的一个辅助构造，用于创建在某个索引空间（通常由逻辑数据的形状定义）上执行操作的 CUDA 核函数（或 CPU 核函数，取决于执行位置）。它简化了常见的数据并行模式的实现。

#### **8.1 parallel\_for 的基本结构**

parallel\_for 构造主要包含四个元素：

1. **执行位置 (Execution Place)**: 指定代码将在哪里执行（例如，exec\_place::device(0) 或 exec\_place::host()）。  
2. **形状 (Shape)**: 定义了生成核函数将迭代的索引空间。这通常是某个逻辑数据的形状 (logical\_data\_handle.shape())，也可以是自定义的 box 形状。  
3. **数据依赖集 (Data Dependencies)**: 与普通任务一样，声明所访问的逻辑数据及其访问模式（read(), write(), rw(), reduce()）。  
4. **代码体 (Body of Code)**: 使用 \-\>\* 操作符指定的一个 lambda 函数。这个 lambda 函数就是将在每个索引上执行的核函数体。

**lambda 函数的参数:**

* 对于一个 N 维的形状，lambda 的前 N 个参数是 size\_t 类型的索引（例如，对于二维形状是 size\_t i, size\_t j）。  
* 后续参数是与 parallel\_for 中声明的逻辑数据依赖相对应的数据实例（通常是 slice 对象）。  
* 如果执行位置是设备，则 lambda 函数需要有 \_\_device\_\_ (或 \_device\_) 修饰符。

#### **8.2 parallel\_for 处理一维数组示例 (文档 Page 24\)**

\#include \<cuda/experimental/stf.cuh\>  
\#include \<vector\>  
\#include \<iostream\>

int main() {  
    using namespace cuda::experimental::stf;  
    context ctx;

    const size\_t N\_array \= 128;  
    std::vector\<int\> host\_A\_vec(N\_array);

    auto lA \= ctx.logical\_data(host\_A\_vec);  
    lA.set\_symbol("A\_1D\_pfor");

    // 在设备1上执行 parallel\_for，迭代 lA 的形状 (0 到 N\_array-1)  
    // lA 以只写模式访问  
    ctx.parallel\_for(exec\_place::current\_device(), // 或者 exec\_place::device(0) 如果只有一个GPU  
                     lA.shape(),  
                     lA.write())  
        \-\>\*\[&\](size\_t i, slice\<int\> sA\_kernel\_arg) \_\_device\_\_ {  
            // 这是核函数体，会在每个索引 i 上执行  
            sA\_kernel\_arg(i) \= 2 \* i \+ 1;  
    };

    ctx.finalize();

    // 验证结果 (数据会从设备写回到 host\_A\_vec)  
    // for(size\_t i \= 0; i \< 5; \++i) { // 打印前几个元素  
    //     std::cout \<\< "host\_A\_vec\[" \<\< i \<\< "\] \= " \<\< host\_A\_vec\[i\]  
    //               \<\< " (Expected: " \<\< (2 \* i \+ 1\) \<\< ")" \<\< std::endl;  
    // }  
    if (host\_A\_vec\[N\_array-1\] \== (2\*(N\_array-1)+1)) {  
        std::cout \<\< "parallel\_for 1D example: Correct\!" \<\< std::endl;  
    } else {  
        std::cout \<\< "parallel\_for 1D example: Incorrect\!" \<\< std::endl;  
    }

    return 0;  
}

**请打开您本地的 stf/01-axpy-parallel\_for.cu。** 这个示例应该展示了如何使用 parallel\_for 来实现 AXPY 操作。将其与我们之前讨论的基于 cuda\_kernel 的 AXPY 实现进行比较。

#### **8.3 parallel\_for 处理多维数组示例 (文档 Page 25\)**

对于多维数据形状，parallel\_for 的 lambda 函数会接收对应数量的索引参数。

\#include \<cuda/experimental/stf.cuh\>  
\#include \<vector\>  
\#include \<iostream\>  
\#include \<tuple\> // For std::tuple

int main() {  
    using namespace cuda::experimental::stf;  
    context ctx;

    const size\_t M \= 4; // 例如 4 行  
    const size\_t K \= 3; // 例如 3 列  
    std::vector\<double\> host\_X\_matrix\_vec(M \* K);

    // 创建一个二维 slice 来描述这个矩阵  
    // make\_slice(pointer, {num\_rows, num\_cols}, row\_stride)  
    // 对于行主序的 M x K 矩阵，行步长是 K  
    auto sX\_host \= make\_slice(host\_X\_matrix\_vec.data(), std::tuple{M, K}, K);  
    auto lX\_matrix \= ctx.logical\_data(sX\_host);  
    lX\_matrix.set\_symbol("X\_2D\_pfor");

    ctx.parallel\_for(exec\_place::current\_device(),  
                     lX\_matrix.shape(), // 迭代 lX\_matrix 的二维形状  
                     lX\_matrix.write())  
        \-\>\*\[&\](size\_t i, size\_t j, slice\<double, 2\> sX\_kernel\_arg) \_\_device\_\_ {  
            // i 是行索引 (0 到 M-1), j 是列索引 (0 到 K-1)  
            sX\_kernel\_arg(i, j) \= (double)(i \* 10 \+ j);  
    };

    ctx.finalize();

    // 验证  
    // for(size\_t i \= 0; i \< M; \++i) {  
    //     for(size\_t j \= 0; j \< K; \++j) {  
    //         std::cout \<\< "host\_X\_matrix\_vec\[" \<\< i\*K+j \<\< "\] (" \<\< i \<\< "," \<\< j \<\< ") \= "  
    //                   \<\< host\_X\_matrix\_vec\[i\*K+j\] \<\< " (Expected: " \<\< (i\*10+j) \<\< ")" \<\< std::endl;  
    //     }  
    // }  
    if (host\_X\_matrix\_vec\[M\*K-1\] \== (double)((M-1)\*10 \+ (K-1))) {  
         std::cout \<\< "parallel\_for 2D example: Correct\!" \<\< std::endl;  
    } else {  
        std::cout \<\< "parallel\_for 2D example: Incorrect\!" \<\< std::endl;  
    }

    return 0;  
}

**请打开您本地的 stf/parallel\_for\_2D.cu。** 这个示例会更完整地展示二维 parallel\_for 的应用。

#### **8.4 box 形状 (文档 Page 26-27)**

有时，迭代的索引空间不直接对应于某个逻辑数据的形状。对于这些情况，CUDASTF 提供了 box\<size\_t dimensions \= 1\> 模板类，允许用户定义具有显式边界的多维形状。

* **基于范围的 box**: box\<2\>({dim0\_extent, dim1\_extent}) 表示一个二维迭代空间，第一个索引从 0 到 dim0\_extent-1，第二个索引从 0 到 dim1\_extent-1。  
  // ctx.parallel\_for(exec\_place::current\_device(), box\<2\>({2, 3})) // 迭代 i=0..1, j=0..2  
  //     \-\>\*\[\](size\_t i, size\_t j) \_\_device\_\_ {  
  //         printf("Box extent: %ld, %ld\\n", i, j);  
  // };

* **基于上下界的 box**: box\<2\>({{lower0, upper0}, {lower1, upper1}})。下界包含，上界不包含。  
  // ctx.parallel\_for(exec\_place::current\_device(), box\<2\>({{5, 8}, {2, 4}})) // i=5..7, j=2..3  
  //     \-\>\*\[\](size\_t i, size\_t j) \_\_device\_\_ {  
  //         printf("Box bounds: %ld, %ld\\n", i, j);  
  // };

#### **8.5 reduce() 访问模式 (文档 Page 27-29)**

parallel\_for 支持 reduce() 访问模式，这使得在 parallel\_for 生成的计算核函数内部实现归约操作成为可能。

reduce() 接受的参数：

1. **归约操作符 (Reduction Operator)**: 定义了如何组合多个值以及如何初始化一个值（例如，求和归约会将两个值相加，并将初始值设为0）。这些操作符定义在 cuda::experimental::stf::reducer 命名空间中。  
2. **可选的 no\_init{} 标签**: 如果提供此标签，则归约结果将累加到逻辑数据中已存在的值上（类似于 rw() 访问模式）。默认情况下（不提供 no\_init{}），逻辑数据的内容将被归约结果覆盖（类似于 write() 访问模式）。对没有有效实例的逻辑数据（例如，仅从形状定义的）使用 no\_init{} 会导致错误。  
3. **其他参数**: 与其他访问模式相同，例如数据位置。

只能对数据接口定义了 owning\_container\_of trait 类的逻辑数据应用 reduce() 访问模式。scalar\_view\<T\> 数据接口就是这种情况，它将 owning\_container\_of 设置为 T。传递给 parallel\_for lambda 的归约参数是对该类型对象的引用。

**点积示例 (来自文档 Page 28):**

\#include \<cuda/experimental/stf.cuh\>  
\#include \<vector\>  
\#include \<numeric\> // For std::iota  
\#include \<iostream\>

int main() {  
    using namespace cuda::experimental::stf;  
    context ctx;

    const size\_t N\_dot \= 1024;  
    std::vector\<double\> h\_x\_dot(N\_dot), h\_y\_dot(N\_dot);  
    std::iota(h\_x\_dot.begin(), h\_x\_dot.end(), 1.0);  
    std::iota(h\_y\_dot.begin(), h\_y\_dot.end(), 1.0);

    auto lX\_dot \= ctx.logical\_data(h\_x\_dot);  
    auto lY\_dot \= ctx.logical\_data(h\_y\_dot);  
    // 创建一个用于存储标量归约结果的逻辑数据 (从形状定义)  
    auto lsum\_dot \= ctx.logical\_data(shape\_of\<scalar\_view\<double\>\>());

    lX\_dot.set\_symbol("X\_for\_dot\_reduce");  
    lY\_dot.set\_symbol("Y\_for\_dot\_reduce");  
    lsum\_dot.set\_symbol("Sum\_result\_dot\_reduce");

    // 计算 sum(X\_i \* Y\_i)  
    ctx.parallel\_for(exec\_place::current\_device(),  
                     lY\_dot.shape(), // 迭代空间  
                     lX\_dot.read(),  
                     lY\_dot.read(),  
                     lsum\_dot.reduce(reducer::sum\<double\>{}) // 对 lsum\_dot 进行求和归约  
                    )  
        \-\>\*\[&\](size\_t i, slice\<const double\> sX, slice\<const double\> sY, double& sum\_ref) \_\_device\_\_ {  
            // sum\_ref 是对 lsum\_dot 设备实例中用于累加的部分的引用  
            // CUDASTF 会处理线程间的归约细节  
            sum\_ref \+= sX(i) \* sY(i);  
    };

    // 使用 ctx.wait() 获取归约结果 (阻塞调用)  
    double dot\_product\_result \= ctx.wait(lsum\_dot);  
    ctx.finalize(); // 确保所有操作完成

    std::cout \<\< "Dot product (via parallel\_for.reduce and ctx.wait): " \<\< dot\_product\_result \<\< std::endl;

    double expected\_dot\_product \= 0;  
    for(size\_t i \= 0; i \< N\_dot; \++i) expected\_dot\_product \+= h\_x\_dot\[i\] \* h\_y\_dot\[i\];  
    std::cout \<\< "Expected dot product: " \<\< expected\_dot\_product \<\< std::endl;

    if (std::abs(dot\_product\_result \- expected\_dot\_product) \< 1e-9 \* expected\_dot\_product) {  
        std::cout \<\< "Dot product example: Correct\!" \<\< std::endl;  
    } else {  
        std::cout \<\< "Dot product example: Incorrect\!" \<\< std::endl;  
    }

    return 0;  
}

预定义的归约操作符 (文档 Page 28-29):  
sum, product, maxval, minval, logical\_and, logical\_or, bitwise\_and, bitwise\_or, bitwise\_xor。  
用户也可以定义自己的归约操作符 (文档 Page 29)。  
**请打开您本地的 stf/09-dot-reduce.cu 和 stf/word\_count\_reduce.cu (如果存在)。** 这些示例会展示 reduce() 的实际应用。

### **9\. launch 构造 (文档 Page 30-33)**

ctx.launch 原语是 CUDASTF 中一种核函数启动机制，它隐式地处理单个核函数到执行位置的映射和启动。与 parallel\_for（在每个索引点应用相同操作）不同，launch 执行的是一个基于**线程层级 (thread hierarchy)** 的结构化计算核函数。

#### **9.1 launch 的基本语法**

// ctx.launch(\[thread\_hierarchy\_spec\], // 可选的线程层级规范  
//            \[execution\_place\],       // 执行位置  
//            logicalData1.accessMode(),  
//            logicalData2.accessMode(), ...)  
//     \-\>\*\[capture\_list\] \_\_device\_\_ (thread\_hierarchy\_spec\_t th, // 线程层级对象  
//                                   auto data1, auto data2...) {  
//     // 核函数实现  
// };  
\`\`\`launch\` 构造包含五个主要元素：  
1\.  \*\*可选的执行策略/线程层级规范 (Execution Policy / Thread Hierarchy Specification)\*\*: 显式指定启动形状。例如，指定一组独立线程或可同步线程。  
2\.  \*\*执行位置 (Execution Place)\*\*: 指示代码将在哪里执行。  
3\.  \*\*数据依赖集 (Data Dependencies)\*\*: 与其他任务构造类似。  
4\.  \*\*代码体 (Body of Code)\*\*: 使用 \`-\>\*\` 指定的 lambda 函数，带有 \`\_\_device\_\_\` 修饰符。  
5\.  \*\*线程层级参数 (\`thread\_info\` 或 \`thread\_hierarchy\_spec\_t th\`)\*\*: lambda 的第一个参数，用于查询线程属性（如全局ID、线程总数）和层级结构。

\*\*示例 (来自文档 Page 30，稍作调整):\*\*  
\`\`\`cpp  
\#include \<cuda/experimental/stf.cuh\>  
\#include \<cuda/experimental/stf/thread\_hierarchy.cuh\> // For par, con etc.  
\#include \<vector\>  
\#include \<iostream\>  
\#include \<cmath\> // For std::sin, std::cos

// 假设的 AXPY 逻辑，使用 launch  
// N 和 alpha 需要从捕获列表或作为逻辑数据传入  
// 这里我们假设它们是全局可访问或捕获的，仅为演示 launch 结构

const size\_t LAUNCH\_N \= 1024;  
const double LAUNCH\_ALPHA \= 2.5;

int main() {  
    using namespace cuda::experimental::stf;  
    context ctx;

    std::vector\<double\> h\_x\_launch(LAUNCH\_N), h\_y\_launch(LAUNCH\_N);  
    for(size\_t i=0; i\<LAUNCH\_N; \++i) {  
        h\_x\_launch\[i\] \= std::sin((double)i);  
        h\_y\_launch\[i\] \= std::cos((double)i);  
    }

    auto lX\_launch \= ctx.logical\_data(h\_x\_launch);  
    auto lY\_launch \= ctx.logical\_data(h\_y\_launch);  
    lX\_launch.set\_symbol("X\_for\_launch");  
    lY\_launch.set\_symbol("Y\_for\_launch");

    // par(1024) 表示一个包含1024个独立（不可同步）线程的并行组  
    // all\_devs 是一个执行位置，表示在所有可用设备上启动 (需要更复杂的设置，这里用 current\_device)  
    // cdp 是 data\_place 的简写，通常是 data\_place::current\_device() 或 data\_place::affine()  
    ctx.launch(par(1024), // 线程层级：1024个并行线程  
               exec\_place::current\_device(),  
               lX\_launch.read(data\_place::affine()), // 或者 .read()  
               lY\_launch.rw(data\_place::affine())    // 或者 .rw()  
              )  
        \-\>\*\[=\] \_\_device\_\_ (thread\_info t, slice\<const double\> x\_slice, slice\<double\> y\_slice) {  
            // t 是线程信息对象  
            size\_t tid \= t.thread\_id(); // 获取当前线程的全局唯一ID  
            size\_t nthreads \= t.get\_num\_threads(); // 获取此 launch 启动的总线程数

            for (size\_t ind \= tid; ind \< LAUNCH\_N /\*x\_slice.size()\*/; ind \+= nthreads) {  
                y\_slice(ind) \+= LAUNCH\_ALPHA \* x\_slice(ind);  
            }  
    };

    ctx.finalize();

    // 验证  
    // ... (省略验证代码，与之前类似) ...  
    std::cout \<\< "launch example completed." \<\< std::endl;

    return 0;  
}

#### **9.2 描述线程层级 (文档 Page 31\)**

线程层级规范描述了核函数的并行结构。层级大小可以自动计算、动态指定或编译时指定。

* **par(num\_threads)**: 并行组 (parallel group)，线程独立执行，不可组内同步。  
  * par\<128\>(): 静态指定大小。  
* **con(num\_threads)**: 并发组 (concurrent group)，组内线程可以使用 sync() API 进行同步（组级别屏障）。  
* **层级嵌套**: 可以嵌套多个组，例如 par(128, con\<256\>()) 表示128个独立的组，每个组包含256个可同步的线程。  
* **共享内存**: con(256, mem(64)) 表示256个线程的组，共享64字节的内存（由STF自动分配在适当的内存层级）。  
* **硬件范围亲和性 (Hardware Scope Affinity)**: 可以指定线程组映射到特定的机器层级。  
  * hw\_scope::thread: CUDA 线程。  
  * hw\_scope::block: CUDA 块。  
  * hw\_scope::device: CUDA 设备。  
  * hw\_scope::all: 整台机器。  
  * 示例: par(hw\_scope::device | hw\_scope::block, par\<128\>(hw\_scope::thread))

#### **9.3 操作线程层级对象 (th) (文档 Page 32-33)**

传递给 launch 核函数体的线程层级对象（通常命名为 th 或 t）提供了查询层级结构和线程交互的方法：

* th.rank(): 线程在整个层级中的全局排名。  
* th.size(): 整个层级的总线程数。  
* th.rank(level\_idx): 线程在第 level\_idx 层的排名。  
* th.size(level\_idx): 第 level\_idx 层的线程数。  
* th.is\_synchronizable(level\_idx): 检查第 level\_idx 层是否可同步。  
* th.sync(level\_idx): 同步第 level\_idx 层的所有线程。  
* th.sync(): 同步最顶层（0级）的所有线程。  
* th.get\_scope(level\_idx): 获取第 level\_idx 层的硬件范围亲和性。  
* th.template storage\<T\>(level\_idx): 获取与第 level\_idx 层关联的本地存储（作为 slice\<T\>）。  
* th.depth(): (constexpr) 层级的深度。  
* th.inner(): 获取移除了最顶层后的线程层级子集。

**请打开您本地的 stf/launch\_sum.cu、stf/launch\_scan.cu 或 stf/launch\_histogram.cu。** 这些示例会展示 launch 构造以及线程层级操作的实际应用，通常用于实现比简单数据并行更复杂的并行模式（例如，自定义的归约、扫描等）。

**动手试试:**

1. **编译并运行上面提供的 parallel\_for (1D 和 2D) 以及 launch 的示例代码。** 确保您理解它们的行为和输出。  
2. **研究 stf/01-axpy-parallel\_for.cu**: 将其与 stf/01-axpy-cuda\_kernel.cu 进行比较。parallel\_for 如何简化了 AXPY 的实现？  
3. **研究 stf/09-dot-reduce.cu**:  
   * 它是如何使用 parallel\_for 和 lsum.reduce(reducer::sum\<double\>{}) 来计算点积的？  
   * sum\_ref 在 lambda 中是如何工作的？  
   * ctx.wait(lsum\_dot) 如何用于获取最终结果？  
4. **(挑战)** 尝试修改 stf/09-dot-reduce.cu，使用不同的归约操作符，例如 reducer::maxval\<double\>{} 来找两个向量对应元素乘积的最大值。  
5. **研究 stf/launch\_sum.cu**:  
   * launch 的线程层级是如何定义的？  
   * 核函数体是如何使用线程层级对象 th 来执行求和的？是否有使用 th.sync() 或共享内存？

我们已经学习了 CUDA STF 中强大的 parallel\_for 和 launch 构造，它们使得表达复杂并行模式更为便捷。

在教程的 Part 6，我们将讨论 cuda\_kernel 和 cuda\_kernel\_chain (虽然之前已提及，但会再次回顾其在整体结构中的位置)，以及 STF 如何通过 C++ 类型系统来增强代码的健壮性，最后是模块化使用 STF 的一些高级技巧。