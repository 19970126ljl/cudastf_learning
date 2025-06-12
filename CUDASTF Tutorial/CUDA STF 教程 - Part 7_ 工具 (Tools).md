## **CUDA STF 教程 \- Part 7: 工具 (Tools)**

到目前为止，我们已经学习了 CUDA STF 的核心概念、如何定义数据和任务、各种并行构造以及模块化使用技巧。为了更好地理解、调试和优化基于 STF 的应用程序，CUDASTF 提供了一些工具和与现有工具集成的机制。

本部分主要依据您提供的文档中 "Tools" 章节 (Page 41-52) 的内容。

### **14\. 可视化任务图 (Visualizing task graphs) (文档 Page 41-50)**

理解 STF 如何将您的顺序任务描述转换为并行执行图是至关重要的。CUDASTF 可以生成 [Graphviz](https://graphviz.org/) .dot 文件格式的任务图，然后您可以将其转换为图像（如 PNG、PDF）。

#### **14.1 生成基础的任务图可视化**

1. **设置环境变量**: 在运行您的 STF 应用程序之前，设置 CUDASTF\_DOT\_FILE 环境变量，指定输出的 .dot 文件名。  
   \# 假设您的编译后可执行文件位于 build/bin/cudax.cpp17.example.stf.01-axpy  
   CUDASTF\_DOT\_FILE=axpy.dot build/bin/cudax.cpp17.example.stf.01-axpy

2. **从 .dot 文件生成图像**: 使用 Graphviz 的 dot 命令。  
   \# 生成 PDF 格式  
   dot \-Tpdf axpy.dot \-o axpy.pdf

   \# 生成 PNG 格式  
   dot \-Tpng axpy.dot \-o axpy.png

默认情况下，生成的图可能只显示高级别的任务节点，并标记为 undefined(read) 或 undefined(rw)，除非您为逻辑数据和任务添加了符号名称。

#### **14.2 使用符号名称增强可视化 (文档 Page 41-42)**

为了使任务图更具信息性，您可以为逻辑数据和任务设置符号名称 (symbols)。

* **为逻辑数据命名**:  
  // auto lX \= ctx.logical\_data(X);  
  // lX.set\_symbol("MyVectorX");  
  // \--- 或者链式调用 \---  
  auto lY \= ctx.logical\_data(Y).set\_symbol("MyVectorY");

* **为任务命名**:  
  // \--- 内联方式 \---  
  // ctx.task(lX.read(), lY.rw())  
  //    .set\_symbol("AXPY\_Operation")  
  //    \-\>\*\[&\](cudaStream\_t s, auto dX, auto dY) { /\* ... \*/ };

  // \---显式操作任务对象 \---  
  // auto my\_task \= ctx.task(lX.read(), lY.rw());  
  // my\_task.set\_symbol("AXPY\_Operation");  
  // my\_task-\>\*\[&\](cudaStream\_t s, auto dX, auto dY) { /\* ... \*/ };

  当您为 parallel\_for, launch, cuda\_kernel 等构造设置符号时，这个符号名称也会用于后续的 NVTX 标记，这对于性能分析很有用。

**请打开您本地的 stf/axpy-annotated.cu (如果存在)。** 这个示例应该展示了如何添加这些符号名称。

**动手试试:**

1. 选择一个您之前学习过的简单示例，例如 stf/01-axpy.cu 或 stf/01-axpy-cuda\_kernel.cu。  
2. 按照上述方法，为其逻辑数据和任务添加 set\_symbol() 调用。  
3. 重新编译并使用 CUDASTF\_DOT\_FILE 环境变量运行它，然后生成图像。比较生成的图与未添加符号名称时的图。您应该能看到您设置的名称出现在节点上。

#### **14.3 高级可视化选项 (文档 Page 44-50)**

CUDASTF 提供了多个环境变量来控制生成的任务图的细节级别和外观：

* CUDASTF\_DOT\_IGNORE\_PREREQS=0: (文档 Page 44\)  
  默认情况下，STF 可能会隐藏一些内部生成的异步操作（如内存分配、拷贝）。将此变量设置为 0 可以显示这些更底层的操作，从而提供一个更完整的依赖图。生成的图会更复杂，但有助于理解 STF 的内部工作机制。  
  CUDASTF\_DOT\_IGNORE\_PREREQS=0 CUDASTF\_DOT\_FILE=axpy\_detailed.dot build/bin/your\_stf\_executable  
  dot \-Tpng axpy\_detailed.dot \-o axpy\_detailed.png

* CUDASTF\_DOT\_COLOR\_BY\_DEVICE: (文档 Page 45\)  
  如果设置为非空值，任务节点将根据其执行设备进行着色。这对于理解多 GPU 应用中的任务分布非常有用。  
* CUDASTF\_DOT\_REMOVE\_DATA\_DEPS: (文档 Page 45\)  
  如果设置为非空值，将从任务节点标签中移除数据依赖列表，以简化图形显示，专注于任务流。  
* CUDASTF\_DOT\_TIMING: (文档 Page 45\)  
  如果设置为非空值，将在图中包含计时信息。节点会根据其相对持续时间着色，并且测量的持续时间会包含在任务标签中。这对于性能瓶颈分析非常有价值。

#### **14.4 结构化和浓缩图表可视化 (文档 Page 45-50)**

对于包含成千上万个任务的实际工作负载，直接使用 dot 生成的图可能过于庞大且难以阅读。CUDASTF 允许使用 **点段 (dot sections)** 来结构化图表。

* **创建点段**: 通过 ctx.dot\_section("section\_name") 创建一个 dot\_section 对象。该段的范围从对象创建开始，直到对象被销毁（RAII 风格）或显式调用其 end() 方法。点段可以嵌套。  
  // context ctx;  
  // auto lA \= ctx.token().set\_symbol("A");  
  // auto lB \= ctx.token().set\_symbol("B");

  // auto s\_outer \= ctx.dot\_section("Outer\_Calculation"); // 开始外部段  
  // for (int i \= 0; i \< 2; \++i) {  
  //     auto s\_inner \= ctx.dot\_section("Inner\_Loop\_Iter\_" \+ std::to\_string(i)); // 开始内部段 (RAII)  
  //     ctx.task(lA.read(), lB.rw()).set\_symbol("task\_in\_inner")-\>\*\[\]{ /\* ... \*/};  
  // } // s\_inner 在此销毁，内部段结束  
  // s\_outer.end(); // 显式结束外部段  
  // ctx.finalize();

  在生成的图中，这些段会显示为虚线框。  
* CUDASTF\_DOT\_MAX\_DEPTH: (文档 Page 48\)  
  通过设置此环境变量（例如 CUDASTF\_DOT\_MAX\_DEPTH=2），可以控制生成图中显示的嵌套深度。任何嵌套级别超过指定值的段和任务都将被折叠显示，从而简化复杂图表。  
  * CUDASTF\_DOT\_MAX\_DEPTH=0：仅显示最顶层的任务和段。  
  * CUDASTF\_DOT\_MAX\_DEPTH=1：显示到第一层嵌套。  
  * 等等。

**动手试试:**

1. 选择一个更复杂的示例，例如 stf/heat\_mgpu.cu (如果您的环境支持多GPU编译和运行) 或者 stf/01-axpy-cuda\_kernel\_chain.cu。  
2. 尝试使用 CUDASTF\_DOT\_IGNORE\_PREREQS=0 来查看更详细的图。  
3. 如果示例中有循环，尝试在循环内外添加 ctx.dot\_section()。  
4. 使用不同的 CUDASTF\_DOT\_MAX\_DEPTH 值来生成图，观察图是如何被浓缩的。  
5. 如果可以，尝试 CUDASTF\_DOT\_TIMING=1 (可能需要应用程序运行一段时间才能收集有意义的计时)。

### **15\. 使用 ncu 进行核函数性能分析 (Kernel tuning with ncu) (文档 Page 50-52)**

NVIDIA Nsight Compute (ncu) 是一个强大的工具，用于分析 CUDA 核函数的性能。您可以将 ncu 与使用 ctx.parallel\_for 和 ctx.launch (以及其他提交核函数的 STF 构造) 生成的核函数一起使用。

#### **15.1 命名核函数以便 ncu 识别**

默认情况下，由 parallel\_for 或 launch 生成的核函数在 ncu 中的名称可能不够具有描述性 (例如，都显示为 thrust::cuda\_cub::core::\_kernel\_agent)。  
为了解决这个问题，您应该使用任务的 set\_symbol("YourKernelName") 方法。STF 会使用这个符号名称来创建 NVTX (NVIDIA Tools Extension) 范围，ncu 可以利用这些 NVTX 注释来重命名其报告中的核函数。  
// int A\[128\];  
// auto lA \= ctx.logical\_data(A);

// ctx.parallel\_for(lA.shape(), lA.write())  
//    .set\_symbol("updateA\_kernel") // \<--- 为生成的核函数设置符号  
//    \-\>\*\[\] \_\_device\_\_ (size\_t i, auto sA) {  
//        sA(i) \= 2 \* i \+ 1;  
// };

#### **15.2 运行 ncu**

文档以 miniWeather 示例进行了说明。一般的命令流程如下：

1. **编译优化后的代码**: 性能分析应始终在优化构建上进行。  
   \# 假设您的 CMakeLists.txt 和构建系统配置了 Release 或 RelWithDebInfo 构建类型  
   \# make your\_stf\_executable

2. **使用 ncu 运行应用程序**:  
   ncu \--section=ComputeWorkloadAnalysis \--print-nvtx-rename kernel \--nvtx \-o output\_profile build/bin/your\_stf\_executable \[args...\]

   * \--section=ComputeWorkloadAnalysis: 选择要收集的性能指标部分 (可以根据需要选择其他部分)。  
   * \--print-nvtx-rename kernel: (或 \--print-nvtx-rename all) 指示 ncu 根据 NVTX 范围重命名内核（或所有跟踪的区域）。  
   * \--nvtx: 启用 NVTX 跟踪。  
   * \-o output\_profile: 指定输出报告文件的名称 (通常是 .ncu-rep 文件)。  
   * build/bin/your\_stf\_executable \[args...\]: 您的 STF 应用程序及其参数。

**注意**: 根据您的机器配置，您可能需要以 root 用户身份运行 ncu 或进行特定设置以允许访问 NVIDIA GPU 性能计数器 (参见文档 Page 51 ERR\_NVGPUCTRPERM 错误)。

3. **使用 ncu-ui 查看报告**:  
   ncu-ui output\_profile.ncu-rep

   在 ncu-ui 中，您可能需要在选项中设置 "NVTX Rename Mode" 为 "Kernel" 或 "All" 以正确显示重命名的核函数。

**动手试试:**

1. 选择一个使用了 parallel\_for 或 launch 的 STF 示例，例如 stf/01-axpy-parallel\_for.cu 或 stf/09-dot-reduce.cu。  
2. 确保您已经为相关的 parallel\_for 或 launch 构造调用了 .set\_symbol("MyDescriptiveKernelName")。  
3. 编译该示例。  
4. 尝试使用 ncu 命令（如上所述）来分析它。  
5. 如果成功生成了 .ncu-rep 文件，尝试用 ncu-ui 打开它并查看结果。观察核函数名称是否与您设置的符号一致。

### **教程总结**

恭喜您完成了这个 CUDA STF 学习教程！我们从 STF 的基本概念开始，逐步学习了：

* **核心组件**: scheduler, context (包括 stream\_ctx 和 graph\_ctx)。  
* **数据管理**: logical\_data, slice, 从形状创建数据，写回策略。  
* **任务创建**: ctx.task(), ctx.cuda\_kernel(), ctx.cuda\_kernel\_chain(), ctx.host\_launch()，以及数据依赖声明 (read, write, rw, reduce)。  
* **同步机制**: finalize(), submit(), task\_fence(), wait()。  
* **位置管理**: exec\_place 和 data\_place，以及高级的网格和分区。  
* **高级并行构造**: parallel\_for (包括 box 形状和 reduce 模式) 和 launch (包括线程层级)。  
* **类型系统**: STF 如何利用 C++ 类型来增强代码的健壮性。  
* **模块化使用**: 冻结数据和令牌。  
* **工具**: 使用 Graphviz 可视化任务图，以及使用 ncu 进行性能分析。

**后续学习建议：**

1. **实践更多示例**: 遍历您 stf/ 目录下的所有示例，特别是那些涉及更复杂算法（如图算法、线性代数）或多 GPU 的示例。尝试理解它们是如何运用本教程中介绍的 STF 概念的。  
2. **查阅官方文档**: 对于任何不清楚的 API 或概念，[CUDA STF 官方文档](https://nvidia.github.io/cccl/cudax/stf.html) (以及您提供的 PDF) 始终是最终的参考。  
3. **尝试修改示例**: 不要只是运行它们。尝试修改数据大小、算法参数、任务结构，观察性能和行为的变化。  
4. **构建自己的小程序**: 尝试使用 STF 来实现一些您熟悉的小型并行算法，这将是巩固知识的最佳方式。  
5. **关注性能**: 当您熟悉了 STF 的功能后，开始关注性能。使用 ncu 和任务图可视化来识别瓶颈并进行优化。考虑不同的上下文后端 (stream\_ctx vs graph\_ctx) 对性能的影响。

CUDA STF 是一个强大的工具，可以帮助您更轻松地开发复杂的高性能 CUDA 应用程序。希望本教程对您的学习有所帮助！祝您在并行编程的道路上一切顺利！