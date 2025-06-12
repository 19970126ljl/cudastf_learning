#include <cuda/experimental/stf.cuh>
#include <vector>
#include <iostream>
#include <numeric> // For std::iota and other algorithms if needed

int main() {
    using namespace cuda::experimental::stf;
    context ctx;

    const size_t N_array = 128;
    std::vector<int> host_A_vec(N_array);

    auto lA = ctx.logical_data(host_A_vec.data(), host_A_vec.size());
    lA.set_symbol("A_1D_pfor");

    // 在当前设备上执行 parallel_for，迭代 lA 的形状 (0 到 N_array-1)
    // lA 以只写模式访问
    ctx.parallel_for(exec_place::current_device(),
                     lA.shape(),
                     lA.write())
        ->*[] __host__ __device__ (size_t i, slice<int> sA_kernel_arg) -> void {
            // 这是核函数体，会在每个索引 i 上执行
            sA_kernel_arg(i) = 2 * (int)i + 1;
        };

    ctx.finalize();

    // 验证结果 (数据会从设备写回到 host_A_vec)
    bool correct = true;
    for(size_t i = 0; i < N_array; ++i) {
        if (host_A_vec[i] != (2 * (int)i + 1)) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": host_A_vec[" << i << "] = " << host_A_vec[i]
                      << " (Expected: " << (2 * (int)i + 1) << ")" << std::endl;
            break;
        }
    }
    if (correct) {
        std::cout << "parallel_for 1D example: Correct!" << std::endl;
    } else {
        std::cout << "parallel_for 1D example: Incorrect!" << std::endl;
    }

    return 0;
}
