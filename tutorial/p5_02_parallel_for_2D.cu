#include <cuda/experimental/stf.cuh>
#include <vector>
#include <iostream>
#include <tuple> // For std::tuple

int main() {
    using namespace cuda::experimental::stf;
    context ctx;

    const size_t M = 4; // 例如 4 行
    const size_t K = 3; // 例如 3 列
    std::vector<double> host_X_matrix_vec(M * K);

    // 创建一个二维 slice 来描述这个矩阵
    // 使用简化的 make_slice 语法创建连续的 2D slice
    auto sX_host = make_slice(host_X_matrix_vec.data(), M, K);
    auto lX_matrix = ctx.logical_data(sX_host);
    lX_matrix.set_symbol("X_2D_pfor");

    ctx.parallel_for(exec_place::current_device(),
                     lX_matrix.shape(), // 迭代 lX_matrix 的二维形状
                     lX_matrix.write())
        ->*[] _CCCL_DEVICE (size_t i, size_t j, auto sX_kernel_arg) {
            // i 是行索引 (0 到 M-1), j 是列索引 (0 到 K-1)
            sX_kernel_arg(i, j) = (double)(i * 10 + j);
        };

    ctx.finalize();

    // 首先打印实际的数组内容来调试
    std::cout << "Array contents after kernel execution:" << std::endl;
    for(size_t idx = 0; idx < M * K; ++idx) {
        std::cout << "host_X_matrix_vec[" << idx << "] = " << host_X_matrix_vec[idx] << std::endl;
    }

    // 验证 - slice 使用列主序存储，所以 sX_kernel_arg(i,j) 对应 host_X_matrix_vec[j*M + i]
    bool correct = true;
    for(size_t i = 0; i < M; ++i) {
        for(size_t j = 0; j < K; ++j) {
            size_t col_major_index = j * M + i;  // 列主序索引
            double expected_value = (double)(i * 10 + j);
            if (std::abs(host_X_matrix_vec[col_major_index] - expected_value) > 1e-9) {
                 correct = false;
                 std::cout << "Mismatch at (" << i << "," << j << "): host_X_matrix_vec[" << col_major_index << "] = "
                           << host_X_matrix_vec[col_major_index] << " (Expected: " << expected_value << ")" << std::endl;
                 break;
            }
        }
        if (!correct) break;
    }
    if (correct) {
         std::cout << "parallel_for 2D example: Correct!" << std::endl;
    } else {
        std::cout << "parallel_for 2D example: Incorrect!" << std::endl;
    }

    return 0;
}
