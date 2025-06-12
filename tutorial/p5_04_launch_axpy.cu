#include <cuda/experimental/stf.cuh>
#include <vector>
#include <iostream>
#include <cmath> // For std::sin, std::cos, std::abs
#include <numeric> // For std::iota

// LAUNCH_N 和 LAUNCH_ALPHA 可以在 main 中定义并通过捕获列表传递给 lambda 
// 或者作为逻辑数据传递。这里我们为了简化，作为全局常量。
// 但在实际应用中，推荐通过捕获或逻辑数据传递。
// const size_t LAUNCH_N_GLOBAL = 1024; // Example, better to pass via capture
// const double LAUNCH_ALPHA_GLOBAL = 2.5; // Example, better to pass via capture

int main() {
    using namespace cuda::experimental::stf;
    context ctx;

    const size_t LAUNCH_N = 1024;
    const double LAUNCH_ALPHA = 2.5;

    std::vector<double> h_x_launch(LAUNCH_N), h_y_launch(LAUNCH_N), h_y_expected(LAUNCH_N);
    for(size_t i=0; i<LAUNCH_N; ++i) {
        h_x_launch[i] = std::sin((double)i);
        h_y_launch[i] = std::cos((double)i);
        h_y_expected[i] = std::cos((double)i) + LAUNCH_ALPHA * std::sin((double)i); // Precompute expected Y
    }

    auto lX_launch = ctx.logical_data(h_x_launch.data(), LAUNCH_N);
    auto lY_launch = ctx.logical_data(h_y_launch.data(), LAUNCH_N);
    lX_launch.set_symbol("X_for_launch");
    lY_launch.set_symbol("Y_for_launch");

    ctx.launch(par(LAUNCH_N), // 线程层级：LAUNCH_N 个并行线程
               exec_place::current_device(),
               lX_launch.read(), 
               lY_launch.rw()
              )
        ->*[=] __device__ (auto t, slice<const double> x_slice, slice<double> y_slice) {
            size_t tid = t.rank(); // 获取当前线程的全局唯一ID
            // size_t nthreads = t.size(); // 获取此 launch 启动的总线程数
            // In this simple case with par(LAUNCH_N) and vector size LAUNCH_N, tid directly maps to index.
            // For more general cases (e.g., if par_size < vector_size), a grid-stride loop is needed:
            // for (size_t ind = tid; ind < x_slice.size(); ind += nthreads) {
            //    y_slice(ind) += LAUNCH_ALPHA * x_slice(ind);
            // }
            if (tid < x_slice.size()) { // Check bounds, x_slice.size() should be LAUNCH_N here
                 y_slice(tid) += LAUNCH_ALPHA * x_slice(tid);
            }
        };

    ctx.finalize();

    // 验证
    bool correct = true;
    for(size_t i=0; i < LAUNCH_N; ++i) {
        if (std::abs(h_y_launch[i] - h_y_expected[i]) > 1e-9) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": Y_launch[" << i << "] = " << h_y_launch[i] 
                      << ", Expected: " << h_y_expected[i] << std::endl;
            break;
        }
    }
    if (correct) {
        std::cout << "launch AXPY example: Correct!" << std::endl;
    } else {
        std::cout << "launch AXPY example: Incorrect!" << std::endl;
    }
    std::cout << "launch example completed." << std::endl;

    return 0;
}
