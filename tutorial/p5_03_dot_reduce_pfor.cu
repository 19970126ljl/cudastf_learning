#include <cuda/experimental/stf.cuh>
#include <vector>
#include <numeric> // For std::iota
#include <iostream>
#include <cmath> // For std::abs

int main() {
    using namespace cuda::experimental::stf;
    context ctx;

    const size_t N_dot = 1024;
    std::vector<double> h_x_dot(N_dot), h_y_dot(N_dot);
    std::iota(h_x_dot.begin(), h_x_dot.end(), 1.0);
    std::iota(h_y_dot.begin(), h_y_dot.end(), 1.0);

    auto lX_dot = ctx.logical_data(h_x_dot.data(), N_dot);
    auto lY_dot = ctx.logical_data(h_y_dot.data(), N_dot);
    auto lsum_dot = ctx.logical_data(shape_of<scalar_view<double>>());

    lX_dot.set_symbol("X_for_dot_reduce");
    lY_dot.set_symbol("Y_for_dot_reduce");
    lsum_dot.set_symbol("Sum_result_dot_reduce");

    ctx.parallel_for(exec_place::current_device(),
                     lY_dot.shape(), // 迭代空间
                     lX_dot.read(),
                     lY_dot.read(),
                     lsum_dot.reduce(reducer::sum<double>{}) // 对 lsum_dot 进行求和归约
                    )
        ->*[] __device__ (size_t i, slice<const double> sX, slice<const double> sY, double& sum_ref) {
            // sum_ref 是对 lsum_dot 设备实例中用于累加的部分的引用
            // CUDASTF 会处理线程间的归约细节
            sum_ref += sX(i) * sY(i);
        };

    double dot_product_result = ctx.wait(lsum_dot);
    ctx.finalize();

    std::cout << "Dot product (via parallel_for.reduce and ctx.wait): " << dot_product_result << std::endl;

    double expected_dot_product = 0;
    for(size_t i = 0; i < N_dot; ++i) expected_dot_product += h_x_dot[i] * h_y_dot[i];
    std::cout << "Expected dot product: " << expected_dot_product << std::endl;

    if (std::abs(dot_product_result - expected_dot_product) < 1e-9 * expected_dot_product) {
        std::cout << "Dot product example: Correct!" << std::endl;
    } else {
        std::cout << "Dot product example: Incorrect!" << std::endl;
    }

    return 0;
}
