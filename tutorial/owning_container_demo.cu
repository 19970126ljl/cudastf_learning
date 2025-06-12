#include <cuda/experimental/stf.cuh>
#include <iostream>
#include <vector>

using namespace cuda::experimental::stf;

int main() {
    context ctx;
    
    // ✅ 这个可以用于归约 - scalar_view<double> 实现了 owning_container_of
    auto lsum_scalar = ctx.logical_data(shape_of<scalar_view<double>>());
    
    // ❌ 这个不能用于归约 - slice<double> 没有实现 owning_container_of
    // auto lsum_slice = ctx.logical_data(shape_of<slice<double>>(1));
    
    const size_t N = 100;
    std::vector<double> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = i + 1.0;  // 1, 2, 3, ..., 100
    }
    
    auto ldata = ctx.logical_data(data.data(), data.size());
    
    // ✅ 使用 scalar_view 进行归约 - 计算数组元素之和
    ctx.parallel_for(ldata.shape(),
                     ldata.read(),
                     lsum_scalar.reduce(reducer::sum<double>{}))
        ->*[] __device__(size_t i, auto sdata, double& sum) {
            sum += sdata(i);
        };
    
    // ❌ 如果尝试对 slice 进行归约，编译会失败
    // ctx.parallel_for(ldata.shape(),
    //                  ldata.read(),
    //                  lsum_slice.reduce(reducer::sum<double>{}))  // 编译错误！
    //     ->*[] __device__(size_t i, auto sdata, auto& sum_slice) {
    //         sum_slice(0) += sdata(i);
    //     };
    
    double result = ctx.wait(lsum_scalar);
    ctx.finalize();
    
    std::cout << "Sum of 1 to " << N << " = " << result << std::endl;
    std::cout << "Expected: " << (N * (N + 1)) / 2 << std::endl;
    
    return 0;
}
