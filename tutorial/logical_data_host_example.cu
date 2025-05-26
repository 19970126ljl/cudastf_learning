// logical_data_host_example.cu

#include <cuda/experimental/stf.cuh>
#include <cstdio> // For printf

using namespace cuda::experimental::stf;

__global__ void modify_on_device(slice<double> data_slice, double val) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && data_slice.size() > 0) {
        data_slice(0) = val; // Modify the first element
    }
}

int main() {  
    context ctx; 
    const size_t N = 16;  
    double host_array_X[N]; 
    for(size_t i=0; i<N; ++i) host_array_X[i] = (double)i;
    printf("Initial host_array_X[0]: %f\n", host_array_X[0]);

    auto lX = ctx.logical_data(host_array_X);  
    lX.set_symbol("MyHostDataX");

    // 任务在设备上修改 lX
    ctx.task(lX.rw())->*[&](cudaStream_t s, slice<double> dX){
        modify_on_device<<<1,1,0,s>>>(dX, 100.0);
        printf("  Task: dX(0) potentially modified on device.\n");
    };

    ctx.finalize(); // STF将数据写回 host_array_X
    printf("After finalize, host_array_X[0]: %f\n", host_array_X[0]);
    return 0;  
}
