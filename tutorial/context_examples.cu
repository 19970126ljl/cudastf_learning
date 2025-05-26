#include <cuda/experimental/stf.cuh>
#include <cstdio> // For printf

using namespace cuda::experimental::stf;

// Dummy kernel for demonstration
__global__ void my_kernel(int *data, int val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *data = val;
    }
}

int main() {
    int host_data[1];
    host_data[0] = 0;

    // 1. 使用通用的 `context`，并将其赋值为一个 graph_ctx 实例
    printf("Using generic_ctx_as_graph:\n");
    context generic_ctx_as_graph = graph_ctx();
    auto l_data_g = generic_ctx_as_graph.logical_data(host_data);
    generic_ctx_as_graph.task(l_data_g.write())->*[&](cudaStream_t s, slice<int> d_data_g){
        my_kernel<<<1,1,0,s>>>(d_data_g.data_handle(), 10);
    };
    generic_ctx_as_graph.finalize();
    printf("  generic_ctx_as_graph: host_data = %d\n", host_data[0]);
    host_data[0] = 0; // Reset for next context

    // 2. 静态选择基于 CUDA 流和事件的上下文
    printf("Using stream_context:\n");
    stream_ctx stream_context;
    auto l_data_s = stream_context.logical_data(host_data);
    stream_context.task(l_data_s.write())->*[&](cudaStream_t s, slice<int> d_data_s){
        my_kernel<<<1,1,0,s>>>(d_data_s.data_handle(), 20);
    };
    stream_context.finalize();
    printf("  stream_context: host_data = %d\n", host_data[0]);
    host_data[0] = 0; // Reset for next context

    // 3. 静态选择基于 CUDA 图的上下文
    printf("Using graph_context:\n");
    graph_ctx graph_context;
    auto l_data_static_g = graph_context.logical_data(host_data);
    graph_context.task(l_data_static_g.write())->*[&](cudaStream_t s, slice<int> d_data_static_g){
        my_kernel<<<1,1,0,s>>>(d_data_static_g.data_handle(), 30);
    };
    graph_context.finalize();
    printf("  graph_context: host_data = %d\n", host_data[0]);

    printf("Context examples finished.\n");
    return 0;
}
