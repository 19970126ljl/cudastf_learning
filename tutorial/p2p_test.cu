#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        printf("需要至少两个GPU进行P2P测试，当前只有 %d 个GPU\n", deviceCount);
        return 0;
    }

    printf("检测 %d 个GPU的P2P支持情况:\n\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) continue;

            int canAccessPeer = 0;
            cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
            printf("GPU %d -> GPU %d: %s\n", i, j,
                   canAccessPeer ? "支持" : "不支持");

            if (canAccessPeer) {
                cudaSetDevice(i);
                cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                if (err == cudaErrorPeerAccessAlreadyEnabled) {
                    printf("  已启用访问\n");
                } else if (err != cudaSuccess) {
                    printf("  启用失败: %s\n", cudaGetErrorString(err));
                } else {
                    printf("  启用成功\n");
                }
            }
        }
    }

    // 简单的内存拷贝测试（可选）
    int size = 1024 * sizeof(int);
    int *d0, *d1;
    cudaSetDevice(0);
    cudaMalloc(&d0, size);
    cudaSetDevice(1);
    cudaMalloc(&d1, size);

    cudaSetDevice(0);
    cudaMemcpyPeer(d0, 0, d1, 1, size); // 从GPU1拷贝到GPU0
    printf("\n执行cudaMemcpyPeer完成\n");

    cudaFree(d0);
    cudaSetDevice(1);
    cudaFree(d1);

    return 0;
}
