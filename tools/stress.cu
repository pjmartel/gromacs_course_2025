// stress.cu
// Compile with: nvcc -arch=sm_80 stress.cu -o stress
#include <cuda_runtime.h>
#include <stdio.h>

// A kernel that loops forever doing trivial FP ops.
// 'volatile' prevents the compiler from eliding the loop.
__global__ void stress_kernel()
{
    volatile float x = 1.0f;
    while (1) {
        x = x * 1.0000001f + 0.0000001f;
        // (We don’t read 'x' back into global memory,
        // but marking it volatile forces each iteration.)
    }
}

int main()
{
    cudaError_t err;

    // Launch with 1024 blocks × 256 threads each (≈ 262K threads).
    stress_kernel<<<1024, 256>>>();

    // Catch any launch‐time failure (e.g. no CUDA device, invalid config).
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                ">>> Kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    // This will never return unless the kernel is killed or errors out.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // Common on Windows desktop cards if TDR (timeout detection)
        // kills the kernel for running too long.
        fprintf(stderr,
                ">>> cudaDeviceSynchronize error: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
