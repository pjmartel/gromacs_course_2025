// stress.cu
// Compile with: nvcc -arch=sm_80 stress_percent.cu -o stress_percent
#include <cuda_runtime.h>
#include <stdio.h>
#include <thread>       // for std::this_thread::sleep_for
#include <chrono>       // for std::chrono::milliseconds
#include <cstdlib>      // for atoi()

// Kernel: each thread busy‐loops for “cycle_cycles” GPU clocks.
// We use __clock64() (64-bit GPU clock) + a volatile float to force real work.
__global__ void throttle_kernel(unsigned long long cycle_cycles)
{
    // Make 'x' volatile so the compiler cannot optimize away the loop.
    volatile float x = 1.0f;

    // Read GPU clock at start
    unsigned long long start = clock64();

    // Loop until (current_clock − start) >= cycle_cycles
    while ((clock64() - start) < cycle_cycles) {
        // A few cheap FP ops to keep the ALUs busy
        x = x * 1.0000001f + 0.0000001f;
    }
    // Once the loop is done, threads return and kernel ends.
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr,
                "Usage: %s <percent (0–100)>\n"
                "  e.g.  ./stress 75   → target ~75%% GPU busy\n",
                argv[0]);
        return 1;
    }

    // Parse and clamp percentage
    int percent = atoi(argv[1]);
    if (percent < 0)   percent = 0;
    if (percent > 100) percent = 100;

    // We'll use a fixed cycle of 100 ms
    const int cycle_ms  = 100;
    const int active_ms = (cycle_ms * percent) / 100;
    const int idle_ms   = cycle_ms - active_ms;

    // Query device clock rate (in kHz) to convert ms → GPU clock cycles
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Error getting device properties: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    // prop.clockRate is in kHz, i.e. cycles per millisecond = clockRate
    unsigned long long clk_per_ms = (unsigned long long)prop.clockRate;

    // Compute how many GPU clock cycles ≈ active_ms
    unsigned long long cycle_cycles = clk_per_ms * (unsigned long long)active_ms;

    printf(
      "→ Target GPU busy ratio: %d%%\n"
      "   (Active = %d ms, Idle = %d ms per 100 ms cycle)\n"
      "   GPU clockRate = %u kHz → %llu cycles/ms\n"
      "   => kernel busy‐wait cycles = %llu clocks per iteration\n\n",
      percent, active_ms, idle_ms,
      prop.clockRate, clk_per_ms,
      cycle_cycles
    );

    // Keep launching small “throttle_kernel” chunks forever.
    while (true) {
        if (active_ms > 0) {
            // Launch enough threads to saturate the GPU (e.g. 1024×256 ≈ 262k threads)
            throttle_kernel<<<1024, 256>>>(cycle_cycles);

            // Check for launch errors
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr,
                        ">>> Kernel launch failed: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }

            // Wait until this “busy” chunk finishes
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr,
                        ">>> cudaDeviceSynchronize error: %s\n",
                        cudaGetErrorString(err));
                return -1;
            }
        }

        // Sleep on CPU for the “idle” portion of the cycle
        if (idle_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
        }
        // Loop back for the next 100 ms cycle
    }

    return 0;
}
