//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief A reduction kernel written using launch
 */

#include <cuda/experimental/stf.cuh>
#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace cuda::experimental::stf;

double X0(int i)
{
  return sin((double) i);
}

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]\n";
  std::cout << "Options:\n";
  std::cout << "  -s <num>    Number of repeat streams (default: 0, no repeat)\n";
  std::cout << "  -n <num>    Number of devices to use (default: 1)\n";
  std::cout << "  -w <num>    Number of warmup iterations (default: 3)\n";
  std::cout << "  -i <num>    Number of benchmark iterations (default: 10)\n";
  std::cout << "  -h          Show this help message\n";
  std::cout << "\nExamples:\n";
  std::cout << "  " << program_name << " -n 2           # Use 2 devices\n";
  std::cout << "  " << program_name << " -s 4           # Use 4 repeat streams on device 0\n";
  std::cout << "  " << program_name << " -n 2 -s 3      # Use 2 devices, each with 3 repeat streams\n";
  std::cout << "  " << program_name << " -w 5 -i 20     # 5 warmup iterations, 20 benchmark iterations\n";
}

int main(int argc, char* argv[])
{
  context ctx;

  // Default values
  int number_devices = 1;
  int repeat_streams = 0;
  int warmup_iterations = 3;
  int benchmark_iterations = 10;

  // Parse command line options
  int opt;
  while ((opt = getopt(argc, argv, "s:n:w:i:h")) != -1) {
    switch (opt) {
      case 's':
        repeat_streams = std::atoi(optarg);
        if (repeat_streams <= 0) {
          std::cerr << "Error: repeat_streams must be a positive integer" << std::endl;
          return 1;
        }
        break;
      case 'n':
        number_devices = std::atoi(optarg);
        if (number_devices <= 0) {
          std::cerr << "Error: number_devices must be a positive integer" << std::endl;
          return 1;
        }
        break;
      case 'w':
        warmup_iterations = std::atoi(optarg);
        if (warmup_iterations < 0) {
          std::cerr << "Error: warmup_iterations must be a non-negative integer" << std::endl;
          return 1;
        }
        break;
      case 'i':
        benchmark_iterations = std::atoi(optarg);
        if (benchmark_iterations <= 0) {
          std::cerr << "Error: benchmark_iterations must be a positive integer" << std::endl;
          return 1;
        }
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  std::cout << "Using number_devices = " << number_devices << std::endl;
  std::cout << "Using repeat_streams = " << repeat_streams << std::endl;
  std::cout << "Warmup iterations = " << warmup_iterations << std::endl;
  std::cout << "Benchmark iterations = " << benchmark_iterations << std::endl;

  const size_t N = 128 * 1024 * 1024;

  std::vector<double> X(N);
  double sum = 0.0;

  double ref_sum = 0.0;

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = sin((double) ind);
    ref_sum += X[ind];
  }

  auto lX   = ctx.logical_data(&X[0], {N});
  auto lsum = ctx.logical_data(&sum, {1});

  // Choose execution place based on options
  exec_place where;

  if (repeat_streams > 0 && number_devices > 1) {
    // Combine both: repeat on multiple devices
    // Create a vector of repeated execution places for each device
    std::vector<exec_place> device_repeats;
    for (int dev = 0; dev < number_devices; dev++) {
      // Add each repeat instance to the vector
      for (int rep = 0; rep < repeat_streams; rep++) {
        device_repeats.push_back(exec_place::device(dev));
      }
    }
    where = make_grid(std::move(device_repeats));
    std::cout << "Using " << repeat_streams << " repeat streams on each of " << number_devices << " devices" << std::endl;
  } else if (repeat_streams > 0) {
    // Use repeat streams on device 0
    where = exec_place::repeat(exec_place::device(0), repeat_streams);
    std::cout << "Using " << repeat_streams << " repeat streams on device 0" << std::endl;
  } else {
    // Use n_devices for multi-device execution
    where = exec_place::n_devices(number_devices);
    std::cout << "Using " << number_devices << " devices execution place" << std::endl;
  }

  // Function to perform the reduction computation
  auto perform_reduction = [&]() -> double {
    context local_ctx;
    double local_sum = 0.0;

    auto local_lX = local_ctx.logical_data(&X[0], {N});
    auto local_lsum = local_ctx.logical_data(&local_sum, {1});

    auto spec = par<16>(con<32>());

    local_ctx.launch(spec, where, local_lX.read(), local_lsum.rw())->*[] _CCCL_DEVICE(auto th, auto x, auto sum) {
      // Each thread computes the sum of elements assigned to it
      double local_sum = 0.0;
      for (auto i : th.apply_partition(shape(x)))
      {
        local_sum += x(i);
      }

      auto ti = th.inner();

      __shared__ double block_sum[th.static_width(1)];
      block_sum[ti.rank()] = local_sum;

      for (size_t s = ti.size() / 2; s > 0; s /= 2)
      {
        ti.sync();
        if (ti.rank() < s)
        {
          block_sum[ti.rank()] += block_sum[ti.rank() + s];
        }
      }

      if (ti.rank() == 0)
      {
        // if (th.rank() == 0)
        //   printf("the address of sum is %p\n", &sum(0));
        atomicAdd_system(&sum(0), block_sum[0]);
      }
    };

    local_ctx.finalize();
    printf("local_sum = %lf\n", local_sum);
    return local_sum;
  };

  // Warmup phase
  std::cout << "\n=== Warmup Phase ===" << std::endl;
  for (int w = 0; w < warmup_iterations; w++) {
    std::cout << "Warmup iteration " << (w + 1) << "/" << warmup_iterations << std::endl;
    double warmup_result = perform_reduction();
    // Verify correctness during warmup
    if (fabs(warmup_result - ref_sum) >= 0.0001) {
      std::cerr << "Warning: Warmup iteration " << (w + 1) << " failed correctness check!" << std::endl;
      std::cerr << "Expected: " << ref_sum << ", Got: " << warmup_result << std::endl;
    }
  }

  // Benchmark phase
  std::cout << "\n=== Benchmark Phase ===" << std::endl;
  std::vector<double> execution_times;
  execution_times.reserve(benchmark_iterations);

  for (int i = 0; i < benchmark_iterations; i++) {
    std::cout << "Benchmark iteration " << (i + 1) << "/" << benchmark_iterations << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    double result = perform_reduction();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    execution_times.push_back(duration.count() / 1000.0); // Convert to milliseconds

    // Verify correctness
    if (fabs(result - ref_sum) >= 0.0001) {
      std::cerr << "Error: Benchmark iteration " << (i + 1) << " failed correctness check!" << std::endl;
      std::cerr << "Expected: " << ref_sum << ", Got: " << result << std::endl;
      return 1;
    }
  }

  // Calculate and display statistics
  std::sort(execution_times.begin(), execution_times.end());
  double mean_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / execution_times.size();
  double median_time = execution_times[execution_times.size() / 2];
  double min_time = execution_times.front();
  double max_time = execution_times.back();

  std::cout << "\n=== Performance Results ===" << std::endl;
  std::cout << "Mean execution time: " << mean_time << " ms" << std::endl;
  std::cout << "Median execution time: " << median_time << " ms" << std::endl;
  std::cout << "Min execution time: " << min_time << " ms" << std::endl;
  std::cout << "Max execution time: " << max_time << " ms" << std::endl;
  std::cout << "Standard deviation: ";

  // Calculate standard deviation
  double variance = 0.0;
  for (double time : execution_times) {
    variance += (time - mean_time) * (time - mean_time);
  }
  variance /= execution_times.size();
  double std_dev = sqrt(variance);
  std::cout << std_dev << " ms" << std::endl;

  // Calculate throughput
  double data_size_gb = (N * sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
  double throughput_gb_s = data_size_gb / (mean_time / 1000.0);
  std::cout << "Data size: " << data_size_gb << " GB" << std::endl;
  std::cout << "Mean throughput: " << throughput_gb_s << " GB/s" << std::endl;

  std::cout << "\n=== Test Passed ===" << std::endl;
}
