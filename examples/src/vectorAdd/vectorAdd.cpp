#include "../common.hpp"
#include <opencl.hpp>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstdlib>

auto kernel_source_code = R"(
__kernel void vectorAdd(
    __global float       * __restrict C,
    __global float const * __restrict A,
    __global float const * __restrict B,
    unsigned long length)
{
  int i = get_global_id(0);
  if (i < length)
    C[i] = A[i] + B[i];
}
)";

void validate_results(
  std::vector<float> const& h_A,
  std::vector<float> const& h_B,
  std::vector<float> const& h_C)
{
  auto num_elements = h_A.size();
  // The sizes are known to be equal, since we created the vectors that way
  for (unsigned long i = 0; i < num_elements; ++i) {
    if (std::fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5f)  {
      std::cerr
        << "Result verification failed at element " << i << ": "
        << h_A[i] << " + " << h_B[i]  << " != " << h_C[i] << "\n";
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char* argv[])
{
  size_t num_elements = 8UL*32*512;

  auto platform_index = resolve_platform_index(argc, argv);
  auto platform = opencl::platform::get(platform_index);
  std::cout << "Using the " << xth(platform_index+1) << " platform (of " << opencl::platform::count() << "): " << platform.name() << "\n";

  auto device_outside_context = platform.default_device();
  auto context = opencl::context::create(device_outside_context);
  auto device = context.first_device();  // An OpenCL-contextualized GPU device
  std::cout << "Using the first platform device: " << device.name() << std::endl;

  auto source = opencl::program::source::create(context, {"vectorAdd", kernel_source_code});
  auto build_result = opencl::program::build_(source, device);
  if (not build_result.succeeded()) {
    auto log = build_result.info_for(device).log();
    std::cerr << "Build (compiling or linking) failed. Build log:"
              << "\n--------\n" << log << "\n-------\n" << std::flush;
    exit(EXIT_FAILURE);
  }

  auto queue = opencl::queue::create(device);

  auto vectorAdd = opencl::kernel::instantiate(build_result, "vectorAdd");
  // can't we get build_result["vectorAdd"] to work? It could return an
  // instantiation as an rvalue.

  auto h_A = std::vector<float>(num_elements); // could be a unique_span really
  auto h_B = std::vector<float>(num_elements);
  auto h_C = std::vector<float>(num_elements);

  auto generator = [] {
    static std::random_device random_device;
    static std::mt19937 randomness_generator { random_device() };
    static std::uniform_real_distribution<float> distribution { 0.0, 1.0 };
    return distribution(randomness_generator);
  };
  std::generate(h_A.begin(), h_A.end(), generator);
  std::generate(h_B.begin(), h_B.end(), generator);

  auto d_A = opencl::buffer::create(context, num_elements * sizeof(float));
  auto d_B = opencl::buffer::create(context, num_elements * sizeof(float));
  auto d_C = opencl::buffer::create(context, num_elements * sizeof(float));

  opencl::memory::copy(h_A, d_A, queue);
  opencl::memory::copy(h_B, d_B, queue);

  auto launch_config = opencl::builders::launch_config()
      .overall_size(num_elements)
      .workgroup_size(256)
      .build();

  opencl::enqueue_launch(
      vectorAdd, queue, launch_config,
      d_C, d_A, d_B, num_elements
  );

  opencl::memory::copy(d_C, h_C, queue);
  queue.finish();

  validate_results(h_A, h_B, h_C);
  std::cout << "SUCCESS\n";
}
