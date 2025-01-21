/**
 * @file
 *
 * @brief The example program from CLHPP repository's documentation root
 * page - adapted to use the opencl-api-cpp library
 */
#ifndef CL_VERSION_2_0
#define CL_VERSION_2_0 1
#endif

#include <opencl.hpp>

#include <iostream>
#include <vector>
#include <ranges>
#include <algorithm>

[[noreturn]] bool die(std::string message)
{
    std::cerr << message << "\n";
    exit(EXIT_FAILURE);
}

opencl::platform_t get_first_opencl_2_x_platform() {
    auto platforms = opencl::platforms();
    auto it = std::ranges::find_if(platforms,
        [](opencl::platform_t const& p) { return p.version() >= opencl::make_version(2,0); });
    it < platforms.end() or die("No valid OpenCL platform found");
    return *it;
};

void print_output(std::span<int> output)
{
    std::cout << "Output:\n";
    for (auto x : output) {
        std::cout << "\t" << x << "\n";
    }
    std::cout << "\n\n";
}

int main()
{
    const int numElements = 32;

    // C++11 raw string literal for the first kernel
    constexpr auto kernel1{R"CLC(
        global int globalA;
        kernel void updateGlobal()
        {
          globalA = 75;
        }
    )CLC"};

    // Raw string literal for the second kernel
    constexpr auto kernel2{R"CLC(
        typedef struct { global int *bar; } Foo;
        kernel void vectorAdd(global const Foo* aNum, global const int *inputA, global const int *inputB,
                              global int *output, int val, write_only pipe int outPipe, queue_t childQueue)
        {
          output[get_global_id(0)] = inputA[get_global_id(0)] + inputB[get_global_id(0)] + val + *(aNum->bar);
          write_pipe(outPipe, &val);
          queue_t default_queue = get_default_queue();
          ndrange_t ndrange = ndrange_1D(get_global_size(0)/2, get_global_size(0)/2);

          // Have a child kernel write into third quarter of output
          enqueue_kernel(default_queue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange,
            ^{
                output[get_global_size(0)*2 + get_global_id(0)] =
                  inputA[get_global_size(0)*2 + get_global_id(0)] + inputB[get_global_size(0)*2 + get_global_id(0)] + globalA;
            });

          // Have a child kernel write into last quarter of output
          enqueue_kernel(childQueue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange,
            ^{
                output[get_global_size(0)*3 + get_global_id(0)] =
                  inputA[get_global_size(0)*3 + get_global_id(0)] + inputB[get_global_size(0)*3 + get_global_id(0)] + globalA + 2;
            });
        }
    )CLC"};

    auto platform = opencl::platform::get(1); //get_first_opencl_2_x_platform();
    auto context = opencl::context::create_with_all_devices(platform);
    auto sources = { kernel1, kernel2 };
    auto wrapped_sources = opencl::program::source::create(context, sources);
    auto options = opencl::program::compilation::options::create();
    options.language_version = opencl::make_version("2.0");
    auto build_result = opencl::program::build_(wrapped_sources, options);
    if (not build_result.succeeded()) {
        std::cerr << "Build failed.\n";
        for (auto const& target_build_info : build_result.info()) {
            auto target = target_build_info.device();
            std::cerr << "Build log for device " << target.name() << ":\n\n";
            std::cerr << target_build_info.log()<< std::endl;
        }
        exit(EXIT_FAILURE);
    }

    auto update_global = build_result.instantiate_kernel("updateGlobal");
    auto point_1d = opencl::nd_range::point(1);
    auto point_launch_config = opencl::builders::launch_config()
        .workgroup_dimensions(point_1d)
        .overall_dimensions(point_1d)
        .build();
    auto device = context.first_device();
    // You may be wondering why we're choosing just one device when we've created
    // a context with many devices and built our kernels for all of them. The answer
    // is that this is indeed rather silly, but that's what the original CLHPP
    // example program did.
    auto queue = opencl::queue::create(context, device);
    queue.enqueue_kernel_launch(update_global, point_launch_config);

    typedef struct { int *bar; } Foo;

    //////////////////
    // SVM allocations

    auto inputB = opencl::svm::allocate(context, sizeof(int) * numElements);
    auto foo_region = opencl::svm::allocate(context, sizeof(Foo));
    auto foo = static_cast<Foo*>(foo_region.get());
    auto foo_value_region = opencl::svm::allocate(context, sizeof(int));
    foo->bar = foo_value_region.as_span<int>().data();
    auto inputA = opencl::svm::allocate(context, numElements * sizeof(int));
    std::ranges::fill(inputA.as_span<int>(), 1);
    std::ranges::fill(inputB.as_span<int>(), 2);

    //////////////
    // Traditional allocations

    std::vector<int> output(numElements, 0xdeadbeef);

    auto output_buffer = opencl::buffer::create_copy_of(context, output);

    auto pipe = opencl::pipe::create(context, sizeof(cl_int), numElements / 2);
    auto extra_value = 3;

    auto vectorAdd = build_result.instantiate_kernel("vectorAdd");
    auto launch_config = opencl::builders::launch_config()
        .workgroup_size(numElements/2)
        .overall_size(numElements/2)
        .build();
    queue.enqueue_kernel_launch(vectorAdd, launch_config,
        foo,
        inputA.data(),
        inputB.data(),
        output_buffer,
        extra_value,
        pipe,
        queue.handle());

    queue.enqueue_copy(output_buffer, output);
    queue.finish();

    // Note: We're not freeing the shared virtual memory allocations - they will
    // be freed along with the context.

    print_output(output);
}
