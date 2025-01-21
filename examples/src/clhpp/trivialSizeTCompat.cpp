#define CL_HPP_TARGET_OPENCL_VERSION 200
//#define CL_HPP_ENABLE_SIZE_T_COMPATIBILITY // what's this?

#include <opencl.hpp>
#include <iostream>
#include <vector>

[[noreturn]] bool die(std::string const& message)
// With C++17, we would have used a string_view
{
    std::cerr << message << '\n';
    exit(EXIT_FAILURE);
}

constexpr int numElements = 32;

int main()
{
    auto get_first_opencl_2_platform = [] {
        // Filter for a 2.x platform and set it as the default
        auto platforms = opencl::platforms();
        auto it = std::find_if(platforms.begin(), platforms.end(),
            [](opencl::platform_t const & p) { return p.version().major >= 2; });
        it < platforms.end() or die("No OpenCL 2.0-or-higher platform found");
        return *it;
    };

    auto platform = get_first_opencl_2_platform();
    auto device = platform.default_device();
    auto context = opencl::context::create(device);

    auto source_text = R"(
global int globalA;

kernel void updateGlobal(){
  globalA = 75;
}

kernel void vectorAdd(global const int *inputA, global const int *inputB, global int *output, int val){
  output[get_global_id(0)] = inputA[get_global_id(0)] + inputB[get_global_id(0)] + val;
}
)";

    auto wrapped_source = opencl::program::source::create(context, source_text );
    auto options = opencl::program::compilation::options::create();
    options.language_version = opencl::make_version("2.0");
    auto build_result = opencl::program::build_(wrapped_source, device, options);
    if (not build_result.succeeded()) {
        std::cerr << "Build failed." << std::endl;
        auto log = build_result.info_for(device).log();
        std::cerr << "Build log:" << "\n-----\n" << log << "\n----\n" << std::flush;
        exit(EXIT_FAILURE);
    }

    auto updateGlobal = opencl::kernel::instantiate(build_result, "updateGlobal");
    auto point_1d = opencl::nd_range::point(1);
    auto point_launch_config = opencl::builders::launch_config()
        .workgroup_dimensions(point_1d)
        .overall_dimensions(point_1d)
        .build();
    auto queue = opencl::queue::create(context, device);
    // This will also work:
    // opencl::launch(updateGlobal, queue, point_launch_config);
    queue.enqueue_kernel_launch(updateGlobal, point_launch_config);

    auto vectorAdd = build_result.instantiate_kernel("vectorAdd");

    std::vector<int> input_a(numElements, 1);
    std::vector<int> input_b(numElements, 2);
    std::vector<int> output(numElements, 0xdeadbeef);
    // TODO: Maybe a different name for this named constructor idioms, especially
    // if the buffer can be created with no allocation

    auto buffer_builder = opencl::builders::buffer()
        .context(context)
        .size(numElements * sizeof(int));

    auto input_a_buffer = buffer_builder.device_access(opencl::access_kind_t::read).create();
    auto input_b_buffer = buffer_builder.device_access(opencl::access_kind_t::read).create();
    auto output_buffer  = buffer_builder.device_access(opencl::access_kind_t::write).create();

    // TODO: Consider placing copy() in the main namespace - or using `using` to bring it up there
    opencl::memory::copy(input_a, input_a_buffer, queue);
    opencl::memory::copy(input_b, input_b_buffer, queue);

    queue.enqueue_copy(input_a, input_a_buffer);

    auto launch_config = opencl::builders::launch_config()
        .overall_size(numElements/2)
        .workgroup_size(numElements/2)
        .build();

    queue.enqueue_kernel_launch(vectorAdd, launch_config,
        input_a_buffer, input_b_buffer, output_buffer, 3);
    queue.enqueue_kernel_launch(vectorAdd, launch_config,
        input_a_buffer, input_b_buffer, output_buffer, 3);

    queue.finish();

    auto dim_constraint = vectorAdd.nd_range_info_for(device).source_workgroup_dimensions_constraint();
    std::cout << "Array return: " << dim_constraint[0] << ", " << dim_constraint[1] << ", " << dim_constraint[2] << '\n';

    opencl::memory::copy(output_buffer, output, queue);

    std::cout << "Output:\n";
    std::cout << std::hex;
    for (auto e : output) { std::cout << " " << e; }
    std::cout << "\n\nSUCCESS\n";
}
