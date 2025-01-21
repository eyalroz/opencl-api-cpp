#include <opencl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>
#include <unistd.h>

#ifdef READ_KERNEL_SOURCE_FROM_FILE
std::string read_file_contents(char const* path)
{
    using iter_type = std::istreambuf_iterator<char>;
    std::ifstream stream(path);
    return std::string{iter_type{stream}, iter_type{}};
}
#else
static const auto kernel_source_literal = R"(
/**
 * This kernel function sums two arrays of integers and returns its result
 * through a third array.
 **/

__kernel void sumArrays(__global int const * a, __global int const* b, __global int* c){
     int index = get_global_id(0);
     // if (index == 0) {
        // printf("At index 0, a[0] = %d, b[0] = %d, c[0] = %d\n", a[0], b[0], c[0]);
     // }
     c[index] = a[index] + b[index];
}
)";
#endif

namespace cl = opencl;
using cl::span; // Or use C++20's span if you have that

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

struct wide_context_t {
    cl::context_t context;
    cl::program::source_t source;
    cl::program::link_t program;
    cl::kernel_t kernel;
};

wide_context_t initialize();
void seqSumArrays(span<int const> a, span<int const> b, span<int> c);
void parSumArrays(
    cl::kernel_t const& kernel,
    span<int const> a,
    span<int const> b,
    span<int> c);

// =================================================================
// ------------------------- Main Function -------------------------
// =================================================================

int main()
{
    clock_t start, end;
    int num_executions = 10;

    int ARRAYS_DIM = 1 << 20;
    std::vector<int> a(ARRAYS_DIM, 3);
    std::vector<int> b(ARRAYS_DIM, 5);

    std::vector<int> cs(ARRAYS_DIM);
    std::vector<int> cp(ARRAYS_DIM);

    start = clock();
    for(int i = 0; i < num_executions; i++){
        seqSumArrays(cl::as_span(a), cl::as_span(b), cl::as_span(cs));
    }
    end = clock();
    double seqTime = 10e3 * (end - start) / CLOCKS_PER_SEC / num_executions;

    auto wide_context = initialize();

    start = clock();
    for(int i = 0; i < num_executions; i++) {
        std::cout << "Execution " << i << std::endl;
        parSumArrays(wide_context.kernel,
            cl::as_span(a), cl::as_span(b), cl::as_span(cp));
    }
    end = clock();
    double parTime = 10e3 * (end - start) / CLOCKS_PER_SEC / num_executions;

    bool par_equals_seq = std::equal(cs.begin(), cs.end(), cp.data());

    std::cout << "Status: " << (par_equals_seq ? "SUCCESS!" : "FAILED!") << std::endl;
    std::cout << "Results: \n\ta[0] = " << a[0] << "\n\tb[0] = " << b[0] << "\n\tc[0] = a[0] + b[0] = " << cp[0] << std::endl;
    std::cout << "Mean execution time: \n\tSequential: " << seqTime << " ms;\n\tParallel: " << parTime << " ms." << std::endl;
    std::cout << "Performance gain: " << (100 * (seqTime - parTime) / parTime) << "%\n";
}

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================


wide_context_t initialize()
{
    auto platform = cl::platforms().front();
    auto device = platform.devices(cl::device::type_t::gpu).front();
    auto context = device.create_context();

    /**
     * Read OpenCL kernel file as a string.
     * */

#ifdef READ_KERNEL_SOURCE_FROM_FILE
    auto kernel_filename = "array_addition.cl";
    std::string src = read_file_contents(kernel_filename);
#else
    std::string src = kernel_source_literal;
#endif

    auto source = cl::program::source::create(context, {"array_addition", src});
    auto link_result = cl::program::compile_and_link(source);
    if (not link_result.succeeded()) {
        auto build_info = link_result.info_for(device);
        std::cerr << "Error!\nBuild Status: " << name_of(build_info.status())
            << "\nBuild Log:\t " << build_info.log() << std::endl;
        exit(EXIT_FAILURE);
    }
    auto kernel = link_result.instantiate_kernel("sumArrays");
    return {
        std::move(context), std::move(source), std::move(link_result), std::move(kernel)
    };
}

/**
 * Sequentially performs the N-dimensional operation c = a + b.
 * */

void seqSumArrays(span<int const> a, span<int const> b, span<int> c){
    for(size_t i = 0; i < c.size(); i++){
        c[i] = a[i] + b[i];
    }
}

/**
 * Parallelly performs the N-dimensional operation c = a + b.
 * */

void parSumArrays(
    opencl::kernel_t const& kernel,
    span<int const> a,
    span<int const> b,
    span<int> c)
{
    auto context = kernel.context();
    // auto context_alt = kernel.context();
    // if (context_alt != context) {
    //     std::cerr << "They differ.\n";
    // }
    auto input_builder = cl::builders::buffer()
        .context(context)
        // .disable_host_access(cl::access_kind_t::read)
        // .device_access(cl::access_kind_t::read);
        .device_access(cl::access_kind_t::read_and_write)
        .host_access(cl::access_kind_t::read_and_write);
    auto aBuf = input_builder.host_region_to_copy(a).create();
    auto bBuf = input_builder.host_region_to_copy(b).create();
    auto cBuf = cl::builders::buffer()
        .context(context)
        .size_by(c)
        // .device_access(cl::access_kind_t::write)
        .device_access(cl::access_kind_t::read_and_write)
        .host_access(cl::access_kind_t::read_and_write)
        .create();

    auto queue = context.create_queue(context.first_device());
    auto launch_config = cl::builders::launch_config()
        .workgroup_size(1)
        .grid_size(c.size())
        .build();
    queue.enqueue_kernel_launch(kernel, launch_config, aBuf, bBuf, cBuf);
    queue.enqueue_copy(cBuf, c);
    queue.finish();
}