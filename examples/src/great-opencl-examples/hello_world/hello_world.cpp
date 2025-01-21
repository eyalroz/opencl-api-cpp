#include <opencl.hpp>
#include <fstream>
#include <iostream>

namespace cl = opencl;

#ifdef READ_KERNEL_FROM_FILE
std::string read_file_contents(char const* path)
{
    using iter_type = std::istreambuf_iterator<char>;
    std::ifstream stream(path);
    return std::string{iter_type{stream}, iter_type{}};
}
#else
static const auto kernel_source_literal = R"(
/**
 * This kernel function only fills a buffer with the sentence 'Hello World!'.
 **/

 __kernel void helloWorld(__global char* data){
    data[0] = 'H';    data[1] = 'e';    data[2] = 'l';    data[3] = 'l';    data[4] = 'o';
    data[5] = ' ';
    data[6] = 'W';    data[7] = 'o';    data[8] = 'r';    data[9] = 'l';    data[10] = 'd';    data[11] = '!';
    data[12] = '\n';
}
)";
#endif

int main()
{
    auto platform = cl::platforms().front();
    auto device = platform.devices().front();
    auto context = cl::context::create(device);
#if READ_KERNEL_FROM_FILE
    auto src = read_file_contents("hello_world.cl");
#else
    std::string src { kernel_source_literal };
#endif
    auto program = cl::program::source::create(context, src);
    auto build_result = compile_and_link(program);
    if (not build_result.succeeded()) {
        auto build_info = build_result.info_for(device);
        std::cerr
            << "Build Status: " << name_of(build_info.status()) << '\n'
            << "Build Log:\n"
            << build_info.log() << std::endl;
        exit(EXIT_FAILURE);
    }
    auto kernel = build_result.instantiate_kernel("helloWorld");

    char buf[16];
    auto memBuf = cl::builders::buffer()
        .context(context)
        .device_access(cl::access_kind_t::write)
        .host_access(cl::access_kind_t::read)
        .size(sizeof(buf))
        .create();

    auto queue = cl::queue::create(context, device);

    auto launch_config = cl::builders::launch_config()
        .workgroup_size(1)
        .grid_size(1)
        .build();
    queue.enqueue_kernel_launch(kernel, launch_config, memBuf);
    queue.enqueue_copy(memBuf, buf);
    queue.finish();
    std::cout << buf;
}
