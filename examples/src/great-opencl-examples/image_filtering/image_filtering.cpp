/*
 * TODO:
 * 1. Use a 3-value enum for the channels, then make the channel-split image hold a length-3
 *    array and have a single channel() method taking a channel enum
 * 2. Make the image types owning, and pass non-const references
 */

// #define CL_HPP_TARGET_OPENCL_VERSION 300

#include "CImg.h"
#include "../../common.hpp"

#include <opencl.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

using namespace cimg_library;

using byte = unsigned char; // could use std::byte in C++17
using image_dimension_t = uint32_t;
using image_buffer_t = std::vector<byte>; // could have used opencl::dynarray
struct image_dimensions_t {
    image_dimension_t width, height;
    image_dimension_t area() const { return width * height; }
};

// A reference-type
struct channel_split_image_t {
    image_dimensions_t dimensions;
    byte * red_;
    byte * green_;
    byte * blue_;

    image_dimension_t area() const { return dimensions.area(); }
    opencl::span<byte> red() const { return { red_, area() }; }
    opencl::span<byte> green() const { return { red_, area() }; }
    opencl::span<byte> blue() const { return { red_, area() }; }
};

// A reference-type
struct single_channel_image_t {
    image_dimensions_t dimensions;
    byte * gray;
    byte & operator[](ptrdiff_t offset) const { return gray[offset]; }
    image_dimension_t area() const { return dimensions.area(); }
};

// An owning type
struct filter_t {
    using element_type = float;
    using buffer_type = std::vector<element_type>;
    image_dimension_t size; // the square root of mask.size()
    buffer_type mask; // a linearized size x size matrix; could have used an mdspan in C++20
};

struct opencl_state_t {
    opencl::context_t context;
    opencl::program::link_t build_result;
    opencl::queue_t queue;
};

void seqRgb2Gray(channel_split_image_t const & image, byte *output_buffer);
void sequential_convolve(single_channel_image_t image,
                         filter_t const & filter,
                         byte* output_image);

void sequential_filter(channel_split_image_t const& image,
                       filter_t const& low_pass,
                       filter_t const& high_pass,
                       byte* output_image);

void display_image(single_channel_image_t const & image);

opencl_state_t setup_opencl_filtering(int argc, char **argv);

void parallel_filter(opencl_state_t const& opencl_state,
                     channel_split_image_t const& image,
                     filter_t const& low_pass_filter,
                     filter_t const& high_pass_filter,
                     byte* output_image);


template <typename F, typename... Ts>
std::chrono::milliseconds time_execution(F&& f, Ts&&... args_for_f)
{
    using namespace std::chrono;
    auto start = steady_clock::now();
    f(std::forward<Ts>(args_for_f)...);
    auto end = steady_clock::now();
    return duration_cast<milliseconds>(end - start);
}

int main(int argc, char** argv){

    CImg<byte> cimg("input_img.jpg");
    auto image = [&] {
        channel_split_image_t result;
        result.dimensions.width = cimg.width();
        result.dimensions.height = cimg.height();
        result.red_ = cimg.data();
        result.green_ = result.red_ + result.area();
        result.blue_ = result.green_ + result.area();
        return result;
    }();

    auto low_pass_filter = []() -> filter_t {
        constexpr size_t size = 5;
        static float mask[size][size] = {
            {.04,.04,.04,.04,.04},
            {.04,.04,.04,.04,.04},
            {.04,.04,.04,.04,.04},
            {.04,.04,.04,.04,.04},
            {.04,.04,.04,.04,.04},
        };
        return { size, { &mask[0][0], &mask[0][0] + size * size } };
    }();

    auto high_pass_filter = []() -> filter_t {
        constexpr size_t size = 5;
        static float mask[size][size] = {
            {-1,-1,-1,-1,-1},
            {-1,-1,-1,-1,-1},
            {-1,-1,24,-1,-1},
            {-1,-1,-1,-1,-1},
            {-1,-1,-1,-1,-1},
        };
        return { size, { &mask[0][0], &mask[0][0] + size * size } };
    }();

    struct { image_buffer_t sequential, parallel; } filter_output_buffers;
    filter_output_buffers.sequential = image_buffer_t(image.area());
    filter_output_buffers.parallel = image_buffer_t(image.area());

    auto sequential_duration = time_execution([&] {
        sequential_filter(image, low_pass_filter, high_pass_filter, filter_output_buffers.sequential.data());
    });

    auto opencl_state = setup_opencl_filtering(argc, argv);

    auto parallel_duration = time_execution([&] {
        parallel_filter(opencl_state, image, low_pass_filter, high_pass_filter, filter_output_buffers.parallel.data());
    });
    bool equal = std::equal(
        filter_output_buffers.sequential.begin(),
        filter_output_buffers.sequential.end(),
        filter_output_buffers.parallel.begin());

    std::cout << "Status: " << (equal ? "SUCCESS!" : "FAILED!") << std::endl;
    std::cout << "Mean execution time: \n\tSequential: " << sequential_duration.count()
        << " ms;\n\tParallel: " << parallel_duration.count() << " ms." << std::endl;
    std::cout << "Performance gain: " << (100 * (sequential_duration - parallel_duration) / parallel_duration) << "%\n";

    display_image({ image.dimensions, filter_output_buffers.parallel.data() });
}

std::string read_file_contents(char const* path)
{
    using file_iter = std::istreambuf_iterator<char>;
    std::ifstream hello_world_file(path);
    return std::string{file_iter(hello_world_file), file_iter()};
}

opencl_state_t setup_opencl_filtering(int argc, char **argv)
{
    auto platform_index = resolve_platform_index(argc, argv);
    auto platform = opencl::platforms()[platform_index];
    std::cout << "Using the " << xth(platform_index+1) << " platform (of " << opencl::platform::count() << "): " << platform.name() << "\n";
    auto device = platform.default_device();
    auto src = read_file_contents("image_filtering.cl");
    auto context = opencl::context::create(device);
    auto src_program = opencl::program::source::create(context, src);
    auto compilation_result = opencl::program::compile(src_program);
    if (not compilation_result.succeeded()) {
        auto build_info = compilation_result.info_for(device);
        std::cerr
            << "Build Status: " << name_of(build_info.status()) << '\n'
            << "Build Log:\n"
            << build_info.log() << std::endl;
        exit(EXIT_FAILURE);
    }
    auto link = opencl::program::link_(compilation_result);
    if (not link.succeeded()) {
        auto build_info = link.info_for(device);
        std::cerr
            << "Build Status: " << name_of(build_info.status()) << '\n'
            << "Number of kernels: " << link.num_kernels() << '\n'
            << "Build Log:\n"
            << build_info.log() << std::endl;
        exit(EXIT_FAILURE);
    }
    auto queue = opencl::queue::create(context, device);
    return { std::move(context), std::move(link), std::move(queue) };
}

void parallel_filter(opencl_state_t const& opencl_state,
                     channel_split_image_t const& image,
                     filter_t const& low_pass_filter,
                     filter_t const& high_pass_filter,
                     byte* output_image)
{
    auto & context = opencl_state.context;
    auto & build_result = opencl_state.build_result;
    struct {
        opencl::kernel_t gray, filter;
    } kernels = {
        build_result.instantiate_kernel("rgb2gray"),
        build_result.instantiate_kernel("filterImageWithCache")
    };

    // create buffers

    auto input_builder = opencl::builders::buffer()
        .context(context)
        .device_access(opencl::access_kind_t::read)
        .no_host_access();
    auto redBuf = input_builder.host_region_to_copy(image.red()).create();
    auto greenBuf = input_builder.host_region_to_copy(image.green()).create();
    auto blueBuf = input_builder.host_region_to_copy(image.blue()).create();
    auto low_pass_mask_buf = input_builder.host_region_to_copy(low_pass_filter.mask).create();
    auto high_pass_mask_buf = input_builder.host_region_to_copy(high_pass_filter.mask).create();

    auto output_builder = opencl::builders::buffer()
        .context(context)
        .device_access(opencl::access_kind_t::write)
        .no_host_access()
        .size(image.area());
    auto grayscale_image = output_builder.create();
    auto low_pass_output = output_builder.create();
    auto high_pass_output = output_builder.enable_host_access(opencl::access_kind_t::read).create();

    // set up the computational work

    auto const& queue = opencl_state.queue;
    auto gray_launch_config = opencl::builders::launch_config()
        .overall_dimensions(image.dimensions.width, image.dimensions.height)
        .workgroup_size(1)
        .build();
    queue.enqueue_kernel_launch(kernels.gray, gray_launch_config, redBuf, greenBuf, blueBuf, grayscale_image);
    auto filter_launch_config = opencl::builders::launch_config()
        .overall_dimensions(image.dimensions.width, image.dimensions.height)
        .workgroup_dimensions(16,16)
        .round_up_overall_dims()
        .build();
    queue.enqueue_kernel_launch(kernels.filter, filter_launch_config,
        static_cast<unsigned>(low_pass_filter.mask.size()), grayscale_image, low_pass_mask_buf, low_pass_output,
        image.dimensions.width, image.dimensions.height);
    queue.enqueue_kernel_launch(kernels.filter, filter_launch_config,
        static_cast<unsigned>(high_pass_filter.mask.size()), low_pass_output, high_pass_mask_buf, high_pass_output,
        image.dimensions.width, image.dimensions.height);
    queue.enqueue_copy(high_pass_output, opencl::memory::region_t { output_image, image.area() });

    // ... and actually perform all of that work!

    queue.finish();
}

void seqRgb2Gray(channel_split_image_t const & image, byte *output_buffer)
{
    for(image_dimension_t i = 0; i < image.dimensions.width; i++){
        for(image_dimension_t j = 0; j < image.dimensions.height; j++){
            size_t idx = i + j*image.dimensions.width;
            auto r = image.red_[idx];
            auto g = image.green_[idx];
            auto b = image.blue_[idx];
            output_buffer[idx] = (r+g+b)/3;

            //output_buffer[idx] = (image.red_[idx] + image.green_[idx] + image.blue_[idx]) / 3;
        }
    }
}

void sequential_convolve(single_channel_image_t image,
                         filter_t const & filter,
                         byte* output_image)
{
    for(image_dimension_t i = 0; i < image.dimensions.width; i++){
        for(image_dimension_t j = 0; j < image.dimensions.height; j++){
                
            /**
             * Check if the mask cannot be applied to the
             * current image pixel.
             * */
            
            if (i < filter.size/2
               || j < filter.size/2
               || i >= image.dimensions.width - filter.size/2
               || j >= image.dimensions.height - filter.size/2)
            {
                output_image[i + j * image.dimensions.width] = 0;
                continue;
            }
            
            /**
             * Apply mask based on the neighborhood of pixel inputImg(j,i).
             * */
            
            float outSum = 0;
            for(image_dimension_t k = 0; k < filter.size; k++){
                for(image_dimension_t l = 0; l < filter.size; l++){
                  image_dimension_t colIdx = i - filter.size/2 + k;
                  image_dimension_t rowIdx = j - filter.size/2 + l;
                  image_dimension_t maskIdx = (filter.size-1-k) + (filter.size-1-l)*filter.size;
                  outSum += image[rowIdx * image.dimensions.width + colIdx] * filter.mask[maskIdx];
                }
            }

            /**
             * Update output pixel.
             * */

            if(outSum < 0){
                output_image[i + j * image.dimensions.width] = 0;
            } else if(outSum > 255){
                output_image[i + j * image.dimensions.width] = 255;
            } else{
                output_image[i + j * image.dimensions.width] = outSum;
            }
        }
    }
}

void sequential_filter(channel_split_image_t const& image,
                       filter_t const& low_pass,
                       filter_t const& high_pass,
                       byte* output_image)
{
    image_buffer_t rgb2gray_out_buffer(image.area());
    auto rgb2gray_output = single_channel_image_t { image.dimensions,  rgb2gray_out_buffer.data() };
    seqRgb2Gray(image, rgb2gray_output.gray);
    image_buffer_t low_pass_output_buffer(image.area());
    single_channel_image_t low_pass_output { image.dimensions, low_pass_output_buffer.data() };
    sequential_convolve(rgb2gray_output, low_pass, low_pass_output.gray);
    sequential_convolve(low_pass_output, high_pass, output_image);
}

void display_image(single_channel_image_t const & image)
{
    CImg<byte> cimg(image.dimensions.width, image.dimensions.height);

    for(image_dimension_t i = 0; i < image.dimensions.width; i++){
        for(image_dimension_t j = 0; j < image.dimensions.height; j++){
            cimg(i,j) = image[i + image.dimensions.width*j];
        }
    }
    cimg.display();
}
