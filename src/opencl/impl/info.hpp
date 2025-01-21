#ifndef OPENCL_WRAPPERS_IMPL_INFO_HPP
#define OPENCL_WRAPPERS_IMPL_INFO_HPP

#include <CL/cl.h>

#include "types.hpp"
#include "../util/miscellany.hpp"

namespace opencl {

namespace info {

/**
 * Kinds of OpenCL entities about which one can obtain information from the driver
 *
 * @note These kinds are in correspondence with the various clGetSomethingInfo API functions.
 */
enum source_kind_t {
    platform,
    device,
    context,
    memory_object,
    memobject = memory_object,
    image,
    pipe,
#if false
    commandbuffer,
#endif
    queue,
    event,
    program,
    kernel,
    gl_context,
    gl_object,
    gl_texture,
    event_profiling,
    profiling = event_profiling,
    sampler,
    // Can't use the following - their info getter takes extra parameter(s)
    program_build,
    build = program_build,
    kernel_parameter,
    kernel_workgroup,
    workgroup = kernel_workgroup,
    // kernel_subgroup
};

template<source_kind_t Kind> struct traits_t;

#define OCLW_DEFINE_TRAITS(source_kind_, handle_type_, attribute_id_type_, info_getter_) \
template<> struct traits_t<source_kind_> { \
    using handle_type = handle_type_; \
    using attribute_id_type = attribute_id_type_; \
    using info_getter_type = status_t (*)( \
        handle_type         handle, \
        attribute_id_type   attribute, \
        size_t              attribute_value_size, \
        void *              attribute_value, \
        size_t *            attribute_value_size_ret); \
    static status_t info_getter( \
        handle_type         handle, \
        attribute_id_type   attribute, \
        size_t              attribute_value_size, \
        void *              attribute_value, \
        size_t *            attribute_value_size_ret) \
    { \
        return info_getter_(handle, attribute, attribute_value_size, attribute_value, attribute_value_size_ret); \
    } \
    static char const* info_getter_name() noexcept { return STRINGIFY(info_getter_); }; \
    static char const* attribute_name(attribute_id_type) noexcept; \
};
// Note: If we were using C++17, we would have static const _data_ members for the info getter and the info getter's
// name - as it would be possible to initialize them within the class body. The current arrangement is a hack for
// supporting C++11. Not using #ifdef's here because it's a macro.
//
// TODO: Rework this switching to a C++17 dependence

namespace detail {

// Wrappers for a few of OpenCL info-getter functions, which forces
// them into the more common signature in which there's a single "handle"
// parameter
template <typename HandlePair>
cl_int handle_pair_info_getter(
    HandlePair                 handles,
    cl_kernel_work_group_info  param_name,
    size_t                     param_value_size,
    void *                     param_value,
    size_t *                   param_value_size_ret) noexcept;

template<> inline cl_int handle_pair_info_getter<kernel_device_handle_pair_t>(
    kernel_device_handle_pair_t   handles,
    cl_kernel_work_group_info     param_name,
    size_t                        param_value_size,
    void *                        param_value,
    size_t *                      param_value_size_ret) noexcept
{
    return clGetKernelWorkGroupInfo(
        handles.kernel_, handles.device_, param_name, param_value_size, param_value, param_value_size_ret);
}

template<> inline cl_int handle_pair_info_getter<program_device_handle_pair_t>(
    program_device_handle_pair_t  handles,
    cl_kernel_work_group_info     param_name,
    size_t                        param_value_size,
    void *                        param_value,
    size_t *                      param_value_size_ret) noexcept
{
    return clGetProgramBuildInfo(
        handles.program_, handles.device_, param_name, param_value_size, param_value, param_value_size_ret);
}

template<> inline cl_int handle_pair_info_getter<kernel_param_index_pair_t>(
    kernel_param_index_pair_t     handles,
    cl_kernel_arg_info            param_name,
    size_t                        param_value_size,
    void *                        param_value,
    size_t *                      param_value_size_ret) noexcept
{
    return clGetKernelArgInfo(
        handles.kernel_, handles.index_, param_name, param_value_size, param_value, param_value_size_ret);
}

} // namespace detail

OCLW_DEFINE_TRAITS(platform,        platform::handle_t, cl_platform_info,      clGetPlatformInfo);
OCLW_DEFINE_TRAITS(device,          device::handle_t,   cl_device_info,        clGetDeviceInfo);
OCLW_DEFINE_TRAITS(context,         context::handle_t,  cl_context_info,       clGetContextInfo);
OCLW_DEFINE_TRAITS(memory_object,   memory::handle_t,   cl_mem_info,           clGetMemObjectInfo);
OCLW_DEFINE_TRAITS(image,           image::handle_t,    cl_image_info,         clGetImageInfo);
OCLW_DEFINE_TRAITS(program,         program::handle_t,  cl_program_info,       clGetProgramInfo);
OCLW_DEFINE_TRAITS(kernel,          kernel::handle_t,   cl_kernel_info,        clGetKernelInfo);
OCLW_DEFINE_TRAITS(queue,           queue::handle_t,    cl_command_queue_info, clGetCommandQueueInfo);
OCLW_DEFINE_TRAITS(event,           event::handle_t,    cl_event_info,         clGetEventInfo);
OCLW_DEFINE_TRAITS(event_profiling, event::handle_t,    cl_profiling_info,     clGetEventProfilingInfo);
OCLW_DEFINE_TRAITS(sampler,         cl_sampler,         cl_sampler_info,       clGetSamplerInfo);
OCLW_DEFINE_TRAITS(pipe,            pipe::handle_t,     cl_pipe_info,          clGetPipeInfo);
//OCLW_DEFINE_TRAITS(kernel_argument, kernel::handle_t,   cl_kernel_arg_info,    clGetKernelArgInfo);
OCLW_DEFINE_TRAITS(program_build,   detail::program_device_handle_pair_t,   cl_program_build_info,
    detail::handle_pair_info_getter<detail::program_device_handle_pair_t>);
#if CL_VERSION_1_2
OCLW_DEFINE_TRAITS(kernel_parameter, detail::kernel_param_index_pair_t, cl_kernel_arg_info,
    detail::handle_pair_info_getter<detail::kernel_param_index_pair_t>);
#endif // CL_VERSION_1_2
OCLW_DEFINE_TRAITS(kernel_workgroup,       detail::kernel_device_handle_pair_t, cl_kernel_work_group_info,
    detail::handle_pair_info_getter<detail::kernel_device_handle_pair_t>);
// OCLW_DEFINE_TRAITS(kernel_subgroup, kernel::handle_t,     cl_kernel_sub_group_info, clGetKernelSubGroupInfo);

template <int InfoParameter>
constexpr source_kind_t kind() noexcept
{
    return static_cast<source_kind_t>(detail::raw_info_parameter_type_mapper<InfoParameter>::kind);
}

template<int InfoParameter> using handle_t = typename traits_t<kind<InfoParameter>()>::handle_type;
template<source_kind_t Kind> using attribute_id_t = typename traits_t<Kind>::attribute_id_type;

namespace detail {

// TODO: Consider moving this elsewhere
template <typename T, typename Property>
optional<T> lookup_property(Property const* properties, int property) noexcept
{
    static constexpr Property properties_array_terminator = 0;
    if (properties == nullptr) { return {}; }
    while (true) {
        auto prop = *properties;
        if (property == prop) { return reinterpret_cast<T>(properties[1]); }
        if (property == properties_array_terminator) { break; }
        properties += 2; // the property and the value
    }
    return {};
}

template <typename Property, typename T>
dynarray<Property> single_property(int property, T const& value)
{
    static_assert(sizeof(Property) == sizeof(T), "No support yet for properties with size different than a pointer");
    return {
        static_cast<Property>(property),
        reinterpret_cast<Property>(value),
        0
    };
}

// Array info parameters are signified by a pointer type (although the value
// type is _not_ a pointer type. The alternative would be adding another field,
// and I'd rather not do that.

template <int InfoParameter>
using is_array_ = std::is_pointer<typename raw_info_parameter_type_mapper<InfoParameter>::type>;

template <int InfoParameter>
constexpr bool is_array() noexcept { return is_array_<InfoParameter>::value; }

#define OCLW_TYPE_MAPPER(param__, kind__, type__) \
template<> struct raw_info_parameter_type_mapper<param__> { using type = type__; enum { kind = (kind__) }; };

OCLW_TYPE_MAPPER(CL_PLATFORM_PROFILE, info::platform,  char *)
OCLW_TYPE_MAPPER(CL_PLATFORM_VERSION, info::platform, char *)
OCLW_TYPE_MAPPER(CL_PLATFORM_NAME, info::platform, char *)
OCLW_TYPE_MAPPER(CL_PLATFORM_VENDOR, info::platform, char *)
OCLW_TYPE_MAPPER(CL_PLATFORM_EXTENSIONS, info::platform, char *)
OCLW_TYPE_MAPPER(CL_DEVICE_TYPE, info::device, cl_device_type)
OCLW_TYPE_MAPPER(CL_DEVICE_VENDOR_ID, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_COMPUTE_UNITS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_WORK_GROUP_SIZE, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_WORK_ITEM_SIZES, info::device, size_t *)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_CLOCK_FREQUENCY, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_ADDRESS_BITS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_READ_IMAGE_ARGS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_MEM_ALLOC_SIZE, info::device, cl_ulong)
OCLW_TYPE_MAPPER(CL_DEVICE_IMAGE2D_MAX_WIDTH, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_IMAGE2D_MAX_HEIGHT, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_IMAGE3D_MAX_WIDTH, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_IMAGE3D_MAX_HEIGHT, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_IMAGE3D_MAX_DEPTH, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_IMAGE_SUPPORT, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_PARAMETER_SIZE, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_SAMPLERS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MEM_BASE_ADDR_ALIGN, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_SINGLE_FP_CONFIG, info::device, cl_device_fp_config)
OCLW_TYPE_MAPPER(CL_DEVICE_DOUBLE_FP_CONFIG, info::device, cl_device_fp_config)
OCLW_TYPE_MAPPER(CL_DEVICE_HALF_FP_CONFIG, info::device, cl_device_fp_config)
OCLW_TYPE_MAPPER(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, info::device, cl_device_mem_cache_type)
OCLW_TYPE_MAPPER(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, info::device, cl_ulong)
OCLW_TYPE_MAPPER(CL_DEVICE_GLOBAL_MEM_SIZE, info::device, cl_ulong)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, info::device, cl_ulong)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_CONSTANT_ARGS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_LOCAL_MEM_TYPE, info::device, cl_device_local_mem_type)
OCLW_TYPE_MAPPER(CL_DEVICE_LOCAL_MEM_SIZE, info::device, cl_ulong)
OCLW_TYPE_MAPPER(CL_DEVICE_ERROR_CORRECTION_SUPPORT, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_PROFILING_TIMER_RESOLUTION, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_ENDIAN_LITTLE, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_AVAILABLE, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_COMPILER_AVAILABLE, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_EXECUTION_CAPABILITIES, info::device, cl_device_exec_capabilities)
OCLW_TYPE_MAPPER(CL_DEVICE_PLATFORM, info::device, platform::handle_t)
OCLW_TYPE_MAPPER(CL_DEVICE_NAME, info::device, char *)
OCLW_TYPE_MAPPER(CL_DEVICE_VENDOR, info::device, char *)
OCLW_TYPE_MAPPER(CL_DRIVER_VERSION, info::device, char *)
OCLW_TYPE_MAPPER(CL_DEVICE_PROFILE, info::device,  char *)
OCLW_TYPE_MAPPER(CL_DEVICE_VERSION, info::device,  char *)
OCLW_TYPE_MAPPER(CL_DEVICE_EXTENSIONS, info::device,  char *)
OCLW_TYPE_MAPPER(CL_CONTEXT_REFERENCE_COUNT, info::context,  cl_uint)
OCLW_TYPE_MAPPER(CL_CONTEXT_DEVICES, info::context,  device::handle_t *)
OCLW_TYPE_MAPPER(CL_CONTEXT_PROPERTIES, info::context,  cl_context_properties *)
OCLW_TYPE_MAPPER(CL_EVENT_COMMAND_QUEUE, info::event,  queue::handle_t)
OCLW_TYPE_MAPPER(CL_EVENT_COMMAND_TYPE, info::event,  cl_command_type)
OCLW_TYPE_MAPPER(CL_EVENT_REFERENCE_COUNT, info::event,  cl_uint)
OCLW_TYPE_MAPPER(CL_EVENT_COMMAND_EXECUTION_STATUS, info::event,  cl_int)
OCLW_TYPE_MAPPER(CL_PROFILING_COMMAND_QUEUED, info::profiling,  cl_ulong)
OCLW_TYPE_MAPPER(CL_PROFILING_COMMAND_SUBMIT, info::profiling,  cl_ulong)
OCLW_TYPE_MAPPER(CL_PROFILING_COMMAND_START, info::profiling,  cl_ulong)
OCLW_TYPE_MAPPER(CL_PROFILING_COMMAND_END, info::profiling,  cl_ulong)
OCLW_TYPE_MAPPER(CL_MEM_TYPE, info::memory_object,  cl_mem_object_type)
OCLW_TYPE_MAPPER(CL_MEM_FLAGS, info::memory_object,  cl_mem_flags)
OCLW_TYPE_MAPPER(CL_MEM_SIZE, info::memory_object,  size_t)
OCLW_TYPE_MAPPER(CL_MEM_HOST_PTR, info::memory_object,  void*)
OCLW_TYPE_MAPPER(CL_MEM_MAP_COUNT, info::memory_object,  cl_uint)
OCLW_TYPE_MAPPER(CL_MEM_REFERENCE_COUNT, info::memory_object,  cl_uint)
OCLW_TYPE_MAPPER(CL_MEM_CONTEXT, info::memory_object,  context::handle_t)
OCLW_TYPE_MAPPER(CL_IMAGE_FORMAT, info::image,  cl_image_format)
OCLW_TYPE_MAPPER(CL_IMAGE_ELEMENT_SIZE, info::image,  size_t)
OCLW_TYPE_MAPPER(CL_IMAGE_ROW_PITCH, info::image,  size_t)
OCLW_TYPE_MAPPER(CL_IMAGE_SLICE_PITCH, info::image,  size_t)
OCLW_TYPE_MAPPER(CL_IMAGE_WIDTH, info::image,  size_t)
OCLW_TYPE_MAPPER(CL_IMAGE_HEIGHT, info::image,  size_t)
OCLW_TYPE_MAPPER(CL_IMAGE_DEPTH, info::image,  size_t)
OCLW_TYPE_MAPPER(CL_SAMPLER_REFERENCE_COUNT, info::sampler,  cl_uint)
OCLW_TYPE_MAPPER(CL_SAMPLER_CONTEXT, info::sampler,  context::handle_t)
OCLW_TYPE_MAPPER(CL_SAMPLER_NORMALIZED_COORDS, info::sampler,  cl_bool)
OCLW_TYPE_MAPPER(CL_SAMPLER_ADDRESSING_MODE, info::sampler,  cl_addressing_mode)
OCLW_TYPE_MAPPER(CL_SAMPLER_FILTER_MODE, info::sampler,  cl_filter_mode)
OCLW_TYPE_MAPPER(CL_PROGRAM_REFERENCE_COUNT, info::program, cl_uint)
OCLW_TYPE_MAPPER(CL_PROGRAM_CONTEXT, info::program, context::handle_t)
OCLW_TYPE_MAPPER(CL_PROGRAM_NUM_DEVICES, info::program, cl_uint)
OCLW_TYPE_MAPPER(CL_PROGRAM_DEVICES, info::program, device::handle_t *)
OCLW_TYPE_MAPPER(CL_PROGRAM_SOURCE, info::program, char *)
OCLW_TYPE_MAPPER(CL_PROGRAM_BINARY_SIZES, info::program, size_t *)
OCLW_TYPE_MAPPER(CL_PROGRAM_BINARIES, info::program, char **)
OCLW_TYPE_MAPPER(CL_PROGRAM_BUILD_STATUS, info::program_build, cl_build_status)
OCLW_TYPE_MAPPER(CL_PROGRAM_BUILD_OPTIONS, info::program_build, char *)
OCLW_TYPE_MAPPER(CL_PROGRAM_BUILD_LOG, info::program_build, char *)
OCLW_TYPE_MAPPER(CL_KERNEL_FUNCTION_NAME, info::kernel, char *)
OCLW_TYPE_MAPPER(CL_KERNEL_NUM_ARGS, info::kernel, cl_uint)
OCLW_TYPE_MAPPER(CL_KERNEL_REFERENCE_COUNT, info::kernel, cl_uint)
OCLW_TYPE_MAPPER(CL_KERNEL_CONTEXT, info::kernel, context::handle_t)
OCLW_TYPE_MAPPER(CL_KERNEL_PROGRAM, info::kernel, program::handle_t)
OCLW_TYPE_MAPPER(CL_KERNEL_WORK_GROUP_SIZE, info::kernel_workgroup, size_t)
OCLW_TYPE_MAPPER(CL_KERNEL_COMPILE_WORK_GROUP_SIZE, info::kernel_workgroup, size_t *)
OCLW_TYPE_MAPPER(CL_KERNEL_LOCAL_MEM_SIZE, info::kernel_workgroup, cl_ulong)
OCLW_TYPE_MAPPER(CL_QUEUE_CONTEXT, info::queue, context::handle_t)
OCLW_TYPE_MAPPER(CL_QUEUE_DEVICE, info::queue, device::handle_t)
OCLW_TYPE_MAPPER(CL_QUEUE_REFERENCE_COUNT, info::queue, cl_uint)
OCLW_TYPE_MAPPER(CL_QUEUE_PROPERTIES, info::queue, cl_command_queue_properties)

#ifdef CL_VERSION_1_1
OCLW_TYPE_MAPPER(CL_CONTEXT_NUM_DEVICES, info::context, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_OPENCL_C_VERSION, info::device, char *)
OCLW_TYPE_MAPPER(CL_MEM_ASSOCIATED_MEMOBJECT, info::memory_object, memory::handle_t)
OCLW_TYPE_MAPPER(CL_MEM_OFFSET, info::memory_object, size_t)
OCLW_TYPE_MAPPER(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, info::kernel_workgroup, size_t)
OCLW_TYPE_MAPPER(CL_KERNEL_PRIVATE_MEM_SIZE, info::kernel_workgroup, cl_ulong)
OCLW_TYPE_MAPPER(CL_EVENT_CONTEXT, info::event, context::handle_t)
#endif // CL_VERSION_1_1

#ifdef CL_VERSION_1_2
OCLW_TYPE_MAPPER(CL_PROGRAM_NUM_KERNELS, info::program, size_t)
OCLW_TYPE_MAPPER(CL_PROGRAM_KERNEL_NAMES, info::program, char *)
OCLW_TYPE_MAPPER(CL_PROGRAM_BINARY_TYPE, info::program_build, cl_program_binary_type)
OCLW_TYPE_MAPPER(CL_KERNEL_ATTRIBUTES, info::kernel, char *)
OCLW_TYPE_MAPPER(CL_KERNEL_ARG_ADDRESS_QUALIFIER, info::kernel_parameter, cl_kernel_arg_address_qualifier)
OCLW_TYPE_MAPPER(CL_KERNEL_ARG_ACCESS_QUALIFIER, info::kernel_parameter, cl_kernel_arg_access_qualifier)
OCLW_TYPE_MAPPER(CL_KERNEL_ARG_TYPE_NAME, info::kernel_parameter, char *)
OCLW_TYPE_MAPPER(CL_KERNEL_ARG_NAME, info::kernel_parameter, char *)
OCLW_TYPE_MAPPER(CL_KERNEL_ARG_TYPE_QUALIFIER, info::kernel_parameter, cl_kernel_arg_type_qualifier)
OCLW_TYPE_MAPPER(CL_KERNEL_GLOBAL_WORK_SIZE, info::kernel_workgroup, size_t *)
OCLW_TYPE_MAPPER(CL_DEVICE_LINKER_AVAILABLE, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_PARENT_DEVICE, info::device, device::handle_t)
OCLW_TYPE_MAPPER(CL_DEVICE_PARTITION_MAX_SUB_DEVICES, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PARTITION_PROPERTIES, info::device, cl_device_partition_property *)
OCLW_TYPE_MAPPER(CL_DEVICE_PARTITION_TYPE, info::device, cl_device_partition_property *)  \
OCLW_TYPE_MAPPER(CL_DEVICE_REFERENCE_COUNT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_PARTITION_AFFINITY_DOMAIN, info::device, cl_device_affinity_domain)
OCLW_TYPE_MAPPER(CL_DEVICE_BUILT_IN_KERNELS, info::device, char *)
OCLW_TYPE_MAPPER(CL_DEVICE_PRINTF_BUFFER_SIZE, info::device, size_t)
OCLW_TYPE_MAPPER(CL_IMAGE_ARRAY_SIZE, info::image, size_t)
OCLW_TYPE_MAPPER(CL_IMAGE_NUM_MIP_LEVELS, info::image, cl_uint)
OCLW_TYPE_MAPPER(CL_IMAGE_NUM_SAMPLES, info::image, cl_uint)
#endif // CL_VERSION_1_2

#ifdef CL_VERSION_2_0
OCLW_TYPE_MAPPER(CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, info::device, cl_command_queue_properties)
OCLW_TYPE_MAPPER(CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, info::device, cl_command_queue_properties)
OCLW_TYPE_MAPPER(CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_ON_DEVICE_QUEUES, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_ON_DEVICE_EVENTS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_PIPE_ARGS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PIPE_MAX_PACKET_SIZE, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_SVM_CAPABILITIES, info::device, cl_device_svm_capabilities)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_IMAGE_PITCH_ALIGNMENT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, info::device, cl_uint )
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, info::device, size_t )
OCLW_TYPE_MAPPER(CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, info::device, size_t )
OCLW_TYPE_MAPPER(CL_PROFILING_COMMAND_COMPLETE, info::profiling, cl_ulong)
OCLW_TYPE_MAPPER(CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM, info::kernel, cl_bool)
OCLW_TYPE_MAPPER(CL_KERNEL_EXEC_INFO_SVM_PTRS, info::kernel, void**)
OCLW_TYPE_MAPPER(CL_QUEUE_SIZE, info::queue, cl_uint)
OCLW_TYPE_MAPPER(CL_MEM_USES_SVM_POINTER, info::memory_object, cl_bool)
OCLW_TYPE_MAPPER(CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE, info::program_build, size_t)
OCLW_TYPE_MAPPER(CL_PIPE_PACKET_SIZE, info::pipe, cl_uint)
OCLW_TYPE_MAPPER(CL_PIPE_MAX_PACKETS, info::pipe, cl_uint)
#endif // CL_VERSION_2_0

// #define CL_HPP_PARAM_NAME_INFO_SUBGROUP_KHR_(F)
// OCLW_TYPE_MAPPER(CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR, info::kernel_subgroup, size_t)
// OCLW_TYPE_MAPPER(CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR, info::kernel_subgroup, size_t)
//
// #define CL_HPP_PARAM_NAME_INFO_IL_KHR_(F)
// OCLW_TYPE_MAPPER(CL_DEVICE_IL_VERSION_KHR, info::device, char *)
// OCLW_TYPE_MAPPER(CL_PROGRAM_IL_KHR, info::program, cl::vector<unsigned char>)

#ifdef CL_VERSION_2_1
OCLW_TYPE_MAPPER(CL_PLATFORM_HOST_TIMER_RESOLUTION, info::platform, cl_ulong)
OCLW_TYPE_MAPPER(CL_PROGRAM_IL, info::program, char *)
OCLW_TYPE_MAPPER(CL_DEVICE_MAX_NUM_SUB_GROUPS, info::device, cl_uint)
OCLW_TYPE_MAPPER(CL_DEVICE_IL_VERSION, info::device, char *)
OCLW_TYPE_MAPPER(CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_QUEUE_DEVICE_DEFAULT, info::queue, queue::handle_t)
OCLW_TYPE_MAPPER(CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, info::kernel, size_t)
OCLW_TYPE_MAPPER(CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE, info::kernel, size_t)
OCLW_TYPE_MAPPER(CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, info::kernel, size_t)
OCLW_TYPE_MAPPER(CL_KERNEL_MAX_NUM_SUB_GROUPS, info::kernel, size_t)
OCLW_TYPE_MAPPER(CL_KERNEL_COMPILE_NUM_SUB_GROUPS, info::kernel, size_t)
#endif // CL_VERSION_2_1

#ifdef CL_VERSION_2_2
OCLW_TYPE_MAPPER(CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT, info::program, cl_bool)
OCLW_TYPE_MAPPER(CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT, info::program, cl_bool)
#endif // CL_VERSION_2_2

// #define CL_HPP_PARAM_NAME_DEVICE_FISSION_EXT_(F)
// OCLW_TYPE_MAPPER(CL_DEVICE_PARENT_DEVICE_EXT, info::device, device::handle_t)
// OCLW_TYPE_MAPPER(CL_DEVICE_PARTITION_TYPES_EXT, info::device, cl::vector<cl_device_partition_property_ext>)
// OCLW_TYPE_MAPPER(CL_DEVICE_AFFINITY_DOMAINS_EXT, info::device, cl::vector<cl_device_partition_property_ext>)
// OCLW_TYPE_MAPPER(CL_DEVICE_REFERENCE_COUNT_EXT , info::device, cl_uint)
// OCLW_TYPE_MAPPER(CL_DEVICE_PARTITION_STYLE_EXT, info::device, cl::vector<cl_device_partition_property_ext>)
//
// #define CL_HPP_PARAM_NAME_CL_KHR_EXTENDED_VERSIONING_CL3_SHARED_(F)
// OCLW_TYPE_MAPPER(CL_PLATFORM_NUMERIC_VERSION_KHR, info::platform, cl_version_khr)
// OCLW_TYPE_MAPPER(CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR, info::platform, cl::vector<cl_name_version_khr>)
// OCLW_TYPE_MAPPER(CL_DEVICE_NUMERIC_VERSION_KHR, info::device, cl_version_khr)
// OCLW_TYPE_MAPPER(CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR, info::device, cl::vector<cl_name_version_khr>)
// OCLW_TYPE_MAPPER(CL_DEVICE_ILS_WITH_VERSION_KHR, info::device, cl::vector<cl_name_version_khr>)
// OCLW_TYPE_MAPPER(CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR, info::device, cl::vector<cl_name_version_khr>)
//
// #define CL_HPP_PARAM_NAME_CL_KHR_EXTENDED_VERSIONING_KHRONLY_(F)
// OCLW_TYPE_MAPPER(CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR, info::device, cl_version_khr)
//
// // Note: the query for CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR is handled specially!
// #define CL_HPP_PARAM_NAME_CL_KHR_SEMAPHORE_(F)
// OCLW_TYPE_MAPPER(CL_SEMAPHORE_CONTEXT_KHR, info::semaphore, context::handle_t)
// OCLW_TYPE_MAPPER(CL_SEMAPHORE_REFERENCE_COUNT_KHR, info::semaphore, cl_uint)
// OCLW_TYPE_MAPPER(CL_SEMAPHORE_PROPERTIES_KHR, info::semaphore, cl::vector<cl_semaphore_properties_khr>)
// OCLW_TYPE_MAPPER(CL_SEMAPHORE_TYPE_KHR, info::semaphore, cl_semaphore_type_khr)
// OCLW_TYPE_MAPPER(CL_SEMAPHORE_PAYLOAD_KHR, info::semaphore, cl_semaphore_payload_khr)
// OCLW_TYPE_MAPPER(CL_PLATFORM_SEMAPHORE_TYPES_KHR, info::platform,  cl::vector<cl_semaphore_type_khr>)
// OCLW_TYPE_MAPPER(CL_DEVICE_SEMAPHORE_TYPES_KHR, info::device,      cl::vector<cl_semaphore_type_khr>)
//
// #define CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_MEMORY_(F)
// OCLW_TYPE_MAPPER(CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR, info::device, cl::vector<cl::ExternalMemoryType>)
// OCLW_TYPE_MAPPER(CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR, info::platform, cl::vector<cl::ExternalMemoryType>)
//
// #define CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_(F)
// OCLW_TYPE_MAPPER(CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR, info::platform,  cl::vector<cl_external_semaphore_handle_type_khr>)
// OCLW_TYPE_MAPPER(CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR, info::platform,  cl::vector<cl_external_semaphore_handle_type_khr>)
// OCLW_TYPE_MAPPER(CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR, info::device,      cl::vector<cl_external_semaphore_handle_type_khr>)
// OCLW_TYPE_MAPPER(CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR, info::device,      cl::vector<cl_external_semaphore_handle_type_khr>)
// OCLW_TYPE_MAPPER(CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR, info::semaphore,      cl::vector<cl_external_semaphore_handle_type_khr>)
//
// #define CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_OPAQUE_FD_EXT(F)
// OCLW_TYPE_MAPPER(CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR, info::semaphore, int)
//
// #define CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_SYNC_FD_EXT(F)
// OCLW_TYPE_MAPPER(CL_SEMAPHORE_HANDLE_SYNC_FD_KHR, info::semaphore, int)
//
// #define CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_WIN32_EXT(F)
// OCLW_TYPE_MAPPER(CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR, info::semaphore, void*)
// OCLW_TYPE_MAPPER(CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR, info::semaphore, void*)

#ifdef CL_VERSION_3_0
OCLW_TYPE_MAPPER(CL_PLATFORM_NUMERIC_VERSION, info::platform, cl_version)
OCLW_TYPE_MAPPER(CL_PLATFORM_EXTENSIONS_WITH_VERSION, info::platform, cl_name_version *)
OCLW_TYPE_MAPPER(CL_DEVICE_NUMERIC_VERSION, info::device, cl_version)
OCLW_TYPE_MAPPER(CL_DEVICE_EXTENSIONS_WITH_VERSION, info::device, cl_name_version *)
OCLW_TYPE_MAPPER(CL_DEVICE_ILS_WITH_VERSION, info::device, cl_name_version *)
OCLW_TYPE_MAPPER(CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION, info::device, cl_name_version *)
OCLW_TYPE_MAPPER(CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, info::device, cl_device_atomic_capabilities)
OCLW_TYPE_MAPPER(CL_DEVICE_ATOMIC_FENCE_CAPABILITIES, info::device, cl_device_atomic_capabilities)
OCLW_TYPE_MAPPER(CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_OPENCL_C_ALL_VERSIONS, info::device, cl_name_version *)
OCLW_TYPE_MAPPER(CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, info::device, size_t)
OCLW_TYPE_MAPPER(CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_OPENCL_C_FEATURES, info::device, cl_name_version *)
OCLW_TYPE_MAPPER(CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES, info::device, cl_device_device_enqueue_capabilities)
OCLW_TYPE_MAPPER(CL_DEVICE_PIPE_SUPPORT, info::device, cl_bool)
OCLW_TYPE_MAPPER(CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED, info::device, char *)
OCLW_TYPE_MAPPER(CL_QUEUE_PROPERTIES_ARRAY, info::queue, cl_queue_properties *)
OCLW_TYPE_MAPPER(CL_MEM_PROPERTIES, info::memory_object, cl_mem_properties *)
OCLW_TYPE_MAPPER(CL_PIPE_PROPERTIES, info::pipe, cl_pipe_properties *)
OCLW_TYPE_MAPPER(CL_SAMPLER_PROPERTIES, info::sampler, cl_sampler_properties *)
#endif // CL_VERSION_3_0

// #define CL_HPP_PARAM_NAME_CL_IMAGE_REQUIREMENTS_EXT(F)
// OCLW_TYPE_MAPPER(CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT, info::image, size_t)
// OCLW_TYPE_MAPPER(CL_IMAGE_REQUIREMENTS_BASE_ADDRESS_ALIGNMENT_EXT, info::image, size_t)
// OCLW_TYPE_MAPPER(CL_IMAGE_REQUIREMENTS_SIZE_EXT, info::image, size_t)
// OCLW_TYPE_MAPPER(CL_IMAGE_REQUIREMENTS_MAX_WIDTH_EXT, info::image, cl_uint)
// OCLW_TYPE_MAPPER(CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT, info::image, cl_uint)
// OCLW_TYPE_MAPPER(CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT, info::image, cl_uint)
// OCLW_TYPE_MAPPER(CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT, info::image, cl_uint)
//
// #define CL_HPP_PARAM_NAME_CL_IMAGE_REQUIREMENTS_SLICE_PITCH_ALIGNMENT_EXT(F)
// OCLW_TYPE_MAPPER(CL_IMAGE_REQUIREMENTS_SLICE_PITCH_ALIGNMENT_EXT, info::image, size_t)
//
// #define CL_HPP_PARAM_NAME_CL_INTEL_COMMAND_QUEUE_FAMILIES_(F)
// OCLW_TYPE_MAPPER(CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL, info::device, cl::vector<cl_queue_family_properties_intel>)
// OCLW_TYPE_MAPPER(CL_QUEUE_FAMILY_INTEL, info::queue, cl_uint)
// OCLW_TYPE_MAPPER(CL_QUEUE_INDEX_INTEL, info::queue, cl_uint)
//
// #define CL_HPP_PARAM_NAME_CL_INTEL_UNIFIED_SHARED_MEMORY_(F)
// OCLW_TYPE_MAPPER(CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL, info::device, cl_device_unified_shared_memory_capabilities_intel )
// OCLW_TYPE_MAPPER(CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL, info::device, cl_device_unified_shared_memory_capabilities_intel )
// OCLW_TYPE_MAPPER(CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL, info::device, cl_device_unified_shared_memory_capabilities_intel )
// OCLW_TYPE_MAPPER(CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL, info::device, cl_device_unified_shared_memory_capabilities_intel )
// OCLW_TYPE_MAPPER(CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL, info::device, cl_device_unified_shared_memory_capabilities_intel )

#undef OCLW_TYPE_MAPPER

} // namespace detail

template <int InfoParameter>
using element_type_t = typename std::remove_pointer<parameter_value_t<InfoParameter>>::type;

template <int InfoParameter>
value_or_status_t<parameter_value_t<InfoParameter>>
get_scalar_nothrow(handle_t<InfoParameter> handle) noexcept
{
    using value_type = parameter_value_t<InfoParameter>;
    constexpr source_kind_t kind_ = kind<InfoParameter>();
    value_or_status_t<value_type> result;
    result.status = traits_t<kind_>::info_getter(handle, InfoParameter, sizeof(value_type), &result.value, nullptr);
    return result;
}

template <int InfoParameter>
parameter_value_t<InfoParameter>
get_scalar(handle_t<InfoParameter> handle)
{
    using traits = traits_t<kind<InfoParameter>()>;
    auto result = get_scalar_nothrow<InfoParameter>(handle);
    return_or_throw_if_error(result, traits::info_getter_name(),
        std::string("Obtaining attribute ") + traits::attribute_name(InfoParameter)
            + " for " + opencl::detail::identify(handle));
}

template <int InfoParameter>
value_or_status_t<size_t> get_array_size_bytes_nothrow(handle_t<InfoParameter> handle) noexcept
{
    using traits = traits_t<kind<InfoParameter>()>;
    value_or_status_t<size_t> result;
    result.status = traits::info_getter(handle, InfoParameter, 0, nullptr, &result.value);
    return result;
}

template <int InfoParameter>
size_t get_array_size_bytes(handle_t<InfoParameter> handle)
{
    using traits = traits_t<kind<InfoParameter>()>;
    auto size_bytes = get_array_size_bytes_nothrow<InfoParameter>(handle);
    return_or_throw_if_error(size_bytes, traits::info_getter_name(),
        std::string("Obtaining the size of ") + traits::attribute_name(InfoParameter)
            + " for " + opencl::detail::identify(handle));
}

template <int InfoParameter>
value_or_status_t<size_t> get_array_size_nothrow(handle_t<InfoParameter> handle) noexcept
{
    using element_type = element_type_t<InfoParameter>;
    auto size_bytes = get_array_size_bytes_nothrow<InfoParameter>(handle);
    if (not size_bytes) { return { {}, size_bytes.status }; }
    size_t size = *size_bytes / sizeof(element_type);
    auto status = (size * sizeof(element_type) == *size_bytes) ? status::success : CL_INVALID_VALUE;
    return { size, status };
}

template <int InfoParameter>
size_t get_array_size(handle_t<InfoParameter> handle)
{
    using element_type = element_type_t<InfoParameter>;
    size_t size_bytes = get_array_size_bytes<InfoParameter>(handle);
    size_t size = size_bytes / sizeof(element_type);
    if (size * sizeof(element_type) != size_bytes) {
        throw std::runtime_error(std::string{"The size of attribute "}
            + traits_t<kind<InfoParameter>()>::attribute_name(InfoParameter)
            + " of " + opencl::detail::identify(handle) + " is not a multiple of its element type size");
    }
    return size;
}

template <int InfoParameter>
status_t get_array_nothrow(handle_t<InfoParameter> handle, span<element_type_t<InfoParameter>> storage)
{
    using element_type = element_type_t<InfoParameter>;
    auto buffer_size_bytes = storage.size() * sizeof(element_type);
    return traits_t<kind<InfoParameter>()>::info_getter(handle, InfoParameter, buffer_size_bytes, storage.data(), nullptr);
}

template <int InfoParameter>
void get_array(handle_t<InfoParameter> handle, span<element_type_t<InfoParameter>> storage)
{
    using element_type = element_type_t<InfoParameter>;
    using traits = traits_t<kind<InfoParameter>()>;
    auto buffer_size_bytes = storage.size() * sizeof(element_type);
    auto status = traits::info_getter(
        handle, InfoParameter, buffer_size_bytes, storage.data(), nullptr);
    throw_if_error_lazy(status, "clGetKernelInfo", std::string{"Obtaining the info attribute "}
        + traits::attribute_name(InfoParameter) + " of "
        + opencl::detail::identify(handle));
}

template <int InfoParameter>
value_or_status_t<dynarray<element_type_t<InfoParameter>>>
get_array_nothrow(handle_t<InfoParameter> handle) noexcept
{
    using element_type = element_type_t<InfoParameter>;
    auto size = get_array_size_nothrow<InfoParameter>(handle);
    if (not size) { return { {}, size }; }
    value_or_status_t<dynarray<element_type>> arr = { make_dynarray<element_type>(*size), {} };
    arr.status = get_array_nothrow<InfoParameter>(handle, *arr);
    return arr;
}

template <int InfoParameter>
dynarray<element_type_t<InfoParameter>>
get_array(handle_t<InfoParameter> handle)
{
    using element_type = element_type_t<InfoParameter>;
    auto size = get_array_size<InfoParameter>(handle);
    if (size == 0) {
        throw std::runtime_error(std::string{"No "} + traits_t<kind<InfoParameter>()>::attribute_name(InfoParameter)
            + " (i.e. its size is 0) for " + opencl::detail::identify(handle));
    }
    auto result = make_dynarray<element_type>(size);
    get_array<InfoParameter>(handle, result);
    return result;
}

// TODO: Is size-above-0 a valid criterion generally for OpenCL? I wonder.
template <int InfoParameter>
bool has_array(handle_t<InfoParameter> handle)
{
    auto size = get_array_size_bytes<InfoParameter>(handle);
    return (size > 0);
}

template <int InfoParameter>
bool has_string(handle_t<InfoParameter> handle)
{
    return has_array<InfoParameter>(handle);
}

template <int InfoParameter>
value_or_status_t<std::string>
get_string_nothrow(handle_t<InfoParameter> handle) noexcept
{
    auto arr = get_array_nothrow<InfoParameter>(handle);
    if (not arr) { return { {}, arr.status }; }
    if (arr->empty()) {
        // This should not happen - the string should be null-terminated; but let's be accommodating
        return { {}, status::invalid_value };
    }
    return {std::string{arr->data(), arr->size() - 1}, status::success }; // the 1 is for the trailing nul character
}

template <int InfoParameter>
std::string get_string(handle_t<InfoParameter> handle)
{
    // TODO: Try to avoid the gratuitous copying
    auto arr = get_array<InfoParameter>(handle);
    return std::string{arr.data(), arr.size() - 1}; // the 1 is for the trailing nul character
}

template <int InfoParameter>
dynarray<std::string> get_tokenized_string(handle_t<InfoParameter> handle, char delimiter)
{
    auto untokenized = get_array<InfoParameter>(handle);
    auto untokenized_as_sv = string_view{untokenized.data(), untokenized.size()};
    auto tokens_in_arr = opencl::detail::split(untokenized_as_sv, delimiter);
    static auto generator = [&](size_t i) { return tokens_in_arr[i]; };
    return generate_dynarray<std::string>(tokens_in_arr.size(), generator);
}

} // namespace info
} // namespace opencl

#endif //OPENCL_WRAPPERS_IMPL_INFO_HPP
