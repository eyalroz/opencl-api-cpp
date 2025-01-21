
#ifndef OPENCL_WRAPPERS_IMPL_ERROR_HPP_
#define OPENCL_WRAPPERS_IMPL_ERROR_HPP_

#include "../error.hpp"

namespace opencl {

inline string_view description_of(status_t status)
{
    switch(status) {
        case CL_SUCCESS:                         return "Success";
        case CL_DEVICE_NOT_FOUND:                return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE:            return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:          return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:              return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:    return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:           return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:      return "Unsupported image format";
        case CL_BUILD_PROGRAM_FAILURE:           return "Program build failure";
        case CL_MAP_FAILURE:                     return "Memory mapping failure";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:    return "Misaligned sub-buffer offset";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "Execution status error for the events in the wait-list";
        case CL_COMPILE_PROGRAM_FAILURE:         return "Program compilation failure";
        case CL_LINKER_NOT_AVAILABLE:            return "Linker is not available";
        case CL_LINK_PROGRAM_FAILURE:            return "Program link failure";
        case CL_DEVICE_PARTITION_FAILED:         return "Device partitioning failed";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:   return "Information about kernel argument is not available";

    	// Kernel compilation errors
        case CL_INVALID_VALUE:                   return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:             return "Invalid device type";
        case CL_INVALID_PLATFORM:                return "Invalid platform";
        case CL_INVALID_DEVICE:                  return "Invalid device";
        case CL_INVALID_CONTEXT:                 return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:        return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:           return "Invalid command_queue";
        case CL_INVALID_HOST_PTR:                return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:              return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:              return "Invalid image size";
        case CL_INVALID_SAMPLER:                 return "Invalid sampler";
        case CL_INVALID_BINARY:                  return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:           return "Invalid build options";
        case CL_INVALID_PROGRAM:                 return "Invalid OpenCL program";
        case CL_INVALID_PROGRAM_EXECUTABLE:      return "Invalid OpenCL program executable";
        case CL_INVALID_KERNEL_NAME:             return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:       return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                  return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:               return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:               return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:             return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:          return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:         return "Invalid workgroup size";
        case CL_INVALID_WORK_ITEM_SIZE:          return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:           return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:         return "Invalid event wait-list";
        case CL_INVALID_EVENT:                   return "Invalid event";
        case CL_INVALID_OPERATION:               return "Invalid operation";
        case CL_INVALID_GL_OBJECT:               return "Invalid gl object";
        case CL_INVALID_BUFFER_SIZE:             return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:               return "Invalid MIP level";
        case CL_INVALID_GLOBAL_WORK_SIZE:        return "Invalid global work-size";
        case CL_INVALID_PROPERTY:                return "Invalid property";
        case CL_INVALID_IMAGE_DESCRIPTOR:        return "Invalid image descriptor";
        case CL_INVALID_COMPILER_OPTIONS:        return "Invalid compilation options";
        case CL_INVALID_LINKER_OPTIONS:          return "Invalid linking options";
        case CL_INVALID_DEVICE_PARTITION_COUNT:  return "Invalid device partition count";

        // extension-specific errors
        case -1000:                              return "Invalid GL share-group reference";
        case -1001:                              return "platform not found";
        case -1002:                              return "Invalid Direct3D 10 device";
        case -1003:                              return "Invalid Direct3D 10 resource";
        case -1004:                              return "Direct3D 10 resource already acquired";
        case -1005:                              return "Direct3D 10 resource not acquired";
        default:                                 return "Unknown OpenCL error";
    }
}

inline string_view description_of(program::build_status_t build_status)
{
    switch(static_cast<int>(build_status)) {
    case CL_BUILD_SUCCESS:         return "Success";
    case CL_BUILD_PROGRAM_FAILURE: return "Failure";
    case CL_BUILD_NONE:            return "None";
    case CL_BUILD_IN_PROGRESS:     return "In progress";
    default:                       return "Unknown status";
    }
}

namespace detail {

inline char const* description_of_cstr(status_t status)
{
    // This relies on the implementation of describe() using only string literals
    // (which end with a trailing '\0').
    return description_of(status).data();
}

template <bool UpperCase, typename I>
std::string as_hex(I x, unsigned num_hex_digits)
{
    static_assert(std::is_unsigned<I>::value, "only signed representations are supported");
    if (x == 0) return "0x0";

    enum { bits_per_hex_digit = 4 }; // = log_2 of 16
    static const char* digit_characters =
        UpperCase ? "0123456789ABCDEF" : "0123456789abcdef" ;

    std::string result(num_hex_digits,'0');
    for (unsigned digit_index = 0; digit_index < num_hex_digits ; digit_index++)
    {
        size_t bit_offset = (num_hex_digits - 1 - digit_index) * bits_per_hex_digit;
        auto hexadecimal_digit = (x >> bit_offset) & 0xF;
        result[digit_index] = digit_characters[hexadecimal_digit];
    }
    return "0x0" + result.substr(result.find_first_not_of('0'), std::string::npos);
}

template <typename T, bool UpperCase>
std::string ptr_as_hex(T const * ptr)
{
    return as_hex<UpperCase>(reinterpret_cast<uintptr_t>(ptr));
}

} // namespace detail

inline void throw_if_error(status_t status, char const* api_function_name, const std::string& message) noexcept(false)
{
    if (is_failure(status)) { throw runtime_error(status, api_function_name, message); }
}

inline void throw_if_error(status_t status, char const* api_function_name, std::string&& message) noexcept(false)
{
    if (is_failure(status)) { throw runtime_error(status, api_function_name, message); }
}

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_ERROR_HPP_
