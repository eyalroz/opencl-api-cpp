
#ifndef OPENCL_WRAPPERS_IMPL_DEVICE_HPP_
#define OPENCL_WRAPPERS_IMPL_DEVICE_HPP_

#include "types.hpp"
#include "../platform.hpp"
#include "../device.hpp"
#include "../device_capabilities.hpp"
#include "info.hpp"
#include "platform.hpp"
#include "opencl/util/mpark_variant.hpp"

namespace opencl {

inline char const* info::traits_t<info::device>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case CL_DEVICE_TYPE:                                   return "type";
    case CL_DEVICE_VENDOR_ID:                              return "vendor_id";
    case CL_DEVICE_MAX_COMPUTE_UNITS:                      return "maximun number of compute units";
    case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:               return "maximum launch grid dimensions in work-items";
    case CL_DEVICE_MAX_WORK_GROUP_SIZE:                    return "maximum workgroup size";
    case CL_DEVICE_MAX_WORK_ITEM_SIZES:                    return "maximum work-item sizes";
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:            return "preffered vector witdh for CHAR";
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:           return "preffered vector witdh for SHORT";
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:             return "preffered vector witdh for INT";
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:            return "preffered vector witdh for LONG";
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:           return "preffered vector witdh for FLOAT";
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:          return "preffered vector witdh for DOUBLE";
    case CL_DEVICE_MAX_CLOCK_FREQUENCY:                    return "MAX_CLOCK_FREQUENCY";
    case CL_DEVICE_ADDRESS_BITS:                           return "number of bits covering the address space";
    case CL_DEVICE_MAX_READ_IMAGE_ARGS:                    return "maximum possible number of read-only image parameters for a kernel";
    case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:                   return "max possible number of write-only image parameters for a kernel";
    case CL_DEVICE_MAX_MEM_ALLOC_SIZE:                     return "maximum memory allocation size";
    case CL_DEVICE_IMAGE2D_MAX_WIDTH:                      return "maximum 2D image width";
    case CL_DEVICE_IMAGE2D_MAX_HEIGHT:                     return "maximum 2D image height";
    case CL_DEVICE_IMAGE3D_MAX_WIDTH:                      return "maximum 3D image width";
    case CL_DEVICE_IMAGE3D_MAX_HEIGHT:                     return "maximum 3D image height";
    case CL_DEVICE_IMAGE3D_MAX_DEPTH:                      return "maximum 3D image depth";
    case CL_DEVICE_IMAGE_SUPPORT:                          return "support for images";
    case CL_DEVICE_MAX_PARAMETER_SIZE:                     return "maximum parameter size";
    case CL_DEVICE_MAX_SAMPLERS:                           return "maximum number of samplers";
    case CL_DEVICE_MEM_BASE_ADDR_ALIGN:                    return "memory address base alignment";
    case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:               return "minimum alignment size of data types";
    case CL_DEVICE_SINGLE_FP_CONFIG:                       return "single floating-point capabilities";
    case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:                  return "global memory cache type";
    case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:              return "global memory cache line size in bytes";
    case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:                  return "overall size in bytes of global memory cache";
    case CL_DEVICE_GLOBAL_MEM_SIZE:                        return "global memory size";
    case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:               return "maximum constant buffer size";
    case CL_DEVICE_MAX_CONSTANT_ARGS:                      return "maximum number of constant parameters";
    case CL_DEVICE_LOCAL_MEM_TYPE:                         return "local memory type";
    case CL_DEVICE_LOCAL_MEM_SIZE:                         return "local memory size in bytes";
    case CL_DEVICE_ERROR_CORRECTION_SUPPORT:               return "error correction support";
    case CL_DEVICE_PROFILING_TIMER_RESOLUTION:             return "profiling timer resolution";
    case CL_DEVICE_ENDIAN_LITTLE:                          return "little-endian";
    case CL_DEVICE_AVAILABLE:                              return "availability";
    case CL_DEVICE_COMPILER_AVAILABLE:                     return "compiler availability";
    case CL_DEVICE_EXECUTION_CAPABILITIES:                 return "executionc apabilities";
#ifdef CL_VERSION_2_0
    case CL_DEVICE_QUEUE_PROPERTIES:
#else
    case CL_DEVICE_QUEUE_ON_HOST_PROPERTIES:
#endif
                                                           return "supported on-host command queue properties";
    case CL_DEVICE_NAME:                                   return "name";
    case CL_DEVICE_VENDOR:                                 return "vendor name";
    case CL_DRIVER_VERSION:                                return "driver version";
    case CL_DEVICE_PROFILE:                                return "profile";
    case CL_DEVICE_VERSION:                                return "driver version string";
    case CL_DEVICE_EXTENSIONS:                             return "extension names";
    case CL_DEVICE_PLATFORM:                               return "platform";
#ifdef CL_VERSION_1_2
    case CL_DEVICE_DOUBLE_FP_CONFIG:                       return "double floating-point configuration";
#endif
/* 0x1033 reserved for CL_DEVICE_HALF_FP_CONFIG which is already defined in "cl_ext.h" */
#ifdef CL_VERSION_1_1
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:            return "preffered vector witdh for the FP16 (half) type";
    case CL_DEVICE_HOST_UNIFIED_MEMORY:                    return "host unified memory";
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:               return "native vector witdh for the char type";
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:              return "native vector witdh for short type";
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:                return "native vector witdh for int type";
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:               return "native vector witdh for long type";
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:              return "native vector witdh for FP32 (single) type";
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:             return "native vector witdh for FP64 (double) type";
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:               return "native vector witdh for FP16 (half) type";
    case CL_DEVICE_OPENCL_C_VERSION:                       return "OpenCL C standard version";
#endif
#ifdef CL_VERSION_1_2
    case CL_DEVICE_LINKER_AVAILABLE:                       return "linker availability";
    case CL_DEVICE_BUILT_IN_KERNELS:                       return "built-in kernels";
    case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:                  return "maximum image buffer size";
    case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:                   return "maximum image array size";
    case CL_DEVICE_PARENT_DEVICE:                          return "parent device";
    case CL_DEVICE_PARTITION_MAX_SUB_DEVICES:              return "maximum sub-devices in a partition";
    case CL_DEVICE_PARTITION_PROPERTIES:                   return "sub-device partition properties";
    case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:              return "sub-device partition affinity domain";
    case CL_DEVICE_PARTITION_TYPE:                         return "partition type";
    case CL_DEVICE_REFERENCE_COUNT:                        return "reference count";
    case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:            return "preferred interoperability user synchronization";
    case CL_DEVICE_PRINTF_BUFFER_SIZE:                     return "buffer size for printf() calls";
#endif
#ifdef CL_VERSION_2_0
    case CL_DEVICE_IMAGE_PITCH_ALIGNMENT:                  return "IMAGE_PITCH_ALIGNMENT";
    case CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT:           return "IMAGE_BASE_ADDRESS_ALIGNMENT";
    case CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS:              return "MAX_READ_WRITE_IMAGE_ARGS";
    case CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE:               return "MAX_GLOBAL_VARIABLE_SIZE";
    case CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES:             return "QUEUE_ON_DEVICE_PROPERTIES";
    case CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE:         return "QUEUE_ON_DEVICE_PREFERRED_SIZE";
    case CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE:               return "QUEUE_ON_DEVICE_MAX_SIZE";
    case CL_DEVICE_MAX_ON_DEVICE_QUEUES:                   return "MAX_ON_DEVICE_QUEUES";
    case CL_DEVICE_MAX_ON_DEVICE_EVENTS:                   return "MAX_ON_DEVICE_EVENTS";
    case CL_DEVICE_SVM_CAPABILITIES:                       return "SVM_CAPABILITIES";
    case CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE:   return "GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE";
    case CL_DEVICE_MAX_PIPE_ARGS:                          return "MAX_PIPE_ARGS";
    case CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS:           return "PIPE_MAX_ACTIVE_RESERVATIONS";
    case CL_DEVICE_PIPE_MAX_PACKET_SIZE:                   return "PIPE_MAX_PACKET_SIZE";
    case CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT:    return "PREFERRED_PLATFORM_ATOMIC_ALIGNMENT";
    case CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT:      return "PREFERRED_GLOBAL_ATOMIC_ALIGNMENT";
    case CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT:       return "PREFERRED_LOCAL_ATOMIC_ALIGNMENT";
#endif
#ifdef CL_VERSION_2_1
    case CL_DEVICE_IL_VERSION:                             return "intermediate language standard version";
    case CL_DEVICE_MAX_NUM_SUB_GROUPS:                     return "maximum number of subgroups";
    case CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: return "subgroups independent forward progress requirement";
#endif
#ifdef CL_VERSION_3_0
    case CL_DEVICE_NUMERIC_VERSION:                        return "version (numeric)";
    case CL_DEVICE_EXTENSIONS_WITH_VERSION:                return "supported extensions are versioned";
    case CL_DEVICE_ILS_WITH_VERSION:                       return "intermediate language is versioned";
    case CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION:          return "built-in kernels are versioned";
    case CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES:             return "atomic memory acpabl";
    case CL_DEVICE_ATOMIC_FENCE_CAPABILITIES:              return "atomic fence capabilities";
    case CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT:         return "support for non-uniform workgroups";
    case CL_DEVICE_OPENCL_C_ALL_VERSIONS:                  return "support for all versions of OpenCL C";
    case CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:     return "preferred common divider of workgroup sizes";
    case CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT: return "support for workgroup-collective functions";
    case CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT:          return "support for generic addresses";
/* 0x106A to 0x106E - Reserved for upcoming KHR extension */
    case CL_DEVICE_OPENCL_C_FEATURES:                      return "OpenCL C feature";
    case CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES:            return "capabilities for enqueueing from device code";
    case CL_DEVICE_PIPE_SUPPORT:                           return "support for pipes";
    case CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED:      return "latency conformance version has passed";
#endif
    default: return nullptr;
    }
}

inline platform_t device_t::platform() const noexcept
{
    return platform::wrap(platform_handle_);
}

inline context_t device_t::create_context() const { return context::create(*this); }

namespace device {

inline device_t wrap(
    platform::handle_t platform_handle,
    handle_t   handle) noexcept
{
    return { platform_handle, handle };
}

namespace detail {

inline std::string& chomp_trailing_nul(std::string& str) noexcept
{
    if (str.length() > 0 and str[str.length() - 1] == '\0') {
        str.pop_back();
    }
    return str;
}

} // namespace detail

} // namespace device

inline device::capabilities_t device_t::capabilities() const noexcept
{
    return device::capabilities::wrap(handle_);
}

inline device::type_t device_t::type() const
{
    auto raw_returned_type = info::get_scalar<CL_DEVICE_TYPE>(handle_);
#ifndef NDEBUG
    if (raw_returned_type != CL_DEVICE_TYPE_CPU
        and raw_returned_type != CL_DEVICE_TYPE_GPU
        and raw_returned_type != CL_DEVICE_TYPE_ACCELERATOR
#ifdef CL_VERSION_1_2
        and raw_returned_type != CL_DEVICE_TYPE_CUSTOM
#endif
        ) {
        throw runtime_error(status::named_t::invalid_value, "clGetDeviceInfo", "Invalid/unknown device type returned");
    }
#endif
    return static_cast<device::type_t>(raw_returned_type);
}

inline std::string device_t::name() const
{
    return info::get_string<CL_DEVICE_NAME>(handle_);
}

inline vendor_t device_t::vendor() const
{
    auto id = info::get_scalar<CL_DEVICE_VENDOR_ID>(handle_);
#ifndef CL_VERSION_2_0
    return { id }
#else
    auto name = info::get_string<CL_DEVICE_VENDOR>(handle_);
    return { id, std::move(name) };
#endif
}

#ifdef CL_VERSION_1_2
// Note: We could, instead of using an optional, return nullptr when there is no parent
// device; but then the user would need to know about that fact, or at least remember to use
// some no_device named constant we would define. So, we'll go with the "vocabulary type".
inline optional<device_t> device_t::parent() const
{
    auto parent_handle = info::get_scalar<CL_DEVICE_PARENT_DEVICE>(handle_);
    return device::wrap(platform_handle_, parent_handle);
}

inline bool device_t::is_subdevice() const
{
    auto parent_handle = info::get_scalar<CL_DEVICE_PARENT_DEVICE>(handle_);
    return parent_handle != nullptr;
}

namespace device {
namespace detail {

inline count_t num_affinity_domain_parts(handle_t handle, affinity_domain_t domain)
{
    cl_device_partition_property properties[] = { CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, domain };
    count_t result;
    auto status = clCreateSubDevices(handle, properties, 0, nullptr, &result);
    throw_if_error_lazy(status, "clCreateSubDevices", "Getting the number of sub-devices which would "
        "be created if we were to partition " + opencl::detail::identify(handle)
        + " by affinity domain " + opencl::name_of(domain));
    return result;
}

inline count_t num_parts(handle_t handle, partition::spec_t const& spec)
{
    using pm = partition::mode_t;
    switch (spec.mode) {
    case pm::equally: return spec.subspec.num_parts;
    case pm::by_compute_unit_counts: return spec.subspec.compute_unit_counts.size();
    case pm::by_affinity_domain: return num_affinity_domain_parts(handle, spec.subspec.affinity_domain);
    default: throw std::invalid_argument("Unsupported partitioning mode");
    }
}

} // namespace detail
} // namespace device

inline dynarray<device_t> device_t::partition(device::partition::spec_t const& spec) const
{
    auto num_parts = device::detail::num_parts(handle_, spec);
    auto part_handles = make_dynarray<device::handle_t>(num_parts);
    auto properties_array = [&] {
        switch (spec.mode) {
        case device::partition::mode_t::equally:
            return to_dynarray( std::initializer_list<cl_device_partition_property>{ CL_DEVICE_PARTITION_EQUALLY, num_parts } );
        case device::partition::mode_t::by_counts:
            return generate_dynarray<cl_device_partition_property>(num_parts + 1,
                [&](size_t i) {
                    return i == 0 ? CL_DEVICE_PARTITION_BY_COUNTS : spec.subspec.compute_unit_counts[i-1];
                });
        case device::partition::mode_t::by_affinity_domain:
            return to_dynarray( std::initializer_list<cl_device_partition_property>{
                CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, spec.subspec.affinity_domain } );
        }
        throw std::invalid_argument("Unsupported partitioning mode");
    }();
    auto status = clCreateSubDevices(
        handle_, properties_array.data(), num_parts, part_handles.data(), nullptr);
    throw_if_error_lazy(status, "clCreateSubDevices", "Partitioning " + opencl::detail::identify(*this) + " into "
        + std::to_string(num_parts) + " sub-devices " + opencl::name_of(spec.mode));
    auto handle_to_device = [&](device::handle_t handle) { return device::wrap(platform_handle_, handle); };
    return to_dynarray<device_t>(part_handles, handle_to_device);
}
#endif // CL_VERSION_1_2


namespace device {

inline synchronized_timestamps_t get_synchronized_time_with(device_t const& device)
{
    synchronized_timestamps_t result;
    auto status = clGetDeviceAndHostTimer(device.handle(), &result.device, &result.host);
    throw_if_error_lazy(status, "clGetDeviceAndHostTimer", "");
    return result;
}

inline timestamp_t get_host_time_for(device_t const& device)
{
    timestamp_t result;
    auto status = clGetHostTimer(device.handle(), &result);
    throw_if_error_lazy(status, "clGetHostTimer", "");
    return result;
}

namespace detail {

inline index_t index_in_platform_of(handle_t handle, platform::handle_t platform_handle, cl_device_type raw_device_type)
{
    auto num_devices_ =  platform::detail::num_devices(platform_handle, raw_device_type);
    auto device_handles = platform::detail::get_device_handles(platform_handle, raw_device_type, num_devices_);
    auto iter = std::find(device_handles.begin(), device_handles.end(), handle);
    if (iter == device_handles.end()) {
        throw std::runtime_error("Could not find " + opencl::detail::identify(handle, opencl::detail::do_not_obtain_details)
            + " among the devices on " + opencl::detail::identify(platform_handle));
    }
    return iter - device_handles.begin();
}

} // namespace detail


} // namespace device

inline std::pair<context_t, context::device_t> device_t::in_self_context() const
{
    auto context = context::create(*this);
    auto device_in_context = context::device::wrap(platform_handle_, context.handle(), handle_);
    return { std::move(context), device_in_context };
}

inline device::index_t device_t::index_in_platform(device::type_t among_which) const
{
    return device::detail::index_in_platform_of(handle_, platform_handle_, static_cast<cl_device_type>(among_which));
}

inline device::index_t device_t::index_in_platform() const
{
    return device::detail::index_in_platform_of(handle_, platform_handle_, CL_DEVICE_TYPE_ALL);
}

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_DEVICE_HPP_
