
#ifndef OPENCL_WRAPPERS_IMPL_CAPABILITIES_HPP_
#define OPENCL_WRAPPERS_IMPL_CAPABILITIES_HPP_

#include "../device_capabilities.hpp"
#include "device.hpp"
#include "../version.hpp"

namespace opencl {

namespace device {

namespace capabilities {

inline capabilities_t wrap(handle_t device_handle) noexcept
{
    return capabilities_t{ device_handle };
}

} // namespace capabilities

inline nd_range::dimension_index_t capabilities_t::maximum_grid_dimensionality() const
{
    return info::get_scalar<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(device_handle_);
}

// TODO: Subdevice partitioning capabilities

inline nd_range::dimensions_t capabilities_t::maximum_workgroup_dimensions() const
{
    // Note: The number of returned elements should be the max_dimensionality
    auto raw = info::get_array<CL_DEVICE_MAX_WORK_ITEM_SIZES>(device_handle_);
    if (raw.size() > nd_range::max_dimensions) {
        throw std::logic_error("Workgroup dimensionality of "
            + opencl::detail::identify(device_handle_) + " exceeds maximum supported dimensionality; rebuild your program.");
    }
    return to_static_dynarray<nd_range::max_dimensions>(raw);
}

inline nd_range::size_t capabilities_t::maximum_workgroup_size() const
{
    return info::get_scalar<CL_DEVICE_MAX_WORK_GROUP_SIZE>(device_handle_);
}

namespace detail {

template <typename> struct vector_width_device_cap_attributes {};
#define OCLW_VECTOR_WIDTH_ATTRS(tp_lc_, tp_uc_) \
template <> struct vector_width_device_cap_attributes<tp_lc_> { \
enum { \
    preferred = OCLW_EXPAND_THEN_CONCATENATE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_, tp_uc_), \
    native  = OCLW_EXPAND_THEN_CONCATENATE(CL_DEVICE_NATIVE_VECTOR_WIDTH_, tp_uc_), \
}; \
}

OCLW_VECTOR_WIDTH_ATTRS(char, CHAR);
OCLW_VECTOR_WIDTH_ATTRS(short, SHORT);
OCLW_VECTOR_WIDTH_ATTRS(int, INT);
OCLW_VECTOR_WIDTH_ATTRS(long, LONG);
OCLW_VECTOR_WIDTH_ATTRS(float, FLOAT);
OCLW_VECTOR_WIDTH_ATTRS(double, DOUBLE);

#undef OCLW_VECTOR_WIDTH_ATTRS

} // namespace detail

template <typename E>
size_t capabilities_t::preferred_vector_width() const
{
    enum { scalar_code = detail::vector_width_device_cap_attributes<E>::preferred };
    return info::get_scalar<scalar_code>(device_handle_);
}

template <typename E>
size_t capabilities_t::native_vector_width() const
{
    enum { scalar_code = detail::vector_width_device_cap_attributes<E>::native };
    return info::get_scalar<scalar_code>(device_handle_);
}

inline clock_frequency_t capabilities_t::maximum_clock_frequency() const
{
    return info::get_scalar<CL_DEVICE_MAX_CLOCK_FREQUENCY>(device_handle_);
}

inline address_bit_count_t capabilities_t::num_address_bits() const
{
    return info::get_scalar<CL_DEVICE_ADDRESS_BITS>(device_handle_);
}

inline size_t capabilities_t::global_memory_size() const
{
    return info::get_scalar<CL_DEVICE_GLOBAL_MEM_SIZE>(device_handle_);
}

inline size_t capabilities_t::maximum_memory_allocation_size() const
{
    return info::get_scalar<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(device_handle_);
}

inline size_t capabilities_t::local_memory_size() const
{
    return info::get_scalar<CL_DEVICE_LOCAL_MEM_SIZE>(device_handle_);
}

inline size_t capabilities_t::num_compute_units() const
{
    return info::get_scalar<CL_DEVICE_MAX_COMPUTE_UNITS>(device_handle_);
}

inline version_t capabilities_t::supported_opencl_version() const
{
#if CL_VERSION_3_0
    return make_version(supported_opencl_version_string());
#else
    return info::get_array<CL_DEVICE_NUMERIC_VERSION>(device_handle_);
#endif
    return make_version(info::get_array<CL_DEVICE_NUMERIC_VERSION>(device_handle_));
}

inline std::string capabilities_t::supported_opencl_version_string() const
{
    return info::get_string<CL_DEVICE_VERSION>(device_handle_);
}

#if CL_VERSION_2_0
inline size_t capabilities_t::image_pitch_alignment() const
{
    return info::get_scalar<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>(device_handle_);
}

inline size_t capabilities_t::image_base_address_alignment() const
{
    return info::get_scalar<CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT>(device_handle_);
}

inline kernel::parameter::index_t capabilities_t::maximum_writeable_images_per_kernel() const
{
    return info::get_scalar<CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS>(device_handle_);
}

inline dynarray<cl_queue_properties> capabilities_t::supported_queue_properties() const
{
    return info::get_array<CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES>(device_handle_);
}

inline size_t capabilities_t::maximum_queue_size_in_bytes() const
{
    return info::get_scalar<CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE>(device_handle_);
}

inline size_t capabilities_t::preferred_queue_size_in_bytes() const
{
    return info::get_scalar<CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE>(device_handle_);
}

inline size_t capabilities_t::maximum_number_of_queues() const
{
    return info::get_scalar<CL_DEVICE_MAX_ON_DEVICE_QUEUES>(device_handle_);
}

inline size_t capabilities_t::maximum_number_of_events() const
{
    return info::get_scalar<CL_DEVICE_MAX_ON_DEVICE_EVENTS>(device_handle_);
}

inline optional<size_t> capabilities_t::preferred_total_size_of_global_variables() const
{
    auto raw_result = info::get_scalar<CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE>(device_handle_);
    if (raw_result == 0) { return nullopt; }
    return raw_result;
}

inline optional<size_t> capabilities_t::maximum_total_size_of_global_variables() const
{
    auto raw_result = info::get_scalar<CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE>(device_handle_);
    if (raw_result == 0) { return nullopt; }
    return raw_result;
}

inline cl_device_svm_capabilities capabilities_t::raw_svm_capabilities() const
{
    return info::get_scalar<CL_DEVICE_SVM_CAPABILITIES>(device_handle_);
}

inline kernel::parameter::index_t capabilities_t::maximum_pipe_arguments() const
{
    return info::get_scalar<CL_DEVICE_MAX_PIPE_ARGS>(device_handle_);
}

inline cl_uint capabilities_t::maximum_active_pipe_reservations() const
{
    return info::get_scalar<CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS>(device_handle_);
}

inline cl_uint capabilities_t::maximum_pipe_packet_size() const
{
    return info::get_scalar<CL_DEVICE_PIPE_MAX_PACKET_SIZE>(device_handle_);
}

inline size_t capabilities_t::fine_grained_svm_atomic_type_preferred_alignment() const
{
    return info::get_scalar<CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT>(device_handle_);
}

inline std::size_t capabilities_t::preferred_alignment_of_atomic_types_in_global_memory() const
{
    return info::get_scalar<CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT>(device_handle_);
}

inline std::size_t capabilities_t::preferred_alignment_of_atomic_types_in_local_memory() const
{
    return info::get_scalar<CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT>(device_handle_);
}
#endif // CL_VERSION_2_0

#if CL_VERSION_2_1
inline std::string capabilities_t::intermediate_language_version() const
{
    return info::get_string<CL_DEVICE_IL_VERSION>(device_handle_);
}

inline nd_range::dimension_t capabilities_t::maximum_number_of_workgroup_subgroups() const
{
    return info::get_scalar<CL_DEVICE_MAX_NUM_SUB_GROUPS>(device_handle_);
}

inline bool capabilities_t::subgroups_can_make_independent_forward_progress() const
{
    return info::get_scalar<CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS>(device_handle_);
}
#endif // CL_VERSION_2_1

inline count_t capabilities_t::prospective_partition_size_for(affinity_domain_t domain) const
{
    return detail::num_affinity_domain_parts(device_handle_, domain);
}

#if CL_VERSION_3_0
inline dynarray<version::named_t> capabilities_t::supported_extensions() const
{
    auto raw_arr = info::get_array<CL_DEVICE_EXTENSIONS_WITH_VERSION>(device_handle_);
    return to_dynarray(raw_arr, make_named_version);
}
inline dynarray<version::named_t> capabilities_t::supported_intermediate_languages() const
{
    auto raw_arr = info::get_array<CL_DEVICE_ILS_WITH_VERSION>(device_handle_);
    return to_dynarray(raw_arr, make_named_version);
}

inline dynarray<version::named_t> capabilities_t::builtin_kernels() const
{
    auto raw_arr = info::get_array<CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION>(device_handle_);
    return to_dynarray(raw_arr, make_named_version);
}

inline cl_device_atomic_capabilities capabilities_t::raw_atomic_memory_capabilities() const
{
    return info::get_scalar<CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES>(device_handle_);
}

inline cl_device_atomic_capabilities capabilities_t::raw_atomic_fence_capabilities() const
{
    return info::get_scalar<CL_DEVICE_ATOMIC_FENCE_CAPABILITIES>(device_handle_);
}

inline bool capabilities_t::support_grids_with_global_work_dims_indivisible_by_local() const
{
    return info::get_scalar<CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT>(device_handle_);
}

inline dynarray<version::named_t> capabilities_t::all_supported_opencl_versions() const
{
    auto raw_arr = info::get_array<CL_DEVICE_OPENCL_C_ALL_VERSIONS>(device_handle_);
    return to_dynarray(raw_arr, make_named_version);
}

inline nd_range::dimension_t capabilities_t::preferred_workgroup_size_quantum() const
{
    return info::get_scalar<CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device_handle_);
}

inline bool capabilities_t::collective_functions_support() const
{
    return info::get_scalar<CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT>(device_handle_);
}

inline bool capabilities_t::generic_address_space_support() const
{
    return info::get_scalar<CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT>(device_handle_);
}
#endif

template <cl_device_info InfoParameter>
info::parameter_value_t<InfoParameter> capabilities_t::get_raw_parameter() const
{
    using value_type = info::parameter_value_t<InfoParameter>;
    static_assert(not std::is_pointer<value_type>::value, "Obtaining arrays via get_raw_parameter is not yet supported");
    return info::get_scalar<InfoParameter>(device_handle_);
}

} // namespace device

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_CAPABILITIES_HPP_
