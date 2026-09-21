/**
 * @file
 *
 * @brief Definition of the @ref device::capabilities_t class, and declaration
 * of a named constructor idiom for it.
 */
#ifndef OPENCL_WRAPPERS_COMPUTE_CAPABILITIES_HPP_
#define OPENCL_WRAPPERS_COMPUTE_CAPABILITIES_HPP_

#include "types.hpp"

namespace opencl {
namespace device {

namespace capabilities {

capabilities_t wrap(handle_t device_handle) noexcept;

} // namespace capabilities

// TODO: Refactor this, so that we can have "realized" capabilities
// that are just values, and the implicit capabilities of a device, which
// is the only thing we have right now
class capabilities_t {
protected:
    handle_t  device_handle_;

protected:
    capabilities_t(handle_t device_handle) noexcept : device_handle_(device_handle) { }

public:
    friend capabilities_t capabilities::wrap(handle_t device_handle) noexcept;

    nd_range::size_t num_compute_units() const;
    nd_range::dimension_index_t maximum_grid_dimensionality() const;
    // TODO: When switching to C++20, consider stealing types from mdspan extents
    nd_range::dimensions_t maximum_workgroup_dimensions() const;
    nd_range::size_t maximum_workgroup_size() const;
    template <typename E> size_t preferred_vector_width() const;
    template <typename E> size_t native_vector_width() const;
    clock_frequency_t maximum_clock_frequency() const;
    address_bit_count_t num_address_bits() const;
    size_t global_memory_size() const;
    size_t maximum_memory_allocation_size() const;
    size_t local_memory_size() const;
    version_t supported_opencl_version() const;
    std::string supported_opencl_version_string() const;
#if CL_VERSION_2_0
    size_t image_pitch_alignment() const;
    size_t image_base_address_alignment() const;
    kernel::parameter::index_t maximum_writeable_images_per_kernel() const;
    dynarray<cl_queue_properties> supported_queue_properties() const;
    size_t maximum_queue_size_in_bytes() const;
    size_t preferred_queue_size_in_bytes() const;
    size_t maximum_number_of_queues() const;
    size_t maximum_number_of_events() const;
    optional<size_t> preferred_total_size_of_global_variables() const;
    optional<size_t> maximum_total_size_of_global_variables() const;
    cl_device_svm_capabilities raw_svm_capabilities() const;
    kernel::parameter::index_t maximum_pipe_arguments() const;
    cl_uint maximum_active_pipe_reservations() const;
    cl_uint maximum_pipe_packet_size() const;
    // TODO: These names are super-long... perhaps we need sub-capabilities objects for categorization?
    std::size_t fine_grained_svm_atomic_type_preferred_alignment() const;
    std::size_t preferred_alignment_of_atomic_types_in_global_memory() const;
    std::size_t preferred_alignment_of_atomic_types_in_local_memory() const;
#endif // CL_VERSION_2_0
#if CL_VERSION_2_1
    /// @return A string with format <IL_Prefix>_<Major_Version>.<Minor_Version>
    /// @todo define a proper type for these versions...
    std::string intermediate_language_version() const;
    nd_range::dimension_t maximum_number_of_workgroup_subgroups() const;
    bool subgroups_can_make_independent_forward_progress() const;
#endif // CL_VERSION_2_1
#ifdef CL_VERSION_3_0
    dynarray<name_and_version_t> supported_extensions() const;
    dynarray<name_and_version_t> supported_intermediate_languages() const;
    dynarray<name_and_version_t> builtin_kernels() const;
    cl_device_atomic_capabilities raw_atomic_memory_capabilities() const;
    cl_device_atomic_capabilities raw_atomic_fence_capabilities() const;
    bool support_grids_with_global_work_dims_indivisible_by_local() const;
    dynarray<version::named_t> all_supported_opencl_versions() const;
    nd_range::dimension_t preferred_workgroup_size_quantum() const;
    bool collective_functions_support() const;
    bool generic_address_space_support() const;
#endif

    /// Returns the number of parts, which a partition of the device by the
    /// specified affinity domain, would yield.
    count_t prospective_partition_size_for(affinity_domain_t domain) const;

    template <cl_uint InfoParameter>
    info::parameter_value_t<InfoParameter> get_raw_parameter() const;
}; // class capabilities_t

} // namespace device
} // namespace opencl

#endif // OPENCL_WRAPPERS_COMPUTE_CAPABILITIES_HPP_
