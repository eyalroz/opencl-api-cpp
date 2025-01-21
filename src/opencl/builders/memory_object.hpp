
#ifndef OPENCL_WRAPPERS_BUILDERS_MEMORY_OBJECT_HPP_
#define OPENCL_WRAPPERS_BUILDERS_MEMORY_OBJECT_HPP_

// TODO: Add missing support for
// CL_MEM_IMMUTABLE_EXT               if cl_ext_immutable_memory_objects
// CL_MEM_DEVICE_HANDLE_LIST_KHR      if cl_khr_external_memory,
// CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT  if cl_ext_buffer_device_address

#include "../types.hpp"

namespace opencl {

// TODO: Perhaps move this into a namespace and use a non-class enum?
enum class side_t { host, device, kernel = device };

namespace builders {

namespace detail {

// A CRTP base class for building actual memory object
template <typename Builder>
class memory_object_t {
protected:
    using flags_type = memory::flags_t;
    using builder_type = Builder;

    platform::handle_t platform_handle_ = nullptr;
    context::handle_t context_handle_ = nullptr;
    flags_type flags_ { 0u };

    // Note: If the flags indicate we need to copy - we'll use the const ptr; if they indicate
    // we need to use a specified host storage - we'll use the non-const ptr. This redundancy
    // in storage avoids the need to remember whether we const-cast at some point or not.
    void * host_ptr_ { nullptr };
    void const * copy_host_ptr_ { nullptr };
    size_t size_ { 0 };

    void lower_flags(flags_type flags_to_lower) { flags_ &= ~flags_to_lower; }
    void raise_flags(flags_type flags_to_raise) { flags_ |= flags_to_raise; }
    void clear_flags() { flags_ = 0; }
    flags_type flags() const { return flags_; }
    bool has_flags(flags_type flags) const { return flags_ & flags; }
public:
    builder_type& context(context_t const& context);
    builder_type& enable_device_access(access_kind_t access_kind);
    builder_type& disable_device_access(access_kind_t access_kind);
    builder_type& device_access(access_kind_t access_kind);
    builder_type& no_device_access();
#ifdef CL_VERSION_1_2
    builder_type& enable_host_access(access_kind_t access_kind);
    builder_type& disable_host_access(access_kind_t access_kind);
    builder_type& host_access(access_kind_t access_kind);
    builder_type& no_host_access();
    builder_type& enable_access(side_t side, access_kind_t access_kind);
    builder_type& disable_access(side_t side, access_kind_t access_kind);
    builder_type& access(side_t side, access_kind_t access_kind);
#endif
    builder_type& require_allocation(size_t size);
    builder_type& host_side_storage(memory::region_t region);
    builder_type& host_region_to_copy(memory::const_region_t region);
    template <typename Container>
    builder_type& host_region_to_copy(Container const & container);
    builder_type& host_region_to_copy(void const * ptr, size_t size);
    builder_type& host_region(memory::region_t region);
    template <typename Container>
    builder_type& host_region(Container const & container);
    builder_type& host_region_start(void * ptr, size_t size);
    builder_type& no_accessible_host_storage();

    /// Validate the parameters we have set so far are valid and consistent,
    /// and possibly sufficient for creating a buffer
    virtual void validate(bool ensure_sufficiency = false) const noexcept(false) = 0;

    memory::host_and_kernel_access_t get_access_spec() const;
    virtual ~memory_object_t() noexcept = default;
}; // memory_object_t

} // namespace detail

} // namespace builders

} // namespace opencl

#endif // OPENCL_WRAPPERS_BUILDERS_MEMORY_OBJECT_HPP_
