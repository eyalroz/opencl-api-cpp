/**
 * @file
 *
 * @brief Definition of the @ref buffer_t class, and declarations of
 * its named constructor idioms and other related functions.
 */
#ifndef OPENCL_WRAPPERS_BUFFER_HPP_
#define OPENCL_WRAPPERS_BUFFER_HPP_

#include "types.hpp"
#include "util/region.hpp"
#include "memory_object.hpp"
#include "util/optional.hpp"

namespace opencl {

namespace buffer {

// Maybe just put this namespace inside memory? Or only use the memory namespace?
using memory::host_and_kernel_access_t;
using memory::full_access;

buffer_t wrap(
    platform::handle_t platform_handle,
    context::handle_t  context_handle,
    memory::handle_t   handle,
    bool               owning) noexcept;

buffer_t create(
    context_t const& context,
    size_t size,
    host_and_kernel_access_t access_spec = full_access() );

buffer_t create_using(
    context_t const& context,
    memory::region_t host_side_storage,
    host_and_kernel_access_t access_spec = full_access() );

buffer_t create_copy_of(
    context_t const& context,
    memory::region_t region_to_copy,
    host_and_kernel_access_t access_spec = full_access() );

template <typename T>
buffer_t create_using(
    context_t const& context,
    span<T> host_side_storage,
    host_and_kernel_access_t access_spec = full_access());
template <typename T>
buffer_t create_copy_of(context_t const& context,
    span<T> region_to_copy,
    host_and_kernel_access_t access_spec = full_access());

buffer_t create_sub_buffer(
    buffer_t const& parent,
    memory::subregion_spec_t subregion,
    host_and_kernel_access_t access_spec);

// TODO: Support creation with association to a set of devices via CL_MEM_DEVICE_HANDLE_LIST_KHR

} // namespace buffer

class buffer_t : public memory::object_t {
public:
    using parent_type = memory::object_t;
    using parent_type::handle_type;

protected:
    using parent_type::parent_type;

public:
    friend buffer_t buffer::wrap(
        platform::handle_t platform_handle,
        context::handle_t  context_handle,
        memory::handle_t   handle,
        bool               owning) noexcept;

    // TODO: Should I delete instead of move the rvalue-reference methods?

    /// @return The buffer from which this buffer was created as a sub-buffer - or nullopt
    /// if this buffer is not a sub-buffer of anything.
    optional<buffer_t> parent() const;

    buffer_t create_sub_buffer(
        memory::subregion_spec_t subregion,
        memory::host_and_kernel_access_t access_spec = memory::full_access() ) const;

    /// @return the subregion of the parent buffer within which a sub-buffer is associated -
    /// or the full buffer if there is no parent buffer, i.e. the region { 0, size_in_bytes() }
    memory::subregion_spec_t subregion() const;
}; // class buffer_t

std::pair<queue::event_t, memory::region_t>
map(buffer_t const& buffer, memory::mapping::spec_t const& spec, queue_t const& queue, bool blocking);

} // namespace opencl

#endif // OPENCL_WRAPPERS_BUFFER_HPP_
