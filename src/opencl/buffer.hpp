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

/**
 * Wrap a raw OpenCL API buffer handle in a @ref buffer_t object, possibly specifying
 * its context and the platform it exists on.
 *
 * @ingroup nci_wrap
 *
 * @param[in] platform_handle The OpenCL platform on which the context and the buffer exist
 * @param[in] context_handle The OpenCL context in which the buffer was created
 * @param[in] handle this _must_ be a valid (non-released) handle of an OpenCL buffer, even
 *     though its type is common with the handles of all other OpenCL memory objects.
 * @param[in] owning if true, the created @ref buffer_t will own the handle, and will release
 *     have the OpenCL driver release the buffer on destruction. Otherwise, the returned
 *     @ref buffer_t will be a non-owning reference-type objects - for convenient use, but
 *     which will do nothing on destruction; and it will assume the handle'svalidity outlives
 *     it.
 */
buffer_t wrap(
    platform::handle_t platform_handle,
    context::handle_t  context_handle,
    memory::handle_t   handle,
    bool               owning) noexcept;

/// @ingroup nci_from_handle
buffer_t from_handle(handle_t handle, bool owning = false);

/**
 *  @param[in] context the context in which this buffer will be defined (and on whose devices it
 *      might use).
 *  @param[in] access_spec where and how should it be possible to access the new buffer?
 *
 *  @ingroup nci_create
 */
///@{
/**
 *  Create a new OpenCL buffer with no relation to pre-existing  memory allocations and memory objects.
 *
 *  @param[in] context the context in which this buffer will be defined (and on whose devices it might
 *  used).
 *  @param[in] size the new buffer's desired size in bytes
 *
 *  @ingroup nci_create
 */
buffer_t create(
    context_t const& context,
    size_t size,
    host_and_kernel_access_t access_spec = full_access() );

/**
 *  Create a new OpenCL buffer which uses an existing host-side memory region
 *  (rather than allocating its own host-side storage).
 *
 *  @param[in] host_side_storage A region of host-side memory which the buffer
 *      should use for storage. Also defines the new buffer's size.
 *
 */
///@{
buffer_t create_using(
    context_t const& context,
    memory::region_t host_side_storage,
    host_and_kernel_access_t access_spec = full_access() );

template <typename T>
buffer_t create_using(
    context_t const& context,
    span<T> host_side_storage,
    host_and_kernel_access_t access_spec = full_access());
///@}

/**
 *  Create a new OpenCL buffer which initializes its storage to a copy of
 *  some existing region of host-side memory.
 *
 *  @param[in] region_to_copy A region to which the buffer's contents will
 *      initially be identical to. Also, defines the new buffer's size.
 */
///@{
buffer_t create_copy_of(
    context_t const& context,
    memory::region_t region_to_copy,
    host_and_kernel_access_t access_spec = full_access() );

template <typename T>
buffer_t create_copy_of(
    context_t const& context,
    span<T> region_to_copy,
    host_and_kernel_access_t access_spec = full_access());
///@}
///@}

/**
 *  Arrange for a subregion of an existing OpenCL buffer to be usable as a buffer
 *  of its own.
 *
 *  @param[in] parent The larger buffer within which the subregion of interest is
 *      located. Also indicates the containing context.
 *  @param[in] subregion A region to which the buffer's contents will
 *      initially be identical to. Also, defines the new buffer's size.
 *  @param[in] access_spec where and how should it be possible to access the new buffer?
 */
buffer_t create_sub_buffer(
    buffer_t const& parent,
    memory::subregion_spec_t subregion,
    host_and_kernel_access_t access_spec = full_access());

// TODO: Support creation with association to a set of devices via CL_MEM_DEVICE_HANDLE_LIST_KHR

} // namespace buffer

/**
 * The main kind of memory object used for storing and reading data on compute devices; see also
 * {@url https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html#_buffer_objects}.
 *
 * The data in a buffer is plain bytes (octets), untyped, and contiguous. It may have more than
 * one location of physical storage, e.g. storage in host-side memory and in device-side memory,
 * on various devices; with OpenCL taking care the relevant location is up-to-date with the
 * effect of earlier operations, before passing an address to its compute-device storage region,
 * to a kernel.
 *
 * @note A buffer's host-side storage may be allocated by OpenCL itself, or provided by the user on
 * construction.
 */
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

/**
 * Arrange it so that changes to the buffer are reflected in a region of host memory, or
 * vice-versa, that changes to a region of host memory apply to the OpenCL buffer as well.
 *
 * @note this function is similar to the @ref mmap for files: Changes to some region of
 * memory translate into changes to a file (or vice-versa).
 *
 * @param buffer the buffer to map
 * @param spec specification of what kind of mapping to perform: what part of the buffer, and
 *    whether to map for reading buffer contents via the memory region, or for having changes
 *    to the region apply to the buffer as well (mapping for writing). It is also possible to
 *    specify a request to unmap a previously-mapped buffer.
 * @param queue
 * @param blocking
 * @return the mapped region, preceded by an event whose completion guaranbtees the mapped
 * region being ready for use.
 */
std::pair<queue::event_t, memory::region_t>
map(buffer_t const& buffer, memory::mapping::spec_t const& spec, queue_t const& queue, bool blocking);

} // namespace opencl

#endif // OPENCL_WRAPPERS_BUFFER_HPP_
