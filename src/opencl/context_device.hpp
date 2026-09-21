/**
* @file
 *
 * @brief Definition of the @ref context::device_t class (not of its parent,
 * context-independent, @ref opencl::device_t), and a declaration of a named
 * constructor idiom for it.
 */
#ifndef OPENCL_WRAPPERS_CONTEXT_DEVICE_HPP_
#define OPENCL_WRAPPERS_CONTEXT_DEVICE_HPP_

#include "types.hpp"
#include "device.hpp"

namespace opencl {

namespace context {

namespace device {

device_t wrap(platform::handle_t, context::handle_t, handle_t) noexcept;

} // namespace device

// TODO: Consider also holding the device index within the context's devices,
// as a field
class device_t : public opencl::device_t {
public:
    using handle_type = device::handle_t;
    using parent_type = opencl::device_t;
    using index_type = device::index_t;

protected:
    context::handle_t context_handle_;

public:
    context::handle_t context_handle() const noexcept { return context_handle_; }
    context_t context() const;

protected:
    device_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        device::handle_t handle) noexcept
    : parent_type(platform_handle, handle), context_handle_(context_handle) { }

public:
    friend device_t device::wrap(platform::handle_t, context::handle_t, device::handle_t) noexcept;

    opencl::device_t const& as_platform_device() const noexcept { return *this; }

    queue_t create_queue() const;
    buffer_t create_buffer(size_t size, memory::host_and_kernel_access_t access_spec = memory::full_access()) const;
        // Should we even have this? Or - do we want a memory gadget? Or maybe only the context
        // creates buffers?


    /// @return the device's index among the devices  on its platform
    ///@{
    /// @p among_which the type of devices among which to compute the index
    index_type index_in_context() const;
    ///@}

}; // class device_t

// Note: A context-device will compare identical to the same device in another context

} // namespace context

} // namespace opencl

#endif // OPENCL_WRAPPERS_CONTEXT_DEVICE_HPP_
