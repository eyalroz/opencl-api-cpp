
#ifndef OPENCL_WRAPPERS_IMPL_CONTEXT_DEVICE_HPP_
#define OPENCL_WRAPPERS_IMPL_CONTEXT_DEVICE_HPP_

#include "../context_device.hpp"
#include "../identify.hpp"

namespace opencl {

namespace context {

namespace device {

inline device_t wrap(
    platform::handle_t platform_handle,
    context::handle_t context_handle,
    handle_t handle) noexcept
{
    return { platform_handle, context_handle, handle };
}

} // namespace device

inline queue_t device_t::create_queue() const
{
    status_t status { CL_SUCCESS };
#ifdef CL_VERSION_2_0
    auto queue_handle = clCreateCommandQueueWithProperties(context_handle_, handle_, nullptr, &status);
    static constexpr auto api_function_name = "clCreateCommandQueueWithProperties";
#else
    cl_command_queue_properties properties { };
    static constexpr auto api_function_name = "clCreateCommandQueue";
    auto queue_handle = clCreateCommandQueue(context_handle_, handle_, nullptr, &status);
#endif
    throw_if_error_lazy(status, api_function_name, "Creating a command queue on " + opencl::detail::identify(*this) );
    return queue::wrap(platform_handle_, context_handle_, handle_, queue_handle, is_owning);
}

inline buffer_t device_t::create_buffer(size_t size, memory::host_and_kernel_access_t access_spec) const
{
    return buffer::create(context(), size, access_spec);
}

inline context_t device_t::context() const
{
    auto device_handles = detail::get_device_handles(context_handle_);
    return wrap(platform_handle_, std::move(device_handles), context_handle_, is_owning);
}

inline device::index_t device_t::index_in_context() const
{
    auto context_ = context();
    auto device_handles = context_.device_handles();
    auto iter = std::find(device_handles.begin(), device_handles.end(), handle_);
    if (iter == device_handles.end()) {
        throw std::runtime_error("Could not find "
            + opencl::detail::identify(handle_, opencl::detail::do_not_obtain_details)
            + " among the devices on " + opencl::detail::identify(context_));
    }
    return iter - device_handles.begin();
}

} // namespace context

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_CONTEXT_DEVICE_HPP_
