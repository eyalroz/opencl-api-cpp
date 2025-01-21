
#ifndef OPENCL_WRAPPERS_IMPL_PIPE_HPP_
#define OPENCL_WRAPPERS_IMPL_PIPE_HPP_

#if CL_VERSION_2_0

#include "memory_object.hpp"
#include "../pipe.hpp"
#include "../context.hpp"
#include "../identify.hpp"

 namespace opencl {

namespace pipe {

namespace detail {

inline pipe_t create(
    platform::handle_t platform_handle,
    context::handle_t context_handle,
    packet_size_t packet_size,
    capacity_t capacity_in_packets,
    memory::host_and_kernel_access_t access_spec)
{
    cl_mem_flags flags = memory::detail::make_flags(access_spec);
    status_t status;
    cl_pipe_properties * no_pipe_properties { nullptr };
    handle_t new_pipe_handle = clCreatePipe(
        context_handle,
        flags,
        packet_size,
        capacity_in_packets,
        no_pipe_properties, // OpenCL 3.0 mandates this
        &status);
    if (not is_success(status)) {
        std::string extra_message = [&]() {
            switch (status) {
            case CL_INVALID_OPERATION:
                return "No devices in this context support pipes, i.e.";
            case CL_INVALID_VALUE:
                return "Invalid kernel-and-host memory access flags, i.e.";
            default: return "";
            }
        }();
        throw runtime_error(status, "clCreatePipe",
            "Failed creating a pipe in " + opencl::detail::identify(context_handle) +
            (extra_message.empty() ? "" : ": ") + extra_message);
    }

    return wrap(platform_handle, context_handle, new_pipe_handle, is_owning);
}

} // namespace detail

inline pipe_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    handle_t            handle,
    bool                owning) noexcept
{
    return pipe_t { platform_handle, context_handle, handle, owning };
}

inline pipe_t create(
    context_t const& context,
    packet_size_t packet_size,
    capacity_t capacity_in_packets,
    memory::host_and_kernel_access_t access_spec)
{
    return detail::create(context.platform_handle(), context.handle(), packet_size, capacity_in_packets, access_spec);
}

} // namespace pipe

inline pipe::packet_size_t pipe_t::packet_size() const
{
    return info::get_scalar<CL_PIPE_PACKET_SIZE>(handle());
}

inline size_t pipe_t::capacity_in_packets() const
{
    return info::get_scalar<CL_PIPE_MAX_PACKETS>(handle());
}

template <> inline pipe_t upcast<pipe_t>(memory::object_t const& memory_object)
{
    if (memory::detail::is_pipe(memory_object)) {
        return pipe::wrap(
            memory_object.platform_handle(), memory_object.context_handle(), memory_object.handle(), is_not_owning);
    }
    throw std::invalid_argument("Attempt to upcast a memory object, which is not an image, into an image_t");
}

} // namespace opencl

#endif // CL_VERSION_2_0

#endif // OPENCL_WRAPPERS_IMPL_PIPE_HPP_
