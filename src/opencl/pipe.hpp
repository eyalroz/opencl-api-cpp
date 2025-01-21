#ifndef OPENCL_WRAPPERS_PIPE_HPP_
#define OPENCL_WRAPPERS_PIPE_HPP_

#include "types.hpp"

namespace opencl {

namespace pipe {

pipe_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    handle_t            handle,
    bool                owning) noexcept;

pipe_t create(
    context_t const& context,
    packet_size_t packet_size,
    capacity_t capacity_in_packets,
    memory::host_and_kernel_access_t access_spec = memory::full_access());
    // No "pipe properties - as of OpenCL 3.0 these are always null, so we ignore them"

} // namespace pipe

// @note: For in-kernel functions involving pipes see: @url https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/pipeFunctions.html
class pipe_t : public memory::object_t {
public:
    using parent_type = memory::object_t;

protected:
    using parent_type::parent_type;

public:
    friend pipe_t pipe::wrap(
        platform::handle_t platform_handle,
        context::handle_t   context_handle,
        pipe::handle_t      handle,
        bool                owning) noexcept; // Can't I inline it?

    // TODO: Should I delete instead of move the rvalue-reference methods?

    pipe_t& operator= (pipe_t const&) = delete;
    pipe_t& operator= (pipe_t&&) = default;
    pipe_t(pipe_t const& other) = delete;
    pipe_t(pipe_t&& other) noexcept : parent_type(std::move(other)) {}

    // methods based on getinfo
    pipe::packet_size_t packet_size() const;
    size_t capacity_in_packets() const;

    // No support for now for retrieving pipe properties; they are necessarily unused as of OpenCL 3.0

}; // class pipe_t

} // namespace opencl

#endif // OPENCL_WRAPPERS_PIPE_HPP_
