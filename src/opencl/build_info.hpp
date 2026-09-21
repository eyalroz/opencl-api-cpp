/**
* @file
 *
 * @brief Definition of the @ref build_info_t class, declaration of named constructor
 * idioms for it, and some related functions.
 */
#ifndef OPENCL_WRAPPERS_BUILD_INFO_HPP_
#define OPENCL_WRAPPERS_BUILD_INFO_HPP_

#include "program.hpp"

namespace opencl {

namespace program {

namespace detail {

build_info_t wrap_build_info(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    handle_t            program_handle,
    device::handle_t    device_handle) noexcept;

} // namespace detail

// TODO: Should I explicitly take a boolean for success/failure?
class build_info_t {
protected:
    platform::handle_t platform_handle_;
    context::handle_t context_handle_;
    program::handle_t program_handle_;
    device::handle_t target_handle_;
    // Never owning!

protected:
    build_info_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        program::handle_t program_handle,
        device::handle_t target_handle)
    :
        platform_handle_(platform_handle),
        context_handle_(context_handle),
        program_handle_(program_handle),
        target_handle_(target_handle)
    {}

public:
    friend build_info_t detail::wrap_build_info(
        platform::handle_t  platform_handle,
        context::handle_t   context_handle,
        handle_t            program_handle,
        device::handle_t    device_handle) noexcept;

    platform::handle_t platform_handle() const noexcept { return platform_handle_; }
    context::handle_t context_handle() const noexcept { return context_handle_; }
    program::handle_t program_handle() const noexcept { return program_handle_; }
    device::handle_t target_handle() const noexcept { return target_handle_; }

    platform_t platform() const noexcept;
    context_t context() const noexcept;
    program_t program() const noexcept;
    context::device_t device() const noexcept;

    bool in_progress() const;
    build_status_t status() const;
    bool succeeded() const;
    // TODO: Implement parsing options from a string
    // build_options_t options() const;
    std::string marshalled_options() const;
    std::string log() const;
}; // class build_info_t

// Note: The build_info_t class is Not comparable


} // namespace program

} // namespace opencl

#endif // OPENCL_WRAPPERS_BUILD_INFO_HPP_
