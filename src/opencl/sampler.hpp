/**
 * @file
 *
 * @brief Definition of the @ref sampler_t wrapper class - used in OpenCL kernel
 * to access image objects - and related named constructor idios for it.
 */
#ifndef OPENCL_WRAPPERS_SAMPLER_HPP_
#define OPENCL_WRAPPERS_SAMPLER_HPP_

#include "types.hpp"

namespace opencl {
namespace image {
namespace sampler {

namespace detail {

handle_t create(
    context::handle_t context_handle,
    bool normalized_coordinates,
    addressing_mode_t addressing_mode,
    filtering_mode_t filtering_mode);

} // namespace detail

sampler_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    sampler::handle_t   handle,
    bool                owning) noexcept;

sampler_t create(
    context_t const&  context,
    bool              normalized_coordinates,
    addressing_mode_t addressing_mode,
    filtering_mode_t  filtering_mode);

} // namespace sampler

class sampler_t {
public:
    using handle_type = sampler::handle_t;

    friend sampler_t sampler::wrap(
        platform::handle_t  platform_handle,
        context::handle_t   context_handle,
        sampler::handle_t   handle,
        bool                owning) noexcept;

protected:
    platform::handle_t platform_handle_;
    context::handle_t context_handle_;
    handle_type handle_;
    detail::handle_ownership_token_t<sampler_t> ownership_token_;

    sampler_t(
        platform::handle_t  platform_handle,
        context::handle_t   context_handle,
        handle_type         handle,
        bool                owning) noexcept
    : platform_handle_(platform_handle), context_handle_(context_handle), handle_(handle), ownership_token_(owning, handle) { }

public:
    platform::handle_t platform_handle() const noexcept { return platform_handle_; }
    context::handle_t context_handle() const noexcept { return context_handle_; }
    handle_type handle() const noexcept { return handle_; }
    bool owning() const noexcept { return ownership_token_.owning(); }

    platform_t platform() const noexcept;
    context_t context() const noexcept;

    // methods based on getinfo
    filtering_mode_t filtering_mode() const;
    addressing_mode_t addressing_mode() const;
    bool normalized_coordinates() const;

    // TODO: methods for the MIP filter mode and LOD_MIN, LOD_MAX
}; // class sampler_t

} // namespace image
} // namespace opencl

#endif // OPENCL_WRAPPERS_SAMPLER_HPP_
