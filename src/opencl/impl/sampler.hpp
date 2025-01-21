
#ifndef OPENCL_WRAPPERS_IMPL_SAMPLER_HPP_
#define OPENCL_WRAPPERS_IMPL_SAMPLER_HPP_

#include "../sampler.hpp"
#include "identify.hpp"

namespace opencl {

inline char const* info::traits_t<info::sampler>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case CL_SAMPLER_REFERENCE_COUNT:                  return "reference count";
    case CL_SAMPLER_CONTEXT:                          return "context handle";
    case CL_SAMPLER_NORMALIZED_COORDS:                return "normalized coordinates flag";
    case CL_SAMPLER_ADDRESSING_MODE:                  return "addressing mode";
    case CL_SAMPLER_FILTER_MODE:                      return "filtering mode";
#ifdef CL_VERSION_2_0
    case CL_SAMPLER_MIP_FILTER_MODE:                  return "MIP filter mode";
    case CL_SAMPLER_LOD_MIN:                          return "LOD max";
    case CL_SAMPLER_LOD_MAX:                          return "LOD min";
#endif
#ifdef CL_VERSION_3_0
    case CL_SAMPLER_PROPERTIES:                      return "complete sampler properties structure";
#endif
    default: return nullptr;
    }
}

namespace detail {

OCLW_DEFINE_HANDLE_TRAITS("image sampler", image::sampler::handle_t, clReleaseSampler, clRetainSampler);

} // namespace detail

namespace image {
namespace sampler {
namespace detail {

inline handle_t create(
    context::handle_t context_handle,
    bool normalized_coordinates,
    addressing_mode_t addressing_mode,
    filtering_mode_t filtering_mode)
{
    // TODO: Use the with-properties version with OpenCL 2.0 and later
    status_t status;
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    auto handle = clCreateSampler(
        context_handle,
        normalized_coordinates,
        opencl::detail::to_underlying(addressing_mode),
        opencl::detail::to_underlying(filtering_mode),
        &status);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    // TODO: Have the argument values in the string
    throw_if_error_lazy(status, "clCreateSampler", "");
    return handle;
}

} // namespace detail


inline sampler_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    sampler::handle_t   handle,
    bool                owning) noexcept
{
    return sampler_t{ platform_handle, context_handle, handle, owning };
}

inline sampler_t create(
    context_t const&  context,
    bool              normalized_coordinates,
    addressing_mode_t addressing_mode,
    filtering_mode_t  filtering_mode)
{
    auto handle = detail::create(context.handle(), normalized_coordinates, addressing_mode, filtering_mode);
    return wrap(context.platform_handle(), context.handle(), handle, is_owning);
}

} // namespace sampler

inline platform_t sampler_t::platform() const noexcept
{
    return platform::wrap(platform_handle_);
}

inline context_t sampler_t::context() const noexcept
{
    return context::detail::by_handles(platform_handle_, context_handle_, is_not_owning);
}

// methods based on getinfo
inline filtering_mode_t sampler_t::filtering_mode() const
{
    return static_cast<filtering_mode_t>(info::get_scalar<CL_SAMPLER_FILTER_MODE>(handle_));
}

inline addressing_mode_t sampler_t::addressing_mode() const
{
    return static_cast<addressing_mode_t>(info::get_scalar<CL_SAMPLER_ADDRESSING_MODE>(handle_));
}

inline bool sampler_t::normalized_coordinates() const
{
    return info::get_scalar<CL_SAMPLER_NORMALIZED_COORDS>(handle_);
}

} // namespace image

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_SAMPLER_HPP_
