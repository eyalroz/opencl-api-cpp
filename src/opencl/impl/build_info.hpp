
#ifndef OPENCL_WRAPPERS_IMPL_BUILD_INFO_HPP_
#define OPENCL_WRAPPERS_IMPL_BUILD_INFO_HPP_

#include "program.hpp"

#include "../build_info.hpp"

namespace opencl {

inline char const* info::traits_t<info::program_build>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case  CL_PROGRAM_BUILD_STATUS:                     return "build status";
    case  CL_PROGRAM_BUILD_OPTIONS:                    return "build options";
    case  CL_PROGRAM_BUILD_LOG:                        return "build log";
#ifdef CL_VERSION_1_2
    case  CL_PROGRAM_BINARY_TYPE:                      return "binary type";
#endif
#ifdef CL_VERSION_2_0
    case  CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE: return "total size of global variables";
#endif
    default: return nullptr;
    }
}

namespace program {

namespace detail {

inline optional<build_status_t> maybe_build_status_for(program_t const& program, device::handle_t target_handle)
{
    auto result = info::get_scalar_nothrow<CL_PROGRAM_BUILD_STATUS>({program.handle(), target_handle});
    if (is_success(result)) return static_cast<build_status_t>(*result);
    return nullopt;
}

inline optional<device::handle_t> first_build_failure_target(program_t const& program)
{
    auto target_handles = get_default_target_handles(program);
    for (auto target_handle : target_handles) {
        auto maybe_build_status = maybe_build_status_for(program, target_handle);
        if (maybe_build_status and *maybe_build_status == build_status_t::error) {
            return target_handle;
        }
    }
    return nullopt;
}

} // namespace detail

inline build_status_t build_info_t::status() const
{
    auto raw_result = info::get_scalar<CL_PROGRAM_BUILD_STATUS>({program_handle_, target_handle_});
    return static_cast<build_status_t>(raw_result);
}

inline bool build_info_t::succeeded() const
{
    return is_success(status());
}

inline bool build_info_t::in_progress() const
{
    return status() == build_status_t::in_progress;
}

inline std::string build_info_t::log() const
{
    return info::get_string<CL_PROGRAM_BUILD_LOG>({program_handle_, target_handle_});
}

inline std::string build_info_t::marshalled_options() const
{
    return info::get_string<CL_PROGRAM_BUILD_OPTIONS>({program_handle_, target_handle_});
}

inline platform_t build_info_t::platform() const noexcept
{
    return platform::wrap(platform_handle_);
}

inline context_t build_info_t::context() const noexcept
{
    return context::detail::by_handles(platform_handle_, context_handle_, is_not_owning);
}

inline program_t build_info_t::program() const noexcept
{
    return detail::wrap(platform_handle_, context_handle_, program_handle_, is_not_owning);
}

inline context::device_t build_info_t::device() const noexcept
{
    return context::device::wrap(platform_handle_, context_handle_, target_handle_);
}

} // namespace program
} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_BUILD_INFO_HPP_
