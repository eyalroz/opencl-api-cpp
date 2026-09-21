/**
 * @file
 *
 * @brief Definition of the @ref program::compilation_t class, and declarations
 * of its named constructor idioms, and other related functions - including
 * the different {@ref program::compile_} functions typically used to trigger
 * a compilation of OpenCL device-side source code.
 */
#ifndef OPENCL_WRAPPERS_PROGRAM_COMPILE_HPP_
#define OPENCL_WRAPPERS_PROGRAM_COMPILE_HPP_

#include "../program.hpp"
#include "../util/optional_ref.hpp"

namespace opencl {

namespace program {

namespace detail {

compilation_t compile_(
    program_t const&                            main_source,
    span<device::handle_t const>                target_handles,
    span<handle_t const>                        input_header_handles = {},
    span<char const*>                           input_header_names = {},
    optional_ref<compilation::options_t const>  options = {});

}// namespace detail

namespace compilation {

compilation_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    handle_t            program_handle,
    bool                succeeded,
    bool                owning) noexcept;

} // namespace compilation

class compilation_t : public build_step_result_t {
public:
    using parent_type = build_step_result_t;

    friend compilation_t compilation::wrap(
        platform::handle_t  platform_handle,
        context::handle_t   context_handle,
        handle_t            program_handle,
        bool                succeeded,
        bool                owning) noexcept;

protected:
    compilation_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        handle_t program_handle,
        bool succeeded,
        bool owning
        // dynarray<device::handle_t> device_handles,
        // dynarray<program::binary_t> binaries
        ) noexcept
    :
        program_t(platform_handle, context_handle, program_handle, owning),
        parent_type(platform_handle, context_handle, program_handle, succeeded, owning)
    {}
}; // compilation_t

/**
* Note: All compilation work is _blocking_. If you want to perform any
 * such work asynchronously, invokew the function using @ref std::async with
 * the @ref std::launch::async policy.
 */
///@{
template <
    template <typename> class HeaderContainer,
    template <typename> class DeviceContainer
>
compilation_t compile(
    source_t const& main_source,
    DeviceContainer<device_t> const& targets,
    HeaderContainer<header_t const> const& headers = {},
    optional_ref<compilation::options_t const> options = nullopt);

template <template <typename> class HeaderContainer>
compilation_t compile(
    source_t const& main_source,
    HeaderContainer<header_t const> const& headers = {},
    optional_ref<compilation::options_t const> options = nullopt);

template <template <typename> class DeviceContainer>
compilation_t compile(
    with_intermediate_language_t const& main_source,
    DeviceContainer<device_t> const&  targets,
    optional_ref<compilation::options_t const> options = nullopt);

compilation_t compile(
    source_t const& main_source,
    optional_ref<compilation::options_t const> options = nullopt);

compilation_t compile(
    with_intermediate_language_t const& main_source,
    optional_ref<compilation::options_t const> options = nullopt);
///@}

namespace async {

/// Asynchronified versions of @ref program::compile() function variants
///@{

template <
    template <typename> class HeaderContainer,
    template <typename> class DeviceContainer
>
std::future<compilation_t> compile(
    source_t const&,
    DeviceContainer<device_t> const& targets,
    HeaderContainer<header_t const> const&,
    optional_ref<compilation::options_t const> = nullopt);

template <template <typename> class HeaderContainer>
std::future<compilation_t> compile(
    source_t const&,
    HeaderContainer<header_t const> &&,
    optional_ref<compilation::options_t const> = nullopt);

template <template <typename> class DeviceContainer>
std::future<compilation_t> compile(
    source_t const&,
    DeviceContainer<device_t> && targets,
    optional_ref<compilation::options_t const> = nullopt);

std::future<compilation_t> compile(
    source_t const&,
    optional_ref<compilation::options_t const> = nullopt);

} // namespace async

namespace build {
namespace options {
// Build options are (for now) the same as compilation options, s:
using compilation::options::create;
}
}

} // namespace program

} // namespace opencl

#endif // OPENCL_WRAPPERS_PROGRAM_COMPILE_HPP_
