#ifndef OPENCL_WRAPPERS_PROGRAM_LINK_HPP_
#define OPENCL_WRAPPERS_PROGRAM_LINK_HPP_

#include "../program.hpp"

namespace opencl {

namespace program {

namespace detail {

// All objects must have the same context
link_t link_(
    platform::handle_t                  platform_handle,
    context::handle_t                   context_handle,
    span<handle_t const>                objects,
    span<device::handle_t const>        target_handles,
    optional_ref<link::options_t const>  options = nullopt);

link_t build_(
    program_t const&                    source_or_binary,
    span<device::handle_t const>        target_handles,
    optional_ref<build_options_t const> options);

} // namespace detail

class link_t : public with_kernels_t, public build_step_result_t {
public:
    friend link_t detail::link_(
        platform::handle_t                  platform_handle,
        context::handle_t                   context_handle,
        span<handle_t const>                objects,
        span<device::handle_t const>        target_handles,
        optional_ref<link::options_t const>  options);

    friend link_t detail::build_(
        program_t const&                    source_or_binary,
        span<device::handle_t const>        target_handles,
        optional_ref<build_options_t const> options);

protected:
    link_t(
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
        with_kernels_t(platform_handle, context_handle, program_handle, {}, owning),
        build_step_result_t(platform_handle, context_handle, program_handle, succeeded, owning)
    {}
}; // class link_result_

/**
 * Note: All building/linking work is _blocking_. If you want to perform any
 * such work asynchronously, invokew the function using @ref std::async with
 * the @ref std::launch::async policy.
 */
///@{


// All objects must have the same context
template <
    template <typename> class ObjectContainer,
    template <typename> class DeviceContainer
>
link_t link_(
    ObjectContainer<with_binaries_t const> const&  objects,
    DeviceContainer<device_t const> const&         targets,
    optional_ref<link::options_t const>            options = nullopt);

// All objects must have the same context
template <template <typename> class ObjectContainer>
link_t link_(
    ObjectContainer<with_binaries_t const> const&  objects,
    optional_ref<link::options_t const> const&     options = nullopt);

template <
    template <typename> class DeviceContainer
>
link_t link_(
    with_binaries_t const &                     objects,
    DeviceContainer<device_t const> const &     targets,
    optional_ref<link::options_t const>         options = nullopt);

link_t link_(
    with_binaries_t const&                      objects,
    optional_ref<link::options_t const> const&  options = nullopt);


// TODO: Are the header_t's allowed to be owning?
// If not, maybe we'd better take const references to containers
template <
    template <typename> class HeaderContainer,
    template <typename> class DeviceContainer
>
link_t compile_and_link(
    source_t const&                             main_source,
    DeviceContainer<device_t> const&            targets,
    HeaderContainer<header_t const> const&      headers = HeaderContainer<header_t const>{},
    optional_ref<compilation::options_t const>  compilation_options = nullopt,
    optional_ref<link::options_t const>         link_options = nullopt);

template <template <typename> class HeaderContainer>
link_t compile_and_link(
    source_t const&                             main_source,
    HeaderContainer<header_t const> const&      headers,
    optional_ref<compilation::options_t const>  compilation_options = nullopt,
    optional_ref<link::options_t const>         link_options = nullopt);

link_t compile_and_link(
    source_t const&                             source,
    optional_ref<compilation::options_t const>  compilation_options = nullopt,
    optional_ref<link::options_t const>         link_options = nullopt);

template <template <typename> class DeviceContainer>
link_t compile_and_link(
    with_intermediate_language_t const&         main_source,
    DeviceContainer<device_t> const&            targets,
    optional_ref<compilation::options_t const>  compilation_options = nullopt,
    optional_ref<link::options_t const>         link_options = nullopt);

link_t compile_and_link(
    with_intermediate_language_t const&         main_source,
    optional_ref<compilation::options_t const>  compilation_options = nullopt,
    optional_ref<link::options_t const>         link_options = nullopt);

// TODO: Explain the difference between build and compile-and-link

template <template <typename> class DeviceContainer>
link_t build_(
    program_t const&                     source_or_binary,
    DeviceContainer<device_t> const &    targets,
    optional_ref<build_options_t const>  options = nullopt);

link_t build_(
    program_t const&                     source_or_binary,
    device_t const&                      target,
    optional_ref<build_options_t const>  options = nullopt);

link_t build_(
    program_t const&                     source_or_binary,
    optional_ref<build_options_t const>  options = nullopt);

///@}

} // namespace program

} // namespace opencl

#endif // OPENCL_WRAPPERS_PROGRAM_LINK_HPP_
