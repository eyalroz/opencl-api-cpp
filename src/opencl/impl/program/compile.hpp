
#ifndef OPENCL_WRAPPERS_IMPL_PROGRAM_COMPILE_HPP_
#define OPENCL_WRAPPERS_IMPL_PROGRAM_COMPILE_HPP_

#include "../program.hpp"

namespace opencl {

namespace program {

namespace compilation {

inline compilation_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    handle_t            program_handle,
    bool                succeeded,
    bool                owning) noexcept
{
    return { platform_handle, context_handle, program_handle, succeeded, owning };
}

} // namespace compilation

namespace detail {

// Note: This function's name has an extra underscore to avoid template overload ambiguities
inline compilation_t compile_(
    program_t const&                            main_source,
    span<device::handle_t const>                target_handles,
    span<handle_t const>                        input_header_handles,
    span<char const*>                           input_header_names,
    optional_ref<compilation::options_t const>  options)
{
    // TODO: Check whether the program has already been built
#ifndef NDEBUG
    if (input_header_handles.size() != input_header_names.size()) {
        throw std::invalid_argument("Mismatched number of input header names and handles");
    }
    if ((not input_header_handles.empty()) and has_intermediate_language(main_source)) {
        throw std::invalid_argument("Attempt to compile an intermediate-language source with"
            " additional headers (OpenCL ignores them)");
    }
#endif
    auto options_string = options ? options->render() : "";
    auto status = clCompileProgram(
        main_source.handle(),
        target_handles.size(),
        target_handles.data(),
        options_string.c_str(),
        input_header_handles.size(),
        input_header_handles.data(),
        input_header_names.data(),
        nullptr, nullptr);
    if (is_failure(status) and status != CL_COMPILE_PROGRAM_FAILURE) {
        throw runtime_error(status, "clCompileProgram", "Compiling "
            + opencl::detail::identify(main_source));
    }
    // Q: Why are we excepting CL_COMPILE_PROGRAM_FAILURE? Even despite not providing a callback for
    //    an asynchronous build?
    // A: Because a compiler-reported error is not an error with the compiler or the library or a
    //    inappropriate application of the OpenCL API; it is not an "exceptional" situation, just
    //    a valid possible result.
    auto succeeded = is_success(status);
    return compilation::wrap(
        main_source.platform_handle(),
        main_source.context_handle(),
        main_source.handle(),
        succeeded,
        is_not_owning // The main source program owns the handle :-(
    );
}

inline compilation_t compile(
    with_intermediate_language_t const&         il_source,
    span<device::handle_t const>                target_handles,
    optional_ref<compilation::options_t const>  options)
{
    auto input_header_handles = span<handle_t const>{};
    auto input_header_names = span<char const*>{};
    return compile_(il_source, target_handles, input_header_handles, input_header_names, options);
}


template <template <typename> class HeaderContainer>
compilation_t compile(
    program_t const&                            main_source,
    span<device::handle_t const>                target_handles,
    HeaderContainer<header_t const>             headers,
    optional_ref<compilation::options_t const>  options)
{
    auto header_names = make_dynarray<char const*>(headers.size());
    auto header_handles = make_dynarray<handle_t>(headers.size());
    // Yes, this is ugly, enumerate would be nice, or perhaps an unzip function
    for (size_t i = 0; i < headers.size(); ++i) {
        header_names[i] = headers[i].names()->front().data();
        header_handles[i] = headers[i].handle();
    }
    return compile_(
        main_source,
        std::move(target_handles),
        std::move(header_handles),
        std::move(header_names),
        std::move(options));
}

} // namespace detail

template <
    template <typename> class HeaderContainer,
    template <typename> class DeviceContainer
>
compilation_t compile(
    source_t const& main_source,
    DeviceContainer<device_t> const& targets,
    HeaderContainer<header_t const> const& headers,
    optional_ref<compilation::options_t const> options)
{
    auto target_handles = opencl::detail::get_handles(targets);
    return detail::compile(main_source, target_handles, headers, std::move(options));
}

template <template <typename> class HeaderContainer>
compilation_t compile(
    source_t const& main_source,
    HeaderContainer<header_t const> const& headers,
    optional_ref<compilation::options_t const> options)
{
    auto target_handles = detail::get_default_target_handles(main_source);
    return detail::compile(main_source, target_handles, headers, options);
}

template <template <typename> class DeviceContainer>
compilation_t compile(
    source_t const& main_source,
    DeviceContainer<device_t const> const& targets,
    optional_ref<compilation::options_t const> options)
{
    auto headers = span<header_t const>{};
    auto target_handles = opencl::detail::get_handles(targets);
    return detail::compile(main_source, target_handles, headers, options);
}

template <template <typename> class DeviceContainer>
compilation_t compile(
    with_intermediate_language_t const& main_source,
    DeviceContainer<device_t> const& targets,
    optional_ref<compilation::options_t const> options)
{
    auto target_handles = opencl::detail::get_handles(targets);
    span<header_t const> no_header_handles {};
    span<char const *> no_header_names {};
    return detail::compile(main_source, target_handles, no_header_handles, no_header_names, std::move(options));
}

inline compilation_t compile(
    source_t const& main_source,
    optional_ref<compilation::options_t const> options)
{
    auto headers = span<header_t const>{};
    // TODO: Why is the std::move necessary? Doesn't the compile() overload
    // we're calling take a forwarding reference to a header container?
    return compile(main_source, headers, options);
}

inline compilation_t compile(
    with_intermediate_language_t const& main_source,
    optional_ref<compilation::options_t const> options)
{
    context_t context = main_source.context();
    auto target_handles = context.device_handles();
    span<handle_t> no_header_handles{};
    span<char const *> no_header_names{};
    return detail::compile_(main_source, target_handles, no_header_handles, no_header_names, options);
}

namespace async {

template <
    template <typename> class HeaderContainer,
    template <typename> class DeviceContainer
>
std::future<compilation_t> compile(
    source_t const& main_source,
    DeviceContainer<device_t> const& targets,
    HeaderContainer<header_t const> const& headers,
    optional_ref<compilation::options_t const> options)
{
    return std::async(program::compile, main_source, targets, headers, options);
}

template <template <typename> class HeaderContainer>
std::future<compilation_t> compile(
    source_t const& main_source,
    HeaderContainer<header_t const> && headers,
    optional_ref<compilation::options_t const> options )
{
    return std::async(program::compile, main_source, headers, options);
}

template <template <typename> class DeviceContainer>
std::future<compilation_t> compile(
    source_t const& main_source,
    DeviceContainer<device_t> && targets,
    optional_ref<compilation::options_t const> options)
{
    return std::async(program::compile, main_source, targets, options);
}

inline std::future<compilation_t> compile(
    source_t const& main_source,
    optional_ref<compilation::options_t const> options)
{
    context_t context = main_source.context();
    span<device::handle_t const> target_handles = context.device_handles();
    span<handle_t> no_header_handles{};
    span<char const *> no_header_names{};
    program_t const& main_source_ = main_source;
    return std::async(detail::compile_, main_source_, target_handles, no_header_handles, no_header_names, options);
}

} // namespace async

} // namespace program

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_PROGRAM_COMPILE_HPP_
