
#ifndef OPENCL_WRAPPERS_IMPL_PROGRAM_LINK_HPP_
#define OPENCL_WRAPPERS_IMPL_PROGRAM_LINK_HPP_

#include "../program.hpp"
#include "../miscellany.hpp"

namespace opencl {

namespace program {

namespace detail {

// All objects must have the same context
inline link_t link_(
    platform::handle_t                   platform_handle,
    context::handle_t                    context_handle,
    span<program::handle_t const>        objects,
    span<device::handle_t const>         target_handles,
    optional_ref<link::options_t const>  options)
{
    status_t status;
    auto rendered_options = options ? options->render() : "";
    auto result_handle = clLinkProgram(context_handle,
        target_handles.size(), target_handles.data(),
        rendered_options.c_str(),
        objects.size(),
        objects.data(),
        nullptr, nullptr, // no callbacks
        &status);
    throw_if_error_lazy(status, "clLinkProgram", "Linking failed");
    return link_t {
        platform_handle,
        context_handle,
        result_handle,
        is_success(status),
        is_owning
    };
}

template <template <typename> class ObjectContainer>
link_t link_(
    ObjectContainer<with_binaries_t const> const&  objects,
    span<device::handle_t const>                   target_handles,
    optional_ref<link::options_t const>            options)
{
    if (objects.empty()) {
        throw std::invalid_argument("Attempt to link an empty container of object");
    }
    if (target_handles.empty()) {
        throw std::invalid_argument("Attempt to link kernels without any targets");
    }
    auto platform_handle = objects.front().platform_handle();
    auto context_handle = objects.front().context_handle();
    auto object_handles = opencl::detail::get_handles(objects);
    return detail::link_(platform_handle, context_handle, object_handles, target_handles, options);
}

} // namespace detail

template <
    template <typename> class ObjectContainer,
    template <typename> class DeviceContainer
>
link_t link_(
    ObjectContainer<with_binaries_t const> const&  objects,
    DeviceContainer<device_t const> const&         targets,
    optional_ref<link::options_t const>            options)
{
    auto target_handles = opencl::detail::get_handles(targets);
    return detail::link_(objects, target_handles, options);
}

template <template <typename> class ObjectContainer>
link_t link_(
    ObjectContainer<with_binaries_t const> const& objects,
    optional_ref<link::options_t const>    const& options)
{
    auto target_handles = detail::get_default_target_handles(objects.front());
    return detail::link_(objects, target_handles, options);
}

template <
    template <typename> class DeviceContainer
>
link_t link_(
    with_binaries_t const &                     objects,
    DeviceContainer<device_t const> const &     targets,
    optional_ref<link::options_t const>         options)
{
#ifndef NDEBUG
    if (not objects.has_binaries()) {
        throw std::invalid_argument("Attempt to link a program without any compiled binaries: "
            + opencl::detail::identify(objects));
    }
#endif
    auto handle = objects.handle();
    auto objects_span = span<handle_t const> { &handle, 1 };
    return detail::link_(objects_span, targets, options);
}

inline link_t link_(
    with_binaries_t const& objects,
    optional_ref<link::options_t const> const& options)
{
    // TODO: factor out a singleton span idiom
    auto objects_span = span<with_binaries_t const> { &objects, 1 };
    return link_(objects_span, options);
}

namespace detail {

inline std::string describe_build_failure(
    program_t const&        source_or_binary,
    device::handle_t const& target_handle)
{
    auto build_info = build_info_for(source_or_binary, target_handle);
    if (not build_info.succeeded()) {}
    return build_info.log();
}

// TODO Perhaps offer a non-multiline version of this function?
inline std::string describe_build_failures(
    program_t const&                    source_or_binary,
    span<device::handle_t const>        target_handles)
{
    if (target_handles.size() == 1) {
        return describe_build_failure(source_or_binary, target_handles.front());
    }
    std::ostringstream result;
    auto build_failed = [&](device::handle_t const & target) {
        return build_info_for(source_or_binary, target).status() == CL_BUILD_PROGRAM_FAILURE;
    };
    auto num_failures = std::count_if(target_handles.begin(), target_handles.end(), build_failed);
    result << "Build failures for " << num_failures << " devices:\n\n";
    for (auto const& target : target_handles) {
        result << opencl::detail::identify(target) << " :\n" << describe_build_failure(source_or_binary, target) << "\n\n";
    }
    return result.str();
}

inline std::string list_build_status_counts(
    program_t const&                    source_or_binary,
    span<device::handle_t const>        target_handles)
{
    std::ostringstream result;
    enum { num_known_build_stati = 4 };
    auto build_status_to_count_index = [](build_status_t bs) {
        switch (static_cast<int>(bs)) {
        case CL_BUILD_SUCCESS: return 0;
        case CL_BUILD_NONE: return 1;
        case CL_BUILD_ERROR: return 2;
        case CL_BUILD_IN_PROGRESS: return 3;
        default: throw std::invalid_argument("Unknown build status");
        }
    };

    result << "Target device build stati: ";
    size_t counts[num_known_build_stati] = { };
    for (auto const& target : target_handles) {
        counts[build_status_to_count_index(build_info_for(source_or_binary, target).status())]++;
        //result << opencl::detail::identify(target) << " :\n" << describe_build_failure(source_or_binary, target) << "\n\n";
    }
    bool anything_printed { false };
    for (auto i : { CL_BUILD_SUCCESS, CL_BUILD_NONE, CL_BUILD_ERROR, CL_BUILD_IN_PROGRESS }) {
        auto bs = static_cast<build_status_t>(i);
        auto count = counts[i];
        if (count == 0) { continue; }
        if (anything_printed) {
            result << "; ";
        }
        result << name_of(bs) << ": " << count << " device" << (count > 1 ? "s" : "");
        anything_printed = true;
    }
    result << '.';
    return result.str();
}

inline link_t build_(
    program_t const&                    source_or_binary,
    span<device::handle_t const>        target_handles,
    optional_ref<build_options_t const> options)
{
    // TODO: Check whether the program has already been built
    std::string marshalled_options = options ? options->render() : "";
    static constexpr auto no_callback = nullptr;
    static constexpr auto no_callback_user_data = nullptr;
    auto status = clBuildProgram(
        source_or_binary.handle(),
        target_handles.size(),
        target_handles.data(),
        marshalled_options.c_str(),
        no_callback,
        no_callback_user_data);
    auto legitimate_failure = (status == CL_BUILD_PROGRAM_FAILURE);
    if (is_failure(status) and not legitimate_failure) {
        throw runtime_error(status, "clBuildProgram",
            "Possibly-compiling and linking " + opencl::detail::identify(source_or_binary));
    }
    // Q: Why are we excepting CL_BUILD_PROGRAM_FAILURE? Even despite not providing a callback for
    //    an asynchronous build?
    // A: Because a compiler-reported error is not an error with the compiler or the library or a
    //    inappropriate application of the OpenCL API; it is not an "exceptional" situation, just
    //    a valid possible result.
    return link_t {
        source_or_binary.platform_handle(),
        source_or_binary.context_handle(),
        source_or_binary.handle(),
        is_success(status),
        is_not_owning
    };
}

} // namespace detail

template <template <typename> class DeviceContainer>
link_t build_(
    program_t const& source_or_binary,
    DeviceContainer<device_t> const& targets,
    optional_ref<build_options_t const> options)
{
    auto device_handles = opencl::detail::get_handles(targets);
    return detail::build_(source_or_binary, device_handles, options);
}

inline link_t build_(
    program_t const& source_or_binary,
    device_t const& target,
    optional_ref<build_options_t const> options)
{
    auto device_handle = target.handle();
    span<device::handle_t> device_handles { &device_handle, 1 };
    return detail::build_(source_or_binary, device_handles, options);
}

inline link_t build_(
    program_t const& source_or_binary,
    optional_ref<build_options_t const> options)
{
    auto target_handles = detail::get_default_target_handles(source_or_binary);
    return detail::build_(source_or_binary, target_handles, options);
}

template <
    template <typename> class HeaderContainer,
    template <typename> class DeviceContainer
>
link_t compile_and_link(
    source_t const& main_source,
    DeviceContainer<device_t> const& targets,
    HeaderContainer<header_t const> const& headers,
    optional_ref<compilation::options_t const> compilation_options,
    optional_ref<link::options_t const> link_options)
{
    auto compilation_result = compile(main_source, targets, headers, compilation_options);
    return link_(compilation_result, targets, link_options);
}

template <template <typename> class HeaderContainer>
link_t compile_and_link(
    source_t const& main_source,
    HeaderContainer<header_t const> const& headers,
    optional_ref<compilation::options_t const> compilation_options,
    optional_ref<link::options_t const> link_options)
{
    compilation_t compilation_result = compile(main_source, headers, compilation_options);
    if (not compilation_result.succeeded()) {
        throw std::runtime_error("Compilation failed for "
            + opencl::detail::identify(main_source) + " - cannot proceed to linking");
    }
    return link_(compilation_result, link_options);
}

inline link_t compile_and_link(
    source_t const& source,
    optional_ref<compilation::options_t const> compilation_options,
    optional_ref<link::options_t const> link_options)
{
    span<header_t const> no_headers {};
    return compile_and_link(source, no_headers, compilation_options, link_options);
}

template <template <typename> class DeviceContainer>
link_t compile_and_link(
    with_intermediate_language_t const& main_source,
    DeviceContainer<device_t> const& targets,
    optional_ref<compilation::options_t const> compilation_options,
    optional_ref<link::options_t const> link_options)
{
    auto compilation_result = compile(main_source, targets, compilation_options);
    return link_(compilation_result, link_options);
}

inline link_t compile_and_link(
    with_intermediate_language_t const& main_source,
    optional_ref<compilation::options_t const> compilation_options,
    optional_ref<link::options_t const> link_options)
{
    auto compilation_result = compile(main_source, compilation_options);
    return link_(compilation_result, link_options);
}

} // namespace program

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_PROGRAM_LINK_HPP_
