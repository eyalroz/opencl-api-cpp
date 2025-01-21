
#ifndef OPENCL_WRAPPERS_IMPL_PROGRAM_HPP_
#define OPENCL_WRAPPERS_IMPL_PROGRAM_HPP_

#include "../program.hpp"
#include "../options/compilation.hpp"
#include "../kernel.hpp"
#include "ownership_token_t.hpp"
#include "../util/miscellany.hpp"
#include "miscellany.hpp"

namespace opencl {

inline char const* info::traits_t<info::program>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case  CL_PROGRAM_REFERENCE_COUNT:                  return "reference count";
    case  CL_PROGRAM_CONTEXT:                          return "context handle";
    case  CL_PROGRAM_NUM_DEVICES:                      return "number of target devices";
    case  CL_PROGRAM_DEVICES:                          return "target devices";
    case  CL_PROGRAM_SOURCE:                           return "source code";
    case  CL_PROGRAM_BINARY_SIZES:                     return "built binary sizes";
    case  CL_PROGRAM_BINARIES:                         return "built binaries";
#ifdef CL_VERSION_1_2
    case  CL_PROGRAM_NUM_KERNELS:                      return "number of kernels";
    case  CL_PROGRAM_KERNEL_NAMES:                     return "kernel names";
#endif
#ifdef CL_VERSION_2_1
    case  CL_PROGRAM_IL:                               return "intermediate language code";
#endif
#ifdef CL_VERSION_2_2
    case  CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT:       return "non-trivial constructors indicator";
    case  CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT:       return "non-trivial destructors indicator";
#endif
    default: return nullptr;
    }
}

inline char const* name_of(program::build_status_t status)
{
    using bs = program::build_status_t;
    switch (status) {
    case bs::none:                return "(no build)";
    case bs::error:               return "error";
    case bs::in_progress:         return "in progress";
    case bs::success:             return "success";
    }
    return nullptr;
}

namespace detail {

OCLW_DEFINE_HANDLE_TRAITS("program", program::handle_t, clReleaseProgram, clRetainProgram);

} // namespace detail

inline bool has_sources(program_t const& program)
{
    return info::has_string<CL_PROGRAM_SOURCE>(program.handle());
}

inline bool has_intermediate_language(program_t const& program)
{
    return info::has_string<CL_PROGRAM_IL>(program.handle());
}

inline bool has_binaries(program_t const& program)
{
    return info::has_array<CL_PROGRAM_BINARY_SIZES>(program.handle());
}

inline bool has_kernels(program_t const& program)
{
    return info::has_array<CL_PROGRAM_NUM_KERNELS>(program.handle());
}


inline platform_t program_t::platform() const noexcept
{
    return platform::wrap(platform_handle_);
}

inline context_t program_t::context() const noexcept
{
    return context::wrap(platform_handle_, nullopt, context_handle_, is_not_owning);
}

inline bool program_t::has_sources() const { return opencl::has_sources(*this); }
inline bool program_t::has_intermediate_language() const { return opencl::has_intermediate_language(*this); }
inline bool program_t::has_binaries() const { return opencl::has_binaries(*this); }
inline bool program_t::has_kernels() const { return opencl::has_kernels(*this); }

inline bool operator==(const program_t& lhs, const program_t& rhs) noexcept
{
    // TODO: Is it not sufficient to merely compare the handle field? Handles should be unique after all
    return lhs.platform_handle() == rhs.platform_handle() and
        lhs.context_handle() == rhs.context_handle() and
        lhs.handle() == rhs.handle();
}

namespace program {

namespace detail {

context::handle_t get_context_handle(handle_t program_handle);

inline dynarray<device::handle_t> get_default_target_handles(handle_t program_handle)
{
    return info::get_array<CL_PROGRAM_DEVICES>(program_handle);
}

inline dynarray<device::handle_t> get_default_target_handles(program_t const& program)
{
    return get_default_target_handles(program.handle());
}

inline context::handle_t get_context_handle(handle_t program_handle)
{
    return info::get_scalar<CL_PROGRAM_CONTEXT>(program_handle);
}

/// When OpenCL provides us with the kernel names, it does so in a string
/// containing all names, with this character as its delimiter.
static constexpr char kernel_name_delimiter { ';' };


} // namespace detail

inline kernel_t with_kernels_t::instantiate_kernel(char const* kernel_name) const
{
    // We're in a bit of a bind. with_kernels_t is also inherited by classes
    // which represent build step results; so, we want to avoid trying to create
    // the kernel if the build failed; yet - the code here doesn't know about builds;
    // and if we override, then why have this class exist in the first place?
    return kernel::instantiate(*this, kernel_name);
}

inline size_t with_kernels_t::num_kernels() const
{
    return kernel_names_ ? kernel_names_->size() :
        info::get_scalar<CL_PROGRAM_NUM_KERNELS>(handle_);
}

inline dynarray<std::string> const& with_kernels_t::kernel_names() const
{
    if (kernel_names_) {
        return *kernel_names_;
    }
    kernel_names_.emplace(info::get_tokenized_string<CL_PROGRAM_KERNEL_NAMES>(handle_, detail::kernel_name_delimiter));
    return *kernel_names_;
}

namespace detail {

inline program_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    handle_t            program_handle,
    bool                owning) noexcept
{
    return { platform_handle, context_handle, program_handle, owning };
}

inline with_kernels_t wrap_with_kernels(
    platform::handle_t platform_handle,
    context::handle_t context_handle,
    program::handle_t program_handle,
    optional<dynarray<std::string>> kernel_names,
    bool owning) noexcept
{
    return { platform_handle, context_handle, program_handle, kernel_names, owning };
}

inline with_intermediate_language_t wrap_with_intermediate_language(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    program::handle_t   program_handle,
    bool                owning) noexcept
{
    return { platform_handle, context_handle, program_handle, owning };
}

inline build_info_t wrap_build_info(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    handle_t            program_handle,
    device::handle_t    device_handle) noexcept
{
    return { platform_handle, context_handle, program_handle, device_handle };
}

inline build_info_t build_info_for(program_t const& program, device::handle_t device_handle)
{
    return wrap_build_info(program.platform_handle(), program.context_handle(), program.handle(), device_handle);
}

inline build_info_t get_build_info_for(handle_t program_handle, device::handle_t device_handle)
{
    auto context_handle = get_context_handle(program_handle);
    auto platform_handle = context::detail::get_platform_handle(context_handle);
    return wrap_build_info(platform_handle, context_handle, program_handle, device_handle);
}

inline dynarray<build_info_t> get_build_info(handle_t program_handle)
{
    auto device_handles = get_default_target_handles(program_handle);
    return generate_dynarray<build_info_t>(device_handles.size(), [&](size_t i) {
        return get_build_info_for(program_handle, device_handles[i]);
    });
}

inline dynarray<build_info_t> build_info(program_t const& program)
{
    auto device_handles = get_default_target_handles(program);
    return generate_dynarray<build_info_t>(device_handles.size(), [&](size_t i) {
        return detail::build_info_for(program, device_handles[i]);
    });
}

inline dynarray<device::handle_t> get_binary_device_handles(with_binaries_t const& build_step_result)
{
    // TODO: Do we really not need the number of devices?
    // auto num_devices = info::get_scalar<CL_PROGRAM_NUM_DEVICES>(build_step_result.handle());
    return  info::get_array<CL_PROGRAM_DEVICES>(build_step_result.handle());
}

} // namespace detail

inline dynarray<build_info_t> build_step_result_t::info() const
{
    auto device_handles = detail::get_binary_device_handles(*this);
    return generate_dynarray<build_info_t>(device_handles.size(), [&](size_t i) {
        return detail::build_info_for(*this, device_handles[i]);
    });
}

inline build_info_t build_step_result_t::info_for(device_t const& device) const
{
    return detail::build_info_for(*this, device.handle());
}

inline with_binaries_t::map_type with_binaries_t::binaries() const
{
    auto device_handles = info::get_array<CL_PROGRAM_DEVICES>(handle_);
    auto binary_sizes = info::get_array<CL_PROGRAM_BINARY_SIZES>(handle_);
    auto binary_data = info::get_array<CL_PROGRAM_BINARIES>(handle_);
    map_type result;

    // A zip function would be nice...
    for (size_t i = 0; i < binary_sizes.size(); ++i) {
        auto target = device::wrap(platform_handle(), device_handles[i]);
#if CL_VERSION_1_2
        auto type_ = binary_type_for(target);
#endif
        raw_binary_t raw { reinterpret_cast<byte_t const *>(binary_data[i]), binary_sizes[i] };
        binary_t binary { target,
#if CL_VERSION_1_2
            type_,
#endif
            raw };
        result.emplace(std::move(target), std::move(binary));
    }
    return result;
}

inline optional<binary_t> with_binaries_t::binary_for(device_t const& target) const
{
    auto device_handles = context::detail::get_device_handles(context_handle_);
    auto iter = std::find(device_handles.begin(), device_handles.end(), target.handle());
    if (iter == device_handles.end()) {
        return nullopt;
    }
    auto pos = iter - device_handles.begin();
    auto binary_sizes = info::get_array<CL_PROGRAM_BINARY_SIZES>(handle_);
    auto binary_data = info::get_array<CL_PROGRAM_BINARIES>(handle_);
    raw_binary_t raw { reinterpret_cast<byte_t const *>(binary_data[pos]), binary_sizes[pos] };
#if CL_VERSION_1_2
    auto type_ = binary_type_for(target);
    return binary_t{ target, type_, raw };
#else
    return binary_t{ target, raw };
#endif
}

#if CL_VERSION_1_2
inline binary_type_t with_binaries_t::binary_type_for(device_t const& target) const
{
    return static_cast<binary_type_t>(info::get_scalar<CL_PROGRAM_BINARY_TYPE>( { handle_, target.handle() } ));
}
#endif // CL_VERSION_1_2

inline dynarray<device_t> with_binaries_t::targets() const
{
    auto device_handles = context::detail::get_device_handles(context_handle_);
    return generate_dynarray<device_t>(device_handles.size(),
        [&](size_t i) -> device_t { return device::wrap(platform_handle(), device_handles[i]); });
}

inline void unload_compiler()
{
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    auto status = clUnloadCompiler();
#pragma GCC diagnostic pop
#endif
    throw_if_error_lazy(status, "clUnloadCompiler", "");
}

inline dynarray<kernel_t> with_kernels_t::instantiate_kernels(span<char const *> kernel_names) const
{
    if (kernel_names.empty()) {
        return kernel::instantiate_all(*this);
    }
    auto generator = [&](size_t i) { return instantiate_kernel(kernel_names[i]); };
    return generate_dynarray(kernel_names.size(), generator);
}

template <
    template <typename> class BinariesContainer,
    template <typename> class DevicesContainer>
with_binaries_t create(
    context_t const &                                 context,
    BinariesContainer<input::binary_t const> const &  binaries,
    DevicesContainer<device_t const> const &          devices)
{
    auto device_handles = opencl::detail::get_handles(devices);

    auto num_binaries = binaries.size();
    auto binary_length_getter = [&](size_t i) -> size_t { return binaries[i].size(); };
    auto binary_lengths = generate_dynarray(num_binaries, binary_length_getter);
    auto binary_data_getter = [&](size_t i) { return static_cast<unsigned char const *>(binaries[i].data()); };
    auto binary_data_ptrs = generate_dynarray(num_binaries, binary_data_getter);
    auto load_stati = make_dynarray<cl_int>(num_binaries);

    cl_int status;
    auto program_handle = clCreateProgramWithBinary(
        context.handle(),
        device_handles.size(), device_handles.data(),
        binary_lengths.data(), binary_data_ptrs.data(),
        load_stati.data(),
        &status);
    throw_if_error_lazy(status, "clCreateProgramWithBinary", "General failure creating a program with "
        + std::to_string(num_binaries) + " binaries");
    for (auto single_binary_load_status : load_stati) {
        throw_if_error_lazy(single_binary_load_status,
            "clCreateProgramWithBinary", "Creation of multiple binaries completed, but failed to "
            "load binary " + std::to_string(num_binaries) + " of " + std::to_string(num_binaries));
    }
    return detail::wrap_binaries(context.platform_handle(), context.handle(), program_handle, is_owning);
}

template <template <typename> class DevicesContainer>
with_binaries_t create(
    context_t const &                                 context,
    input::binary_t const &                           binary,
    DevicesContainer<device_t const> const &          devices)
{
    span<input::binary_t const> binaries = { &binary, 1lu };
    return create(context, binaries, devices);
}

template <template <typename> class BinariesContainer>
with_binaries_t create(
    context_t const &                                 context,
    BinariesContainer<input::binary_t const> const &  binaries,
    device_t const &                                  device)
{
    span<device_t const> devices = { &device, 1lu };
    return create(context, binaries, devices);
}

inline with_binaries_t create(
    context_t const &        context,
    input::binary_t const &  binary,
    device_t const &         device)
{
    span<input::binary_t const> binaries = { &binary, 1lu };
    span<device_t const> devices = { &device, 1lu };
    return create<span, span>(context, binaries, devices);
}

#ifdef CL_VERSION_1_2

namespace detail {

template <>
inline with_intermediate_language_t upcast<with_intermediate_language_t>(program_t const& program)
{
    auto il_source = info::get_string<CL_PROGRAM_IL>(program.handle());
    if (il_source.data() == nullptr) {
        throw std::invalid_argument(
            "Attempt to upcast an OpenCL program into an intermediate-language source, which does"
            "not have intermediate-language text: " + opencl::detail::identify(program));
    }
    return wrap_with_intermediate_language(
        program.platform_handle(), program.context_handle(), program.handle(), is_not_owning);
}

template <>
inline source_t upcast<source_t>(program_t const& program)
{
    auto num_sources = info::get_array_size<CL_PROGRAM_SOURCE>(program.handle());
    if (num_sources == 0) {
        throw std::invalid_argument(
            "Attempt to upcast an OpenCL program into an source object, while it does"
            "not have any source text: " + opencl::detail::identify(program));
    }
    return source::wrap(program.platform_handle(), program.context_handle(), program.handle(), {}, {}, is_not_owning);
}

template <>
inline with_binaries_t upcast<with_binaries_t>(program_t const& program)
{
    auto num_binaries = info::get_array_size<CL_PROGRAM_BINARY_SIZES>(program.handle());
    if (num_binaries == 0) {
        throw std::invalid_argument(
            "Attempt to upcast an OpenCL program into a compilation-result object, while it does"
            "not have any binaries: " + opencl::detail::identify(program));
    }
    enum { succeeded = true };
    return wrap_binaries(program.platform_handle(), program.context_handle(), program.handle(), is_not_owning);
}

template <>
inline with_kernels_t upcast<with_kernels_t>(program_t const& program)
{
    auto num_kernels = info::get_scalar<CL_PROGRAM_NUM_KERNELS>(program.handle());
    if (num_kernels == 0) {
        throw std::invalid_argument(
            "Attempt to upcast an OpenCL program into a kernel-holding object, while it does"
            "not have any kernels: " + opencl::detail::identify(program));
    }
    enum { succeeded = true };
    return with_kernels_t(program.platform_handle(), program.context_handle(), program.handle(), {}, is_not_owning);
}

} // namespace detail

template <template <typename> class DevicesContainer>
with_kernels_t create_with_builtin_kernels(
    context_t const &                         context,
    span<string_view const>                   kernel_names,
    DevicesContainer<device_t const> const &  devices)
{
    auto device_handles = opencl::detail::get_handles(devices);
    auto semicolon_separated_list = opencl::detail::join(kernel_names, ';');

    cl_int status;
    auto program_handle = clCreateProgramWithBuiltInKernels(
        context.handle(),
        device_handles.size(), device_handles.data(),
        semicolon_separated_list.data(), &status);
    throw_if_error_lazy(status, "clCreateProgramWithBuiltInKernels", "Failure creating a program with "
        + std::to_string(kernel_names.size()) + " built-in kernels");
    auto owning_kernel_names = to_dynarray<std::string>(kernel_names);
    return detail::wrap_with_kernels(context.platform_handle(), context.handle(), program_handle, std::move(owning_kernel_names), is_owning);
}

inline with_kernels_t create_with_builtin_kernels(
    context_t const &                         context,
    span<string_view const>                   kernel_names,
    device_t const &                          device)
{
    auto devices = span<device_t const>{ &device, 1lu };
    return create_with_builtin_kernels<span>(context, kernel_names, devices);
}
#endif // CL_VERSION_1_2

inline with_intermediate_language_t create_with_intermediate_language(
    context_t const &  context,
    memory::region_t   il_source)
{
    status_t status;
#ifdef CL_VERSION_2_1
    static constexpr auto api_function_name = "clCreateProgramWithIL";
    auto api_function = clCreateProgramWithIL;
#elif defined cl_khr_il_program
    static constexpr auto api_function_name = "clCreateProgramWithILKHR";
    auto api_function = clCreateProgramWithILKHR;
#endif
    auto program_handle = api_function(context.handle(), il_source.data(), il_source.size(), &status);
    throw_if_error_lazy(status, api_function_name, "Createing a program from an intermediate-language source");
    return with_intermediate_language_t { context.platform_handle(), context.handle(), program_handle, is_owning };
}

inline string_view with_intermediate_language_t::intermediate_language() const
{
    return info::get_string<CL_PROGRAM_IL>(handle_);
}

} // namespace program

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_PROGRAM_HPP_
