
#ifndef OPENCL_WRAPPERS_IMPL_PROGRAM_SOURCE_HPP_
#define OPENCL_WRAPPERS_IMPL_PROGRAM_SOURCE_HPP_

#include "../../program.hpp"

namespace opencl {

namespace program {

namespace source {

inline source_t wrap(
    platform::handle_t               platform_handle,
    context::handle_t                context_handle,
    program::handle_t                program_handle,
    optional<dynarray<string_view>>  names,
    optional<dynarray<string_view>>  source_texts,
    bool                             owning) NOEXCEPT_IF_NDEBUG
{
#ifndef NDEBUG
    if (names and not source_texts) {
        throw std::invalid_argument(
            "Attempt to associate source names with a program object without also associating the actual sources");
    }
#endif
    return source_t { platform_handle, context_handle, program_handle, std::move(names), std::move(source_texts), owning };
}

namespace detail {

inline handle_t create_handle(
    context::handle_t         context_handle,
    span<string_view> const&  source_strings,
    span<size_t> const&       lengths)
{
    status_t status;
#ifndef NDEBUG
    if (source_strings.size() != lengths.size()) {
        throw std::invalid_argument("Mismatched number of source text pointers and source lengths for "
            "creating an OpenCL program: " + std::to_string(source_strings.size()) + " != " + std::to_string(lengths.size()));
    }
    auto num_sources = source_strings.size();
    for (size_t i = 0 ; i < source_strings.size(); i++) {
        if (source_strings[i].empty()) {
            throw std::invalid_argument("Attempt to create an OpenCL program source object where the "
                + ((num_sources > 1) ? opencl::detail::xth(i + 1) + " " : "")
                + "input has a null pointer to source text");
        }
    }
#endif
    auto source_c_strs = to_dynarray<char const*>(source_strings, [](string_view sv) { return sv.data(); });
    auto program_handle = clCreateProgramWithSource(context_handle, 1,
        source_c_strs.data(), lengths.data(), &status);
    throw_if_error_lazy(status, "clCreateProgramWithSource", "Failed creating an OpenCL program from source code in "
        + opencl::detail::identify(context_handle));
    // TODO: Perhaps list the source names when throwing the exception, or is it too much? Perhaps just the length?
    // TODO: return a status and handle?
    return program_handle;
}

// Remember that if no lengths are specified, the strings must be null-terminated,
// but given the lengths - they don't have to be. An individual length may be 0,
// in which case, again, its corresponding string must be null-terminated.
inline handle_t create_handle(
    context::handle_t              context_handle,
    span<char const *> const&      source_strings,
    optional<span<size_t>> const&  lengths)
{
    status_t status;
#ifndef NDEBUG
    auto num_sources = source_strings.size();
    if (lengths and lengths->size() != num_sources) {
        throw std::invalid_argument("Mismatched number of source text pointers and source lengths for "
            "creating an OpenCL program: " + std::to_string(source_strings.size()) + " != " + std::to_string(lengths->size()));
    }
    for (size_t i = 0 ; i < source_strings.size(); i++) {
        if (source_strings[i] == nullptr) {
            throw std::invalid_argument("Attempt to create an OpenCL program source object where the "
                + ((num_sources > 1) ? opencl::detail::xth(i + 1) + " " : "")
                + "input has a null pointer to source text");
        }
    }
#endif
    auto program_handle = clCreateProgramWithSource(context_handle, 1,
        source_strings.data(), (lengths ? lengths->data() : nullptr), &status);
    throw_if_error_lazy(status, "clCreateProgramWithSource", "Failed creating an OpenCL program from source code in "
        + opencl::detail::identify(context_handle));
    // TODO: Perhaps list the source names when throwing the exception, or is it too much? Perhaps just the length?
    // TODO: return a status and handle?
    return program_handle;
}

inline handle_t create_handle(context::handle_t context_handle, named_source_text_t const& named_source)
{
    auto source_ptr = named_source.text.data();
    auto source_length = named_source.text.length();
    auto source_ptrs = span<char const*>{&source_ptr, 1lu};
    auto source_lengths = span<size_t>{&source_length, 1lu};
    return create_handle(context_handle, source_ptrs, source_lengths);
}

// Q: Why is this in detail?
// A: Because it's easy to mix up the names and source text parameters - as they have the same type
inline source_t create(context_t const& context,
    optional<dynarray<string_view>> names,
    dynarray<string_view> source_texts)
{
    auto raw_source_strings = generate_dynarray<char const *>(source_texts.size(),
        [&](size_t i) { return source_texts[i].data(); });
    auto source_lengths = generate_dynarray<size_t>(source_texts.size(),
        [&](size_t i) { return source_texts[i].length(); });
    auto handle = create_handle(context.handle(), raw_source_strings, source_lengths);
    return wrap(
        context.platform_handle(), context.handle(), handle,
        std::move(names), std::move(source_texts),  is_owning);
}

} // namespace detail

inline source_t create(
    context_t const&                   context,
    span<string_view> const&           source_names,
    span<input::text_source_t> const&  source_texts)
{
    auto source_names_ = to_dynarray<string_view>(source_names);
    auto source_texts_ = to_dynarray<string_view>(source_texts,
        [](input::text_source_t const & ts) { return ts.input; });
    return detail::create(context, std::move(source_names_), std::move(source_texts_));
}

template <typename Container, typename>
source_t create(context_t const& context, Container&& named_sources)
{
    auto names = to_dynarray<string_view>(named_sources,
        [](named_source_text_t const & ns) { return ns.name; });
    auto texts = to_dynarray<string_view>(named_sources,
        [](named_source_text_t const & ns) { return ns.text; });
    return detail::create(context, std::move(names), std::move(texts));
}

template <template <typename> class Container>
source_t create(
    context_t const&                        context,
    Container<input::text_source_t> const&  source_texts)
{
    auto texts_ = to_dynarray<string_view>(source_texts,
        [](named_source_text_t const & ns) { return ns.text; });
    static constexpr auto no_names = nullopt;
    return detail::create(context, no_names, std::move(texts_));
}

template <template <typename> class Container>
source_t create(
    context_t const&               context,
    Container<char const*> const&  source_texts)
{
    auto texts_ = to_dynarray<string_view>(source_texts,
        [](char const* cstr) -> string_view { return cstr; });
    static constexpr auto no_names = nullopt;
    return detail::create(context, no_names, std::move(texts_));
}


template <template <typename> class Container>
source_t create(
    context_t const&               context,
    Container<string_view> const&  source_texts)
{
    static constexpr auto no_names = nullopt;
    return detail::create(context, no_names, to_dynarray<string_view>(source_texts));
}

inline source_t create(context_t const& context, named_source_text_t named_source)
{
    handle_t handle  = detail::create_handle(context.handle(), named_source);
    // TODO: Perhaps define a utility named-constructor-idiom named singleton_dynarray? or wrap_in_dynarray?
    auto names { generate_dynarray<string_view>(1, [&](size_t) { return named_source.name; }) };
    auto source_texts { generate_dynarray<string_view>(1, [&](size_t) { return named_source.text; }) };
    return wrap(
        context.platform_handle(), context.handle(), handle,
        std::move(names), std::move(source_texts),  is_owning);
    // throw_if_error_lazy(status, "clCreateProgramWithSource",
    //     "Failed creating an OpenCL program from the single source named '" + std::string(named_source.name) + "\'");
    // TODO: Maybe an identify() for a named source?
}

inline source_t create(context_t const& context, string_view source_text)
{
    auto source_texts = span<string_view>{&source_text, 1lu};
    return create(context, source_texts);
}

} // namespace source

template <template <typename> class HeaderContainer>
compilation_t source_t::compile(
    device_t const& target,
    HeaderContainer<header_t const> const& headers,
    optional_ref<compilation::options_t const> options) const
{
    return program::compile(*this, target, headers, options);
}

template <
    template <typename> class HeaderContainer,
    template <typename> class DeviceContainer
>
compilation_t source_t::compile(
    DeviceContainer<device_t> const& targets,
    HeaderContainer<header_t const> const& headers,
    optional_ref<compilation::options_t const> options) const
{
    return program::compile(*this, targets, headers, options);
}

template <template <typename> class HeaderContainer>
compilation_t source_t::compile(
    HeaderContainer<header_t const> const& headers,
    optional_ref<compilation::options_t const> options) const
{
    return program::compile(*this, headers, options);
}

inline link_t source_t::build(
    device_t const& target,
    optional_ref<build_options_t const> options) const
{
    return build_(*this, target, options);
}

template <template <typename> class DeviceContainer>
link_t source_t::build(
    DeviceContainer<device_t> const& targets,
    optional_ref<build_options_t const> options) const
{
    return build_(*this, targets, options);
}

inline link_t source_t::build(optional_ref<build_options_t const> options) const
{
    return build_(*this, options);
}

inline optional<dynarray<named_source_text_t>> source_t::named_source_texts() const
{
#ifndef NDEBUG
    if ((names_ and not source_texts_) or (not names_ and source_texts_)) {
        throw std::logic_error("Source program wrapper object source name/text caching inconsistency");
    }
#endif
    if (not source_texts_) { return nullopt;}
    auto get_named_source = [&](size_t i) -> named_source_text_t {
        return { (*names_)[i], (*source_texts_)[i] };
    };
    return generate_dynarray(source_texts_->size(), get_named_source);
}

} // namespace program

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_PROGRAM_SOURCE_HPP_
