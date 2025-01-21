#ifndef OPENCL_WRAPPERS_PROGRAM_SOURCE_HPP_
#define OPENCL_WRAPPERS_PROGRAM_SOURCE_HPP_

#include "../program.hpp"
#include "../util/optional_ref.hpp"

namespace opencl {

namespace program {

namespace source {

source_t wrap(
    platform::handle_t               platform_handle,
    context::handle_t                context_handle,
    program::handle_t                program_handle,
    optional<dynarray<string_view>>  names,
    optional<dynarray<string_view>>  source_texts,
    bool                             owning) NOEXCEPT_IF_NDEBUG;

} // namespace source

class source_t : public virtual program_t {
    using parent_type = program_t;
protected:
    optional<dynarray<string_view>> names_;
    optional<dynarray<string_view>> source_texts_;

protected:
    source_t(
        platform::handle_t               platform_handle,
        context::handle_t                context_handle,
        program::handle_t                program_handle,
        optional<dynarray<string_view>>  names,
        optional<dynarray<string_view>>  source_texts,
        bool owning) noexcept
    :
        parent_type(platform_handle, context_handle, program_handle, owning),
        names_(std::move(names)), source_texts_(std::move(source_texts)) {}

public:
    friend source_t source::wrap(
        platform::handle_t,
        context::handle_t,
        program::handle_t,
        optional<dynarray<string_view>>,
        optional<dynarray<string_view>>,
        bool) NOEXCEPT_IF_NDEBUG;

    // We can default the move ctor and assignment operator because the non-trivial
    // behavior only involves the base class

    source_t& operator=(source_t const&) = delete;
    // virtual source_t& operator=(source_t&&) noexcept = default;
    source_t& operator=(source_t&&) = delete;
    source_t(source_t const& other) = delete;
    source_t(source_t&& other) noexcept = default;

public:
    optional<dynarray<named_source_text_t>> named_source_texts() const;
    optional<dynarray<string_view>> const & names() const noexcept { return names_; }
    optional<dynarray<string_view>> const & source_texts() const noexcept { return source_texts_; }

    template <template <typename> class HeaderContainer>
    compilation_t compile(
        device_t const& target,
        HeaderContainer<header_t const> const& headers = {},
        optional_ref<compilation::options_t const> = {}) const;

    template <
        template <typename> class HeaderContainer,
        template <typename> class DeviceContainer
    >
    compilation_t compile(
        DeviceContainer<device_t> const& targets,
        HeaderContainer<header_t const> const& headers = {},
        optional_ref<compilation::options_t const> = {}) const;

    template <template <typename> class HeaderContainer>
    compilation_t compile(
        HeaderContainer<header_t const> const& headers = {},
        optional_ref<compilation::options_t const> = {}) const;

    link_t build(device_t const& target, optional_ref<build_options_t const> = {}) const;

    template <template <typename> class DeviceContainer>
    link_t build(DeviceContainer<device_t> const& targets, optional_ref<build_options_t const> = {}) const;

    link_t build(optional_ref<build_options_t const> = {}) const;
}; // class source_t

namespace source {

/// A placeholder for clearly indicating that a text source has no associated name,
/// in a ctor taking both a name and the source text.
static constexpr auto no_name = string_view{};

source_t create(
    context_t const&                   context,
    span<string_view> const&           source_names,
    span<input::text_source_t> const&  source_texts);

template <
    typename Container,
    typename = opencl::detail::enable_if_t<std::is_same<typename std::remove_reference<Container>::type::value_type, named_source_text_t>::value>>
source_t create(context_t const& context, Container&& named_sources);

template <template <typename> class Container>
source_t create(
    context_t const&                        context,
    Container<input::text_source_t> const&  source_texts);

// TODO: Try to avoid having this extra ctor explicitly; perhaps by generalizing the
// previous ctor to take a container of anything that's convertible into an
// input::text_source_t?
template <template <typename> class Container>
source_t create(
    context_t const&               context,
    Container<char const*> const&  source_texts);

source_t create(context_t const& context, named_source_text_t named_source);
source_t create(context_t const& context, string_view source_text);

} // namespace source

} // namespace program

} // namespace opencl

#endif // OPENCL_WRAPPERS_PROGRAM_SOURCE_HPP_
