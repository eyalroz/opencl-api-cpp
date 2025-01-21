
#ifndef OPENCL_WRAPPERS_IMPL_BUILDERS_IMAGE_HPP_
#define OPENCL_WRAPPERS_IMPL_BUILDERS_IMAGE_HPP_

#include "../../builders/image.hpp"
#include "opencl/impl/image.hpp"

namespace opencl {

namespace detail {

template <typename T> optional<T> maybe(T const& concrete) { return concrete; }
template <typename T> T           deref(optional<T> const& opt) { return *opt; }

template <typename T, size_t Capacity>
static_dynarray<optional<T>, Capacity> as_optionals(static_dynarray<T, Capacity> const& concrete)
{
    return to_static_dynarray<Capacity>(concrete, maybe<T>);
}

template <typename T, size_t Capacity, typename S = T>
static_dynarray<T, Capacity> concretize(static_dynarray<optional<T>, Capacity> const& optionals, S&& default_)
{
    auto deref_ = [&](optional<T> const& opt) -> T { return opt.value_or(static_cast<T>(default_)); };
    return to_static_dynarray<Capacity>(optionals, deref_);
}

template <typename T, size_t Capacity>
static_dynarray<T, Capacity> concretize(static_dynarray<optional<T>, Capacity> const& optionals)
{
    return to_static_dynarray<Capacity>(optionals, maybe);
}

} // namespace detail

namespace builders {

inline void image_t::infer_missing()
{
    assert(false && "WRITEME");
}

inline void image_t::validate(bool ensure_sufficiency) const
{
    (void) ensure_sufficiency;
    assert(false && "WRITEME");
}

inline opencl::image_t image_t::create()
{
    constexpr auto ensure_sufficiency { true };
    validate(ensure_sufficiency);
    infer_missing();

    // add methods to create this stuff (which can also be accessed from outside?
    image::spec_t spec;
    spec.element_format.channel_format = {
        *format_.normalized,
        *format_.signed_,
        *format_.floating_point,
        *format_.size,
    };
    spec.element_format.channel_order = *format_.order;
    spec.element_format.bit_decomposition = format_.bit_decomposition;

    spec.layout.extents = opencl::detail::concretize(layout_.extents, 0);
    spec.layout.pitches = opencl::detail::concretize(layout_.pitches, 0);
    spec.layout.num_mipmap_levels = layout_.num_mipmap_levels.value_or(image::default_mipmap_level_count);

    auto context = context::detail::by_handle(context_handle_, is_not_owning);

    auto access_spec = get_access_spec();

    return image::create(context, std::move(spec), access_spec);
}

inline opencl::image_t image_t::build() { return create(); }

inline image_t& image_t::extents(image::dimensions_t const& extents)
{
    layout_.extents = opencl::detail::as_optionals(extents);
    return *this;
}

inline image_t& image_t::dimensions(image::dimensions_t const& dimensions) { return extents(dimensions); }
inline image_t& image_t::pitches(image::pitches_t const& pitches)
{
    layout_.pitches = opencl::detail::as_optionals(pitches);
    return *this;
}

inline image_t& image_t::parent_buffer(memory::handle_t buffer_handle)
{
#ifndef NDEBUG
    if (not memory::detail::is_buffer(buffer_handle)) {
        throw std::invalid_argument("Attempt to use a non-buffer memory object as the parent buffer of an image to be built");
    }
#endif
    parent_buffer_ = buffer_handle;
    return *this;
}

inline image_t& image_t::parent_buffer(opencl::buffer_t const& buffer) { return parent_buffer(buffer.handle()); }

inline image_t& image_t::parent_image(memory::handle_t image_handle)
{
#ifndef NDEBUG
    if (not memory::detail::is_image(image_handle)) {
        throw std::invalid_argument("Attempt to use a non-image memory object as the parent image of an image to be built");
    }
#endif
    parent_image_ = image_handle;
    return *this;
}
inline image_t& image_t::parent_image(opencl::image_t const& image) { return parent_image(image.handle()); }


inline image_t& image_t::layout(image::layout_t const& layout)
{
    extents(layout.extents);
    pitches(layout.pitches);
    layout_.num_mipmap_levels = layout.num_mipmap_levels;
    return *this;
}

inline image_t& image_t::element_channel_order(image::channel::order_t const& order) { format_.order = order;  return *this; }
inline image_t& image_t::channel_value_type_normalization(bool normalized)
{
    format_.normalized = normalized;
    return *this;
}
inline image_t& image_t::channel_values_are_normalized() { return channel_value_type_normalization(true); }
inline image_t& image_t::channel_value_arent_normalized() { return channel_value_type_normalization(false); }
inline image_t& image_t::channel_values_have_sign(bool signedness)
{
    format_.signed_ = signedness;
    return *this;
}
inline image_t& image_t::channel_values_are_signed() { return channel_values_have_sign(true); }
inline image_t& image_t::channel_values_arent_unsigned() { return channel_values_have_sign(false); }
inline image_t& image_t::channel_value_floatness(bool floating_point)
{
    format_.floating_point = floating_point;
    return *this;
}
inline image_t& image_t::floating_point_channel_values() { return channel_value_floatness(true);}
inline image_t& image_t::integral_channel_values() { return channel_value_floatness(false); }
inline image_t& image_t::channel_value_size_bytes(size_t size) { format_.size = size; return *this; }
inline image_t& image_t::channel_value_format(image::type_descriptor_t const& value_format)
{
    channel_value_type_normalization(value_format.normalized);
    channel_values_have_sign(value_format.signed_);
    channel_value_floatness(value_format.floating_point);
    channel_value_size_bytes(value_format.size);
    return *this;
}
inline image_t& image_t::element_bit_decomposition(image::bit_decomposition_t bit_decomposition)
{
    format_.bit_decomposition = bit_decomposition;
    return *this;
}
inline image_t& image_t::no_element_bit_decomposition() { return element_bit_decomposition(image::bit_decomposition_t::separate_); }
inline image_t& image_t::format(image::format_t const& format)
{
    element_channel_order(format.channel_order);
    element_bit_decomposition(format.bit_decomposition);
    channel_value_format(format.channel_format);
    return *this;
}

inline image_t& image_t::width(image::dimension_t width) { *layout_.extents[0] = width; return *this; }
inline image_t& image_t::height(image::dimension_t height) { layout_.extents[1] = height; return *this;}
inline image_t& image_t::depth(image::dimension_t depth) { layout_.extents[2] = depth; return *this;}
inline image_t& image_t::row_pitch(image::dimension_t row_pitch) { layout_.pitches[0] = row_pitch; return *this; }
inline image_t& image_t::slice_pitch(image::dimension_t slice_pitch) { layout_.pitches[1] = slice_pitch; return *this; }
inline image_t& image_t::num_mipmap_levels(image::mipmap_level_t num_levels) { layout_.num_mipmap_levels = num_levels; return *this; }
inline image_t& image_t::no_mipmap() { layout_.num_mipmap_levels = nullopt; return *this; }
inline image_t& image_t::limit_to_2D() {  layout_.extents[2] = nullopt; return *this; }
inline image_t& image_t::limit_to_1D() { layout_.extents[1] = nullopt; return limit_to_2D(); }

// private methods

// inline void image_t::verify_host_region_size()
// {
//     // TODO: Support image arrays
//     if (not host_region_.size) { return; }
//     auto size_bytes = *host_region_.size;
//     auto element_size = get_element_size();
//     auto num_elements = size_bytes / element_size;
//     if (not host_region_.ptr) {
//         throw std::logic_error("A size was specified for the host region, but no pointer; this does not make sense");
//         // ... because we set the size for host point allocation if that's relevant
//     }
//     // At this point, we know we've been provided with a proper host-side region...
//     auto dimensionality = get_dimensionality();
//     verify_dimensionality();
//     struct { image::dimension_t row, slice; } effective_pitches = {
//         pitches_.row.value_or(*dimensions_[0]),
//         pitches_.slice.value_or(*dimensions_[1])
//     };
//     size_t required_num_elements = effective_pitches[0];
//     if (dimensionality == 2) { required_num_elements *= effective_pitches[1]; }
//     if (dimensionality == 3) { required_num_elements *= *dimensions_[2]; }
//     if (not num_elements >= required_num_elements) {
//         std::ostringstream oss;
//         oss << "Host region associated with a " << dimensionality << "D image is of size " << size_bytes << " bytes, "
//             << "fitting " << num_elements << " elements (of size " << element_size << "each) - less than "
//             << "row pitch " << effective_pitches[0];
//         if (dimensionality >= 2) { oss << " x slice pitch " << effective_pitches.slice; }
//         if (dimensionality == 3) { oss << " x depth " << *dimensions_[2]; }
//         oss << " elements required overall for the image";
//         throw runtime_error(status::named_t::invalid_value, oss.str());
//     }
// }

} // namespace builders

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_BUILDERS_IMAGE_HPP_
