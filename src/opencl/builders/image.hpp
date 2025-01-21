
#ifndef OPENCL_WRAPPERS_BUILDERS_IMAGE_HPP_
#define OPENCL_WRAPPERS_BUILDERS_IMAGE_HPP_

#include "../image.hpp"
#include "memory_object.hpp"

namespace opencl {

//TODO: Will we need these types elsewhere? I suspect that we might...

namespace builders {

// TODO: Add support for properties and clCreateBufferWithProperties
class image_t : public detail::memory_object_t<image_t> {
public:
    using parent_type = detail::memory_object_t<image_t>;
    using parent_type::parent_type;

protected:
    using flags_type = memory::flags_t; // which is also image::flags_t
    using dimension_type = image::dimension_t;
    using dimensions_type = image::dimensions_t;

    struct  {
        // optionalized components of format_t...
        optional<image::channel::order_t> order;
        optional<bool> normalized;
        optional<bool> signed_;
        optional<bool> floating_point;
        optional<size_t> size;
        image::bit_decomposition_t bit_decomposition { image::bit_decomposition_t::separate_typed_element_per_channel };
    } format_;

    memory::handle_t parent_image_ = nullptr;
    memory::handle_t parent_buffer_ = nullptr;

    // Note: The flags may indicate read-only access, but we use
    // a non-const pointer regardless, for simplicity of implementation
    // optional<memory::region_t> host_region_;

    // Notes:
    // - The OpenCL C API 'encodes' unused dimensions using a 0-value. We
    //   do not; nor do we accept this encoding from the user; nor do we let the
    //   user ask for it.
    // - Recall that some combinations of (engaged and unengaged) values here
    //   are acceptable during building, but not acceptable for finalization:
    //   having an engaged dimension follow an unengaged one and non-positive
    //   dimension values.
    struct {
        static_dynarray<optional<dimension_t>, image::max_image_dimensions>      extents;
        static_dynarray<optional<dimension_t>, image::max_image_dimensions - 1>  pitches;
        // TODO: Shouldn't we condition this on #if CL_VERSION_1_2 && defined(cl_khr_mipmap_image)
        optional<image::mipmap_level_count_t>  num_mipmap_levels;
    } layout_;

    void infer_missing();

public:

    image_t& extents(image::dimensions_t const&);
    image_t& dimensions(image::dimensions_t const&);
    image_t& pitches(image::pitches_t const&);
    image_t& num_mipmap_levels(image::mipmap_level_t);
    image_t& no_mipmap();
    image_t& parent_buffer(memory::handle_t);
    image_t& parent_buffer(opencl::buffer_t const&);
    image_t& parent_image(memory::handle_t);
    image_t& parent_image(opencl::image_t const&);
    image_t& layout(image::layout_t const&);

    image_t& element_channel_order(image::channel::order_t const&);
    image_t& channel_value_type_normalization(bool normalized);
    image_t& channel_values_are_normalized();
    image_t& channel_value_arent_normalized();
    image_t& channel_values_have_sign(bool signedness);
    image_t& channel_values_are_signed();
    image_t& channel_values_arent_unsigned();
    image_t& channel_value_floatness(bool floating_point);
    image_t& floating_point_channel_values();
    image_t& integral_channel_values();
    image_t& channel_value_size_bytes(size_t size);
    image_t& channel_value_format(image::type_descriptor_t const&);
    image_t& element_bit_decomposition(image::bit_decomposition_t bit_decomposition);
    image_t& no_element_bit_decomposition();
    image_t& format(image::format_t const&);

    image_t& width(image::dimension_t);
    image_t& height(image::dimension_t);
    image_t& depth(image::dimension_t);
    image_t& row_pitch(image::dimension_t);
    image_t& slice_pitch(image::dimension_t);
    image_t& limit_to_2D();
    image_t& limit_to_1D();


    opencl::image_t create();
    opencl::image_t build();

    // No support for sub-buffers for now (and even when we do
    // support them, it might not be here)

    void validate(bool ensure_sufficiency) const override;

}; // image_t

/// A slightly shorter-named construction idiom for the @ref image_t builder
inline image_t image() { return {}; }

} // namespace builders

} // namespace opencl

#endif // OPENCL_WRAPPERS_BUILDERS_IMAGE_HPP_
