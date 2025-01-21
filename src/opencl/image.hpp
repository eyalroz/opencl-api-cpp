#ifndef OPENCL_WRAPPERS_IMAGE_HPP_
#define OPENCL_WRAPPERS_IMAGE_HPP_

#include "types.hpp"
#include "impl/memory_object.hpp"
#include "util/optional.hpp"

namespace opencl {

namespace image {

/// Decomposition of the image-element value's bits among the
/// different channels
enum class bit_decomposition_t {
    /// Instead of a single value for all channels, each
    /// channel has its own typed value
    separate_typed_element_per_channel,
    default_ = separate_typed_element_per_channel,
    separate_ = separate_typed_element_per_channel,

    // TODO: Is the bit decomposition order low-to-high, or high-to-low? :-(
    // TODO: Validation function for matching channel order and this value

    /// Red 5 bits, Green 6 bits, Blue 5 bits; channel order must be RGB or RGBx
    _5_6_5,
    /// Dummy 1 bit, Red 5 bits, Green 5 bits, Blue 5 bits; channel order must be RGB or RGBx
    x_5_5_5,
    /// Dummy 2 bits, Red 10 bits, Green 10 bits, Blue 10 bits; channel order must be RGB or RGBx
    x_10_10_10,
    /// Red 10 bits, Green 10 bits, Blue 10 bits, Alpha 2 bits; channel order must be RGBA
    _10_10_10_2,
    /// Red 10 bits, Green 10 bits, Blue 10 bits, Dummy 2 bits; channel order must be ABGR
    _2_10_10_10,
#ifdef cl_ext_image_unsigned_10x6_12x4_14x2
    /// Each channel is 16 bits, and the top 10 are used for the value; channel order must be CL_R, CL_RG, or CL_RGBA.
    _top_10_each,
    /// Each channel is 16 bits, and the top 12 are used for the value; channel order must be CL_R, CL_RG, or CL_RGBA.
    _top_12_each,
    /// Each channel is 16 bits, and the top 14 are used for the value; channel order must be CL_R, CL_RG, or CL_RGBA.
    _top_14_each,
#endif
};

namespace channel {

enum class order_t : cl_channel_order {
    red   = CL_R,                           R = red,
    alpha = CL_A,                           A = alpha,
    red_green = CL_RG,                      RG = red_green,
    red_alpha = CL_RA,                      RA = red_alpha,
    red_green_blue = CL_RGB,                RGB = red_green_blue,
    red_green_blue_alpha = CL_RGBA,         RGBA = red_green_blue_alpha,
    blue_green_red_alpha = CL_BGRA,         BGRA = blue_green_red_alpha,
    alpha_red_green_blue = CL_ARGB,         ARGB = alpha_red_green_blue,
    intensity = CL_INTENSITY,
    luminance = CL_LUMINANCE,
#ifdef CL_VERSION_1_1
    red_ignore = CL_Rx,                     Rx = red_ignore,
    red_green_ignore = CL_RGx,              RGx = red_green_ignore,
    red_green_blue_ignore = CL_RGBx,        RGBx = red_green_blue_ignore,
#endif
#ifdef CL_VERSION_2_0
    depth = CL_DEPTH,
    sRGB_red_green_blue = CL_sRGB,          sRGB_RGB  = sRGB_red_green_blue,
    sRGB_red_green_blue_ignore = CL_sRGBx,  sRGB_RGBx = sRGB_red_green_blue_ignore,
    sRGB_red_green_blue_alpha = CL_sRGBA,   sRGB_RGBA = sRGB_red_green_blue_alpha,
    sRGB_blue_green_red_alpha = CL_sBGRA,   sRGB_BGRA = sRGB_blue_green_red_alpha,
    sRGB_alpha_blue_green_red = CL_ABGR,    sRGB_ABGR = sRGB_alpha_blue_green_red,
#endif
};

// Note that some channels may not be in use, but still take up space
count_t cardinality(order_t) noexcept;

} // namespace channel

// Note: This may be the type of one typed element - either in each
// of the channels, or for the entirety of the channels with some bit
// decomposition of the value
struct type_descriptor_t {
    bool normalized;
    bool signed_;
    bool floating_point;
    size_t size; // is this derived from the rest of the values here?
};

bool operator==(type_descriptor_t const&, type_descriptor_t const&) noexcept;

struct format_t {
    channel::order_t channel_order;
    bit_decomposition_t bit_decomposition;
    type_descriptor_t channel_format;

    /// @return the size of an image element in bytes
    size_t size() const noexcept;
    channel::count_t num_channels() const noexcept;
};

// Maybe just put this namespace inside memory? Or only use the memory namespace?
using memory::host_and_kernel_access_t;

struct spec_t {
    layout_t layout;
    format_t element_format;
    optional<memory::object_t> backing_object; // TODO: Can we even have this with OpenCL 1.0?

    size_t size_in_bytes() const noexcept;
    /// @note The volume is the _effective_ volume of image elements, not the space taken
    /// up considering the pitches
    size_t volume() const noexcept { return layout.volume(); }
    size_t element_size_in_bytes() const noexcept { return element_format.size(); }
    dimensionality_t dimensionality() const noexcept { return layout.dimensionality(); }
    // TODO: Perhaps add effective_dimensionality() ?
    optional<dimension_t> row_pitch() const noexcept { return layout.row_pitch(); }
    optional<dimension_t> slice_pitch() const noexcept { return layout.slice_pitch(); }
};

image_t wrap(
    platform::handle_t platform_handle,
    context::handle_t  context_handle,
    memory::handle_t   handle,
    optional<spec_t>   spec,
    bool               owning) NOEXCEPT_IF_NDEBUG;

// What about the storage? Make copy of? and so on?
image_t create(
    context_t const&          context,
    spec_t&&                  spec,
    host_and_kernel_access_t  access_spec = memory::full_access() );

image_t create_using(
    context_t const&          context,
    spec_t&&                  spec,
    memory::region_t          host_side_storage,
    host_and_kernel_access_t  access_spec = memory::full_access() );

buffer_t create_copy_of(
    context_t const&          context,
    spec_t&&                  spec,
    memory::region_t          region_to_copy,
    host_and_kernel_access_t  access_spec = memory::full_access() );

image_t create(
    context_t const&          context,
    layout_t const&           layout,
    format_t const&           element_format,
    host_and_kernel_access_t  access_spec = memory::full_access() );

// Note: Remember to verify the pitches are 0 in this case
image_t create_using(
    context_t const&          context,
    layout_t const&           layout,
    format_t const&           element_format,
    memory::region_t          host_side_storage,
    host_and_kernel_access_t  access_spec = memory::full_access() );

// Note: Remember to verify the pitches are 0 in this case
buffer_t create_copy_of(
    context_t const&          context,
    layout_t const&           layout,
    format_t const&           element_format,
    memory::region_t          region_to_copy,
    host_and_kernel_access_t  access_spec = memory::full_access() );

namespace array {

// What about the storage? Make copy of? and so on?
// TODO: With image arrays, the associated mem object backs the array, not individual images; yet -
// the element spec class has it. How should we handle this? For now, we'll just demand the element
// spec has no backing object
array_t create(
    context_t const&          context,
    size_t                    num_images,
    spec_t&&                  single_image_spec,
    host_and_kernel_access_t  access_spec = memory::full_access() );

array_t create(
    context_t const&            context,
    size_t                      num_images,
    layout_t const&             single_image_layout,
    format_t const&             element_format,
    optional<memory_object_t>&& backing_object,
    host_and_kernel_access_t    access_spec = memory::full_access() );

} // namespace array

} // namespace image

/**
 * @note This class is not intended to represent image arrays.
 */
class image_t : public memory::object_t {
public:
    using parent_type = memory::object_t;
    using parent_type::handle_type; // which is the same as image::handle_t

protected:
    mutable optional<image::spec_t> spec_;

public:
    friend image_t image::wrap(
        platform::handle_t       platform_handle,
        context::handle_t        context_handle,
        memory::handle_t         handle,
        optional<image::spec_t>  spec,
        bool                     owning) NOEXCEPT_IF_NDEBUG;

    image_t(
        platform::handle_t       platform_handle,
        context::handle_t        context_handle,
        memory::handle_t         handle,
        optional<image::spec_t>  spec,
        bool                     owning) NOEXCEPT_IF_NDEBUG
    : parent_type(platform_handle, context_handle, handle, owning), spec_(std::move(spec))
    {
#ifndef NDEBUG
        auto raw_type = memory::detail::raw_type_of(handle);
        if (not memory::detail::is_image(handle)) {
            throw std::invalid_argument("Attempt to construct a (non-array) image wrapper object using a"
                "raw memory object of type " + std::string{memory::detail::name_of(raw_type)});
        }
#endif
    }

    image::spec_t const& spec() const;
    image::layout_t const& layout() const { return spec().layout; }
    image::format_t const& element_format() const { return spec().element_format; }
#if CL_VERSION_1_1
    optional_ref<memory::object_t const> associated_memory_object() const { return spec().backing_object; }
    /// @return nullopt if this image was not created with a backing buffer
    optional<buffer_t> associated_buffer() const;
#endif

    /// Return size of each image element in bytes.
    size_t element_size_in_bytes() const { return spec().element_size_in_bytes(); }

    /**
     * Returns the pitch in bytes of a row (= scan-line) of elements of the image
     *
     * @note The image_row_pitch must be zero if the image was not copied from a
     * host memory region, was not created from an external memory handle,
     * and is not a 2D image created from a buffer.
     *
     * @todo perhaps make this optional, for those cases the row pitch has no
     * valid positive value?
     */
    optional<dimension_t> row_pitch() const;

    /**
     * Returns the slice pitch in bytes of a 2D slice for the 3D image object
     *
     * @todo perhaps make this optional, for those cases the slice pitch has no
     * valid positive value?
     */
    optional<dimension_t> slice_pitch() const;

    /**
     * Returns the dimensions with which the image was created.
     *
     * @note OpenCL images may have no more than three dimensions.
     */
    image::dimensions_t dimensions() const;

    /**
     * Returns the number of dimensions the image has.
     *
     * @param[in] one_is_trivial
     *     When false, the dimensions follow how OpenCL itself considers the image;
     *     and it does not care whether on a certain axis, the dimension is greater than 1,
     *     so that a 2x1x1 image would be considered 3-dimensional;
     *     when true, we consider a dimension of 1 as degenerate, so that a 2x1x1 image
     *     would be considered 1-dimensional.
     *
     * @note OpenCL images may have no more than three dimensions.
     */
    dimensionality_t dimensionality(bool one_is_trivial = true) const;
}; // class image_t

inline std::pair<queue::event_t, memory::region_t>
map(image_t const& image, image::mapping::spec_t const& spec, queue_t const& queue, bool blocking = is_non_blocking);

namespace image {

namespace array {

array_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    memory::handle_t    handle,
    size_t              num_images,
    optional<spec_t>&&  constituent_image_spec,
    bool                owning) NOEXCEPT_IF_NDEBUG;

} // namespace array


/**
 * @note This class is not intended to represent image arrays, although
 * technically you can always wrap an image array in an image_t. At a
 * future time, a separate class will be written for image arrays, similarly
 * to how we have different classes for different kinds of OpenCL programs.
 */
class array_t : public memory::object_t {
public:
    using parent_type = memory::object_t;
    using parent_type::handle_type; // which is the same as image::handle_t
    using value_type = image_t;

protected:
    size_t num_images_;
    mutable optional<spec_t> constituent_image_spec_;

    array_t(
        platform::handle_t platform_handle,
        context::handle_t  context_handle,
        memory::handle_t   handle,
        size_t             num_images,
        optional<spec_t>   constituent_image_spec,
        bool               owning) NOEXCEPT_IF_NDEBUG
        : parent_type(platform_handle, context_handle, handle, owning), num_images_(num_images),
            constituent_image_spec_(std::move(constituent_image_spec))
    {
#ifndef NDEBUG
        auto raw_type = memory::detail::raw_type_of(handle);
        if (not memory::detail::is_image_array(handle)) {
            throw std::invalid_argument("Attempt to construct a image array object using a"
                "raw memory object of type " + std::string{memory::detail::name_of(raw_type)});
        }
#endif
    }

public:
    // using parent_type::parent_type;

    friend array_t array::wrap(
        platform::handle_t  platform_handle,
        context::handle_t   context_handle,
        memory::handle_t    handle,
        size_t              num_images,
        optional<spec_t>&&  constituent_image_spec,
        bool                owning) NOEXCEPT_IF_NDEBUG;


    spec_t const& constituent_image_spec() const;
    size_t size() const noexcept { return num_images_; }
    size_t num_images() const noexcept { return size(); }
#if CL_VERSION_1_2
    optional<buffer_t> associated_buffer() const;
#endif

    /// Return size of each image element in bytes.
    /// TODO: Am I missing some pitche here?
    // size_t constituent_image_size() const { return constituent_image_spec().size_in_bytes() * num_images(); }

    /// Returns the row pitch in bytes of a row of elements of the image
    // optional<dimension_t> image_row_pitch() const { return constituent_image_spec().row_pitch(); }

    /// Returns the slice pitch in bytes of a 2D slice for the 3D image object or size
    /// of each image in a 1D or 2D image array
    // optional<dimension_t> image_slice_pitch() const { return constituent_image_spec().slice_pitch(); }

    /**
     * Returns the dimensions with which the image was created.
     *
     * @note OpenCL images may have no more than three dimensions.
     */
    // dimensions_t image_dimensions() const { return constituent_image_spec().layout.extents; }

    /**
     * Returns the number of dimensions the image has.
     *
     * @param[in] one_is_trivial
     *     When false, the dimensions follow how OpenCL itself considers the image;
     *     and it does not care whether on a certain axis, the dimension is greater than 1,
     *     so that a 2x1x1 image would be considered 3-dimensional;
     *     when true, we consider a dimension of 1 as degenerate, so that a 2x1x1 image
     *     would be considered 1-dimensional.
     *
     * @note OpenCL images may have no more than three dimensions.
     */
    // dimensionality_t image_dimensionality(bool one_is_trivial = true) const { return image_dimensions().dimensionality(); }
}; // class array_t

} // namespace image

template <> image_t upcast<image_t>(memory::object_t const& memory_object);
template <> image::array_t upcast<image::array_t>(memory::object_t const& memory_object);

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMAGE_HPP_
