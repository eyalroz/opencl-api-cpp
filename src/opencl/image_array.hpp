/**
 * @file
 *
 * @brief Definition of the @ref image::array_t class and declarations of
 * its named constructor idioms and other related functions.
 */
#ifndef OPENCL_WRAPPERS_IMAGE_ARRAY_HPP_
#define OPENCL_WRAPPERS_IMAGE_ARRAY_HPP_

#include "image.hpp"

namespace opencl {

namespace image {

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

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMAGE_ARRAY_HPP_
