
#ifndef OPENCL_WRAPPERS_IMPL_IMAGE_ARRAY_HPP_
#define OPENCL_WRAPPERS_IMPL_IMAGE_ARRAY_HPP_

#include "image.hpp"

namespace opencl {
namespace image {
namespace array {

inline array_t wrap(
    platform::handle_t platform_handle,
    context::handle_t  context_handle,
    memory::handle_t   handle,
    size_t             num_images,
    optional<spec_t>&& constituent_image_spec,
    bool               owning) NOEXCEPT_IF_NDEBUG
{
    return array_t{platform_handle, context_handle, handle, num_images, std::move(constituent_image_spec), owning};
}

inline array_t create(
    context_t const&          context,
    size_t                    num_images,
    spec_t&&                  single_image_spec,
    host_and_kernel_access_t  access_spec )
{
    // The hack here is, that we use the backing object for the entire array for making the OpenCL call,
    // not each individual image, even though that's what the single image spec supposedly signifiees
    auto result_handle = detail::create_single_image_or_array(context, single_image_spec, access_spec, num_images);
    return array::wrap(context.platform_handle(), context.handle(), result_handle, num_images, std::move(single_image_spec), is_owning);
}

inline array_t create(
    context_t const&             context,
    size_t                       num_images,
    layout_t const&              single_image_layout,
    format_t const&              element_format,
    optional<memory_object_t>&&  backing_object,
    host_and_kernel_access_t     access_spec)
{
    auto spec = spec_t { single_image_layout, element_format, std::move(backing_object) };
    return create(context, num_images, std::move(spec), access_spec);
}

} // namespace array

inline spec_t const& array_t::constituent_image_spec() const
{
    if (not constituent_image_spec_) {
        constituent_image_spec_ = detail::get_spec(handle_);
    }
    return *constituent_image_spec_;
}

inline optional<buffer_t> array_t::associated_buffer() const
{
    auto const& backing_object = constituent_image_spec().backing_object;
    if (backing_object and memory::detail::is_buffer(backing_object->handle())) {
        return buffer::wrap(platform_handle_, context_handle_, backing_object->handle(), is_not_owning);
    }
    return nullopt;
}

} // namespace image

template <>
inline image::array_t upcast<image::array_t>(memory::object_t const& memory_object)
{
    if (memory::is_image_array(memory_object)) {
        auto num_images = info::get_scalar<CL_IMAGE_ARRAY_SIZE>(memory_object.handle());
        return image::array::wrap(
            memory_object.platform_handle(), memory_object.context_handle(), memory_object.handle(),
            num_images, nullopt, is_not_owning);
    }
    throw std::invalid_argument("Attempt to upcast a memory object, which is not an image array, into an image::array_t");
}

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_IMAGE_ARRAY_HPP_
