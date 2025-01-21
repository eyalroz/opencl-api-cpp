
#ifndef OPENCL_WRAPPERS_IMPL_IMAGE_HPP_
#define OPENCL_WRAPPERS_IMPL_IMAGE_HPP_

#include "buffer.hpp"
#include "memory_object.hpp"
#include "../identify.hpp"
#include "../image.hpp"
#include "../buffer.hpp"

#include <climits>

namespace opencl {

inline char const* info::traits_t<info::image>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case  CL_IMAGE_FORMAT:                    return "element format";
    case  CL_IMAGE_ELEMENT_SIZE:              return "element size";
    case  CL_IMAGE_ROW_PITCH:                 return "row pitch";
    case  CL_IMAGE_SLICE_PITCH:               return "slice pitch";
    case  CL_IMAGE_WIDTH:                     return "width";
    case  CL_IMAGE_HEIGHT:                    return "height";
    case  CL_IMAGE_DEPTH:                     return "depth";
#ifdef CL_VERSION_1_2
    case  CL_IMAGE_ARRAY_SIZE:                return "array size";
    case  CL_IMAGE_BUFFER:                    return "buffer";
    case  CL_IMAGE_NUM_MIP_LEVELS:            return "number of mip levels";
    case  CL_IMAGE_NUM_SAMPLES:               return "number of samples";
#endif
    default: return nullptr;
    }
}

namespace image {
namespace detail {

// TODO: Consider defining an image element size type
inline optional<size_t> overall_size(bit_decomposition_t decomposition) noexcept
{
    static const size_t sizes[] = {
        0, // default
        1, // _5_6_5
        2, // x_5_5_5
        4, // x_10_10_10
        4, // _10_10_10_2
        4, // _2_10_10_10
#ifdef cl_ext_image_unsigned_10x6_12x4_14x2
        0, // _top_10_each
        0, // _top_12_each
        0, // _top_14_each
#endif // cl_ext_image_unsigned_10x6_12x4_14x2
    };
    auto result_ = sizes[opencl::detail::to_underlying(decomposition)];
    return (result_ == 0) ? nullopt : optional<size_t>(result_);
}

} // namespace detail

inline bool operator==(type_descriptor_t const& lhs, type_descriptor_t const& rhs) noexcept
{
    return
        lhs.normalized     == rhs.normalized and
        lhs.signed_        == rhs.signed_ and
        lhs.floating_point == rhs.floating_point and
        lhs.size           == rhs.size;
}

inline size_t format_t::size() const noexcept
{
    return detail::overall_size(bit_decomposition).value_or(num_channels() * channel_format.size);
}

inline channel::count_t format_t::num_channels() const noexcept
{
    return channel::cardinality(channel_order);
}

inline channel::count_t num_channels(channel::order_t order) noexcept
{
    // using ord = opencl::image::channel::order_t;
    switch (opencl::detail::to_underlying(order)) {
    case CL_R:
    case CL_A:
    case CL_INTENSITY:
    case CL_LUMINANCE:
#ifdef CL_VERSION_2_0
    case CL_DEPTH:
#endif
        return 1;
    case CL_RG:
    case CL_RA:
#ifdef CL_VERSION_1_1
    case CL_Rx:
#endif
        return 2;
    case CL_RGB:
#ifdef CL_VERSION_1_1
    case CL_RGx:
#endif
#ifdef CL_VERSION_2_0
    case CL_sRGB:
#endif
        return 3;
    case CL_RGBA:
    case CL_BGRA:
    case CL_ARGB:
#ifdef CL_VERSION_1_1
    case CL_RGBx:
#endif
#ifdef CL_VERSION_2_0
    case CL_sRGBx:
    case CL_sRGBA:
    case CL_sBGRA:
    case CL_ABGR:
#endif
        return 4;
    default:
        // should not be possible to get here
        std::terminate();
    }
}

// TODO: Move this out of image:: ?
inline bool validate(format_t const&)
{
    // optional<size_t> required_bit_width;
    //
    // switch (format.bit_decomposition) {
    // case bit_decomposition_t::x_5_5_5:
    // }
    return true;
}

/// @note a 1D image may already have a row pitch
inline optional<dimension_t> layout_t::row_pitch() const noexcept
{
    return pitches.size() >= 1 ? optional<dimension_t>{pitches[0]} : nullopt;
}

inline optional<dimension_t> layout_t::slice_pitch() const noexcept
{
    return pitches.size() >= 2 ? optional<dimension_t>{pitches[1]} : nullopt;
}


namespace detail {

using raw_to_articulated_format_mapping_t = struct {
    cl_channel_type raw; // this is the key, the other fields are the values
    bit_decomposition_t decomposition;
    type_descriptor_t type_descriptor; // either the type for the entire element or for each channel separately
};

// I wish we had a nice static map in C++...
static span<raw_to_articulated_format_mapping_t> raw_format_info_map()
{
    using bd = bit_decomposition_t;
    enum { normalized = true, unnormalized = false };
    enum { fp = true, int_ = false };
    enum { signed_ = true, unsigned_ = false };

    static raw_to_articulated_format_mapping_t raw_type_info_[] = {
        { CL_SNORM_INT8,            bd::separate_,      { normalized,   signed_,     int_, 1 } },
        { CL_SNORM_INT16,           bd::separate_,      { normalized,   signed_,     int_, 2 } },
        { CL_UNORM_INT8,            bd::separate_,      { normalized,   unsigned_,   int_, 1 } },
        { CL_UNORM_INT16,           bd::separate_,      { normalized,   unsigned_,   int_, 2 } },
        { CL_UNORM_SHORT_565,       bd::_5_6_5,         { normalized,   unsigned_,   int_, 2 } },
        { CL_UNORM_SHORT_555,       bd::x_5_5_5,        { normalized,   unsigned_,   int_, 2 } },
        { CL_UNORM_INT_101010,      bd::x_10_10_10,     { normalized,   unsigned_,   int_, 4 } },
#ifdef CL_VERSION_2_1
        { CL_UNORM_INT_101010_2,   bd::_10_10_10_2,     { unnormalized, unsigned_,   int_, 4 } },
#endif
        { CL_SIGNED_INT8,           bd::separate_,      { unnormalized, signed_,     int_, 1 } },
        { CL_SIGNED_INT16,          bd::separate_,      { unnormalized, signed_,     int_, 2 } },
        { CL_SIGNED_INT32,          bd::separate_,      { unnormalized, signed_,     int_, 4 } },
        { CL_UNSIGNED_INT8,         bd::separate_,      { unnormalized, unsigned_,   int_, 1 } },
        { CL_UNSIGNED_INT16,        bd::separate_,      { unnormalized, unsigned_,   int_, 2 } },
        { CL_UNSIGNED_INT32,        bd::separate_,      { unnormalized, unsigned_,   int_, 4 } },
        { CL_HALF_FLOAT,            bd::separate_,      { unnormalized, signed_,     fp,   2 } },
        { CL_FLOAT,                 bd::separate_,      { unnormalized, signed_,     fp,   4 } },
#ifdef cl_ext_image_unsigned_10x6_12x4_14x2
        { CL_UNSIGNED_INT10X6_EXT,  bd::_top_10_each,   { unnormalized, unsigned_,   int_, 2 } },
        { CL_UNSIGNED_INT12X4_EXT,  bd::_top_12_each,   { unnormalized, unsigned_,   int_, 2 } },
        { CL_UNSIGNED_INT14X2_EXT,  bd::_top_14_each,   { unnormalized, unsigned_,   int_, 2 } },
        { CL_UNORM_INT10X6_EXT,     bd::_top_10_each,   { normalized,   unsigned_,   int_, 2 } },
        { CL_UNORM_INT12X4_EXT,     bd::_top_12_each,   { normalized,   unsigned_,   int_, 2 } },
        { CL_UNORM_INT14X2_EXT,     bd::_top_14_each,   { normalized,   unsigned_,   int_, 2 } },
#endif // cl_ext_image_unsigned_10x6_12x4_14x2
    };
    // TODO: Why can't I just pass the array to the span ctor?
    return {raw_type_info_, sizeof(raw_type_info_) / sizeof(raw_to_articulated_format_mapping_t)};
}

inline cl_channel_type raw_channel_type_for(format_t const& format)
{
    auto const rti = raw_format_info_map();
    auto predicate = [&](raw_to_articulated_format_mapping_t const& entry) {
        return entry.decomposition == format.bit_decomposition and entry.type_descriptor == format.channel_format;
    };
    auto iter = std::find_if(rti.begin(), rti.end(), predicate);
    if (iter != rti.end()) {
        return iter->raw;
    }
    throw std::invalid_argument("No known raw OpenCL image element data type for the specified format");
}

inline cl_image_format raw_element_format_for(format_t const& format)
{
    cl_image_format result;
    result.image_channel_data_type = raw_channel_type_for(format);
    result.image_channel_order = opencl::detail::to_underlying(format.channel_order);
    return result;
}

inline std::pair<bit_decomposition_t, type_descriptor_t>
format_components_for(cl_image_format raw_format)
{
    auto const raw_format_info_map_ = raw_format_info_map();
    auto pred = [&](raw_to_articulated_format_mapping_t const& entry) -> bool{ return entry.raw == raw_format.image_channel_data_type; };
    auto iter = std::find_if(raw_format_info_map_.begin(), raw_format_info_map_.end(), pred);
    if (iter != raw_format_info_map_.end()) {
        return { iter->decomposition, iter->type_descriptor };
    }
    throw std::invalid_argument("Unknown raw OpenCL image format");
}

inline cl_mem_object_type raw_image_type(dimensions_t const& single_image_dims, bool array, bool backed_by_buffer)
{
    // static constexpr auto one_is_not_trivial { false };
    // TODO: Are we properly handling the case of 0's in the image dims? What about 1's?
    auto dimensionality = single_image_dims.dimensionality();
    switch (dimensionality) {
    case 3: {
        if (array) { throw std::invalid_argument("OpenCL does not support arrays of 3D images"); }
        return CL_MEM_OBJECT_IMAGE3D;
    }
    case 2: return array ? CL_MEM_OBJECT_IMAGE2D_ARRAY : CL_MEM_OBJECT_IMAGE2D;
    case 1: return array ?
        CL_MEM_OBJECT_IMAGE1D_ARRAY :
        backed_by_buffer ? CL_MEM_OBJECT_IMAGE1D_BUFFER : CL_MEM_OBJECT_IMAGE1D;
    default: throw std::invalid_argument("No valid OpenCL image type when the (single) image dimensionality is "
        + std::to_string(dimensionality));
    }
}

inline void validate_layout(layout_t const& layout, optional<size_t> array_size)
{
    if (array_size) {
        if (*array_size == 0) {
            throw std::invalid_argument("An image array must have a positive size");
        }
        if (layout.dimensionality() == max_image_dimensions) {
            throw std::invalid_argument("Attempt to create an array of " + std::to_string(layout.dimensionality())
                + "-dimensional images, while OpenCL image arrays can only hold images of dimensionality "
                "up to " + std::to_string(max_image_dimensions - 1));
        }
    }
    else {
        if (layout.dimensionality() > max_image_dimensions) {
            throw std::invalid_argument("Attempt to create a " + std::to_string(layout.dimensionality())
                + "-dimensional images, while OpenCL only supports images of up to "
                + std::to_string(max_image_dimensions) + " dimensions");
        }
    }
    // TODO: More checks
    // TODO: We can't validate the correspondence of the backing memory object to the image type, e.g.:
    // "mem_object can be an image object if image_type is CL_MEM_OBJECT_IMAGE2D [21]."
}

// Note: This is not a unary cast member of format_t, as I don't want to the user
// to know or care about it. They can find this function for sure, but only if
// they explicitly look for it.
inline cl_image_desc raw_descriptor(
    layout_t const& single_image_layout,
    optional<size_t> array_size,
    memory::handle_t backing_object = nullptr)
{
    validate_layout(single_image_layout, array_size); // and ensure that if it's an array, there are
    cl_image_desc result;
    result.image_type = raw_image_type(single_image_layout.extents, array_size.has_value(), backing_object != nullptr);
    result.image_width = single_image_layout.extents.safe_width();
    result.image_height = single_image_layout.extents.safe_height();
    result.image_depth = single_image_layout.extents.safe_depth();
    result.image_array_size = array_size.value_or(0);
    result.image_row_pitch = single_image_layout.pitches.size() > 0 ? single_image_layout.pitches[0] : 0;
    result.image_slice_pitch = single_image_layout.pitches.size() > 1 ? single_image_layout.pitches[1] : 0;
    result.num_mip_levels = single_image_layout.num_mipmap_levels;
    result.num_samples = 0; // OpenCL reserves this for future use it seems
#if CL_VERSION_2_0
#if __CL_HAS_ANON_UNION__
    result.mem_object = backing_object;
#else
    result.buffer = backing_object; // shameless punning
#endif
#else
    if (backing_object) { throw std::invalid_argument("Backing objects are not supported with this version of OpenCL"); }
#endif
    return result;
}

// TODO: Can't this be avoided in favor of other implementations?
inline dimensionality_t effective_dimensionality_of(dimension_t width, dimension_t height, dimension_t depth, bool one_is_trivial = true)
{
    dimension_t max_trivial = one_is_trivial ? 1 : 0;
    bool trivial[3] = { width <= max_trivial, height <= max_trivial, depth <= max_trivial };
    if ((trivial[0] and not trivial[1]) or
        (trivial[1] and not trivial[2]))
    {
        throw std::invalid_argument("image dimensions with a trivial dimension following a non-trivial one");
    }
    return 3 - trivial[0] + trivial[1] + trivial[2];
}

inline dimensionality_t effective_dimensionality_of(dimensions_t dims, bool one_is_trivial = true)
{
    return effective_dimensionality_of(
        dims.size() > 0 ? dims[0] : 0,
        dims.size() > 1 ? dims[1] : 0,
        dims.size() > 2 ? dims[2] : 0,
        one_is_trivial);
}

inline image_t wrap_using_context(
    context_t const& context,
    handle_t handle,
    optional<spec_t> spec,
    bool owning = is_owning) noexcept
{
    return wrap(context.platform_handle(), context.handle(), handle, std::move(spec), owning);
}

struct creation_result_t {
    handle_t handle;
    status_t status;
    char const* api_function_name;
};

#ifdef CL_VERSION_1_2
inline creation_result_t create_single_or_array_with_raw_args(
    context::handle_t       context_handle,
    memory::flags_t         flags,
    cl_image_format const & raw_format,
    cl_image_desc const &   raw_descriptor,
    void*                   host_ptr)
{
    creation_result_t result;
    result.api_function_name = "clCreateImage";
    result.handle = clCreateImage(context_handle, flags, &raw_format, &raw_descriptor, host_ptr, &result.status);
    return result;
}

#else // CL_VERSION_1_2

inline creation_result_t create_with_raw_args(
    context::handle_t       context_handle,
    memory::flags_t         flags,
    cl_image_format const & raw_format,
    dimensions_t            dimensions,
    pitches_t               pitches,
    void*                   host_ptr)
{
    creation_result_t result;
    // TODO: Validate the dimensions - what about 0's? what about 1's?
    // TODO: Make sure the pitches match the dimensions
    switch (dimensions.size()) {
    case 2:
        result.api_function_name = "clCreateImage2D";
        result.handle = clCreateImage2D(
            context_handle, flags, &raw_format, dimensions[0], dimensions[1], pitches[0], host_ptr, &result.status);
    case 3:
        result.api_function_name = "clCreateImage3D";
        result.handle = clCreateImage3D(context_handle, flags, &raw_format,
            dimensions[0], dimensions[1], dimensions[2],
            pitches[0], pitches[1], host_ptr, &result.status);
    default:
        throw std::invalid_argument("Unsupported dimensionality for an OpenCL image:" + std::to_string(dimensions.size()));
    }
}

#endif // CL_VERSION_1_2

inline handle_t create_single_image_or_array(
    context_t const&          context,
    spec_t const&             spec,
    host_and_kernel_access_t  access_spec,
    optional<size_t>          num_array_images)
{
    enum { one_is_not_trivial = false };
    auto dims = spec.layout.extents;
    if (dims.size() > max_image_dimensions) {
        throw std::invalid_argument(std::to_string(dims.size()) + " dimensions specified for a prospective "
            "image, over the supported maximum of " + std::to_string(max_image_dimensions));
    }
    auto flags = memory::detail::make_flags(access_spec);
    auto no_host_ptr = nullptr;
    auto raw_format = raw_element_format_for(spec.element_format);
    auto backing_object_handle = spec.backing_object ? spec.backing_object->handle() : nullptr;
    auto raw_descriptor = detail::raw_descriptor(spec.layout, num_array_images, backing_object_handle);
#if CL_VERSION_1_2
    auto creation_result = create_single_or_array_with_raw_args(
        context.handle(), flags, raw_format, raw_descriptor, no_host_ptr);
    throw_if_error_lazy(creation_result.status, creation_result.api_function_name,
        std::string{"Creating a "} + (num_array_images ? "array of" : "")
        + std::to_string(detail::effective_dimensionality_of(dims, one_is_not_trivial))
        + "-dimensional image" + (num_array_images ? "s" : "") + " in " + opencl::detail::identify(context));
#else
    if (num_array_images) {
        throw std::runtime_error("Image arrays are not supported with OpenCL versions under 1.2");
    }
    auto creation_result = detail::create_with_raw_args(
        context.handle(), flags, raw_format, spec.layout.extents.data(), spec.layout.pitches.data(), no_host_ptr);
    throw_if_error_lazy(creation_result.status, creation_result.api_function_name,
        "Creating a " + (array ? "array of" : "") + std::to_string(detail::effective_dimensionality_of(dims, one_is_not_trivial))
        + "-dimensional image" + (array ? "s" : "") + " in " + opencl::detail::identify(context));
#endif
    return creation_result.handle;
}

} // namespace detail

inline size_t layout_t::storage_size_in_elements() const noexcept
{
    return std::accumulate(pitches.begin(), pitches.end(), extents.back(), std::multiplies<size_t>());
}

inline size_t spec_t::size_in_bytes() const noexcept
{
    return element_format.size() * layout.storage_size_in_elements();
}

inline dimensionality_t effective_dimensionality_of(dimensions_t const& dimensions, bool one_is_trivial = true)
{
    if (dimensions.size() > 3) {
        throw std::invalid_argument("Received an image dimensions value with more than 3 nominal dimensions");
    }
    auto dim_with_fallback = [&](std::size_t idx) { return dimensions.size() > idx ? dimensions[idx] : 0; };
    return detail::effective_dimensionality_of(
        dim_with_fallback(0), dim_with_fallback(1), dim_with_fallback(2), one_is_trivial);
}

inline image_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    memory::handle_t    handle,
    optional<spec_t>    spec,
    bool                owning) NOEXCEPT_IF_NDEBUG
{
    return { platform_handle, context_handle, handle, std::move(spec), owning };
}

inline image_t create(
    context_t const&          context,
    spec_t&&                  spec,
    host_and_kernel_access_t  access_spec)
{
    auto result_handle = detail::create_single_image_or_array(context, spec, access_spec, nullopt);
    return detail::wrap_using_context(context, result_handle, std::move(spec), is_owning);
}

inline image_t create(
    context_t const&          context,
    layout_t const&           layout,
    format_t const&           element_format,
    host_and_kernel_access_t  access_spec)
{
    auto no_backing_object = nullopt;
    spec_t spec { layout, element_format, no_backing_object };
    return create(context, std::move(spec), access_spec);
}

namespace detail {

inline dimensions_t get_dimensions(handle_t handle)
{
    auto width  = info::get_scalar<CL_IMAGE_WIDTH>(handle);
    auto height = info::get_scalar<CL_IMAGE_HEIGHT>(handle);
    auto depth  = info::get_scalar<CL_IMAGE_DEPTH>(handle);
    return dimensions_t{ width, height, depth };
}

inline pitches_t get_pitches(handle_t handle)
{
    static_assert(max_image_dimensions == 3, "Unexpected max_image_dimensions");
    pitches_t result{
        info::get_scalar<CL_IMAGE_ROW_PITCH>(handle),
        info::get_scalar<CL_IMAGE_SLICE_PITCH>(handle)
    };
    return result;
}

inline format_t make_format(cl_image_format const& raw_format)
{
    auto decomposition_and_type_descriptor = format_components_for(raw_format);
    format_t result;
    result.channel_order = static_cast<channel::order_t>(raw_format.image_channel_order);
    result.bit_decomposition = decomposition_and_type_descriptor.first;
    result.channel_format = decomposition_and_type_descriptor.second;
    return result;
}

inline format_t get_element_format(handle_t handle)
{
    cl_image_format raw_format = info::get_scalar<CL_IMAGE_FORMAT>(handle);
    return make_format(raw_format);
}

inline layout_t get_layout(handle_t handle)
{
    auto dims = get_dimensions(handle);
    auto pitches = get_pitches(handle);
    auto num_mip_levels = info::get_scalar<CL_IMAGE_NUM_MIP_LEVELS>(handle);
    return {  dims, pitches, num_mip_levels };
}

inline optional<memory::handle_t> associated_memory_object_handle_for(handle_t handle)
{
    auto memory_object_handle = info::get_scalar<CL_MEM_ASSOCIATED_MEMOBJECT>(handle);
    if (memory_object_handle == nullptr) {
        return nullopt;
    }
    return memory_object_handle;
}

inline spec_t get_spec(platform::handle_t platform_handle, context::handle_t context_handle, handle_t handle)
{
    auto format = get_element_format(handle);
    auto layout = get_layout(handle);
    auto maybe_memobj_handle = associated_memory_object_handle_for(handle);
    auto maybe_memobj = maybe_memobj_handle ?
        optional<memory::object_t>{memory::wrap(platform_handle, context_handle, *maybe_memobj_handle, is_not_owning)} : nullopt;
    return spec_t{ std::move(layout), format, std::move(maybe_memobj) };
}

inline spec_t get_spec(handle_t handle)
{
    auto context_handle = info::get_scalar<CL_MEM_CONTEXT>(handle);
    auto platform_handle = context::detail::get_platform_handle(context_handle);
    return get_spec(platform_handle, context_handle, handle);
}

} // namespace detail

} // namespace image

inline image::spec_t const& image_t::spec() const
{
    if (not spec_) {
        spec_ = image::detail::get_spec(platform_handle_, context_handle_, handle_);
    }
    return spec_.value();
}

#if CL_VERSION_1_1
inline optional<buffer_t> image_t::associated_buffer() const
{
    auto memory_object = associated_memory_object();
    return memory_object and memory::is_buffer(*memory_object) ?
        optional<buffer_t>{buffer::wrap(platform_handle_, context_handle_, memory_object->handle(), is_not_owning)} :
        nullopt;
}
#endif // CL_VERSION_1_1

inline optional<dimension_t> image_t::row_pitch() const
{
    return dimensionality() >= 2 ? optional<dimension_t>{spec().layout.pitches[0]} : nullopt;
}

inline optional<dimension_t> image_t::slice_pitch() const
{
    return dimensionality() >= 3 ? optional<dimension_t>{spec().layout.pitches[1]} : nullopt;
}

inline image::dimensions_t image_t::dimensions() const
{
    return spec().layout.extents;
}

inline dimensionality_t image_t::dimensionality(bool one_is_trivial) const
{
    auto dims = spec().layout.extents;
    return image::detail::effective_dimensionality_of(dims, one_is_trivial);
}

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
inline image_t upcast<image_t>(memory::object_t const& memory_object)
{
    if (memory::is_image(memory_object)) {
        return image::wrap(
            memory_object.platform_handle(), memory_object.context_handle(), memory_object.handle(),
            nullopt, is_not_owning);
    }
    throw std::invalid_argument("Attempt to upcast a memory object, which is not an image, into an image_t");
}

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

#endif // OPENCL_WRAPPERS_IMPL_IMAGE_HPP_
