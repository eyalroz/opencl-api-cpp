
#ifndef OPENCL_WRAPPERS_IMPL_MEMORY_OBJECT_HPP_
#define OPENCL_WRAPPERS_IMPL_MEMORY_OBJECT_HPP_

#include "context.hpp"
#include "../memory_object.hpp"
#include "info.hpp"

namespace opencl {

inline char const* info::traits_t<info::memory_object>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case CL_MEM_TYPE:                      return "type";
    case CL_MEM_FLAGS:                     return "flags";
    case CL_MEM_SIZE:                      return "size";
    case CL_MEM_HOST_PTR:                  return "host-side pointer";
    case CL_MEM_MAP_COUNT:                 return "mapping count";
    case CL_MEM_REFERENCE_COUNT:           return "reference count";
    case CL_MEM_CONTEXT:                   return "context";
#ifdef CL_VERSION_1_1
    case CL_MEM_ASSOCIATED_MEMOBJECT:      return "associated memory object";
    case CL_MEM_OFFSET:                    return "offset";
#endif
#ifdef CL_VERSION_2_0
    case CL_MEM_USES_SVM_POINTER:          return "uses shared virtual memory pointer";
#endif
#ifdef CL_VERSION_3_0
    case CL_MEM_PROPERTIES:                return "properties";
#endif
    default:                               return nullptr;
    }
}


namespace detail {

OCLW_DEFINE_HANDLE_TRAITS("memory object", memory::handle_t, clReleaseMemObject, clRetainMemObject);

} // namespace detail

namespace memory {

namespace detail {

inline char const* name_of(cl_mem_object_type raw_object_type)
{
    switch (raw_object_type) {
    case CL_MEM_OBJECT_BUFFER: return "buffer";
    case CL_MEM_OBJECT_IMAGE2D: return "2D image";
    case CL_MEM_OBJECT_IMAGE3D: return "3D image";
#ifdef CL_VERSION_1_2
    case CL_MEM_OBJECT_IMAGE2D_ARRAY: return "array of 2D images";
    case CL_MEM_OBJECT_IMAGE1D: return "1D image";
    case CL_MEM_OBJECT_IMAGE1D_ARRAY: return "array of 1D images";
    case CL_MEM_OBJECT_IMAGE1D_BUFFER: return "buffer-backed 1D image";
#ifdef CL_VERSION_2_0
    case CL_MEM_OBJECT_PIPE: return "pipe";
#endif // CL_VERSION_2_0
#endif // CL_VERSION_1_2
    default: return nullptr;
    }
}

inline cl_mem_object_type raw_type_of(handle_t handle)
{
    return info::get_scalar<CL_MEM_TYPE>(handle);
}

inline cl_mem_object_type raw_type_of(object_t const& object)
{
    return raw_type_of(object.handle());
}

inline bool is_buffer(handle_t handle)
{
    return raw_type_of(handle) == CL_MEM_OBJECT_BUFFER;
}

inline bool is_pipe(handle_t handle)
{
    return raw_type_of(handle) == CL_MEM_OBJECT_PIPE;
}

inline bool is_pipe(object_t const& object)
{
    return is_pipe(object.handle());
}

inline bool is_image(cl_mem_object_type raw_type)
{
    static constexpr auto raw_image_types = {
        CL_MEM_OBJECT_IMAGE1D,
        CL_MEM_OBJECT_IMAGE2D,
        CL_MEM_OBJECT_IMAGE3D,
        CL_MEM_OBJECT_IMAGE1D_BUFFER
    };
    auto iter = std::find(raw_image_types.begin(), raw_image_types.end(), raw_type);
    return iter != raw_image_types.end();
}

inline bool is_image(handle_t handle)
{
    return is_image(raw_type_of(handle));
}

inline bool is_image_array(handle_t handle)
{
    auto raw_type = raw_type_of(handle);
    return (raw_type == CL_MEM_OBJECT_IMAGE1D_ARRAY or raw_type == CL_MEM_OBJECT_IMAGE2D_ARRAY);
}

inline flags_t make_host_access_flags(access_kind_t kind)
{
    switch (kind) {
        // Note: when no flags are set, buffers can be both "read by" the host
        // (i.e. copied from device to host) and "written by" the host (i.e.
        // copied from host to device)
    case access_kind_t::read_and_write: return 0;
    case access_kind_t::read:           return CL_MEM_HOST_READ_ONLY;
    case access_kind_t::write:          return CL_MEM_HOST_WRITE_ONLY;
    case access_kind_t::none:           return CL_MEM_HOST_NO_ACCESS;
    default:
        throw std::invalid_argument("Unsupported kind of host access to buffer");
    }
}

inline flags_t make_kernel_access_flags(access_kind_t kind)
{
    switch (kind) {
    case access_kind_t::read_and_write: return CL_MEM_READ_WRITE;
    case access_kind_t::read:           return CL_MEM_READ_ONLY;
    case access_kind_t::write:          return CL_MEM_WRITE_ONLY;
    case access_kind_t::none:
        throw std::logic_error("Attempt to create kernel access flags for no read nor write access");
    default:
        throw std::invalid_argument("Unsupported kind of kernel access");
    }
}

inline flags_t make_flags(host_and_kernel_access_t access_spec) noexcept
{
    return make_host_access_flags(access_spec.host) | make_kernel_access_flags(access_spec.kernel);
}

inline host_and_kernel_access_t from_flags(flags_t flags) noexcept
{
    host_and_kernel_access_t result;
    result.host = flags & CL_MEM_HOST_WRITE_ONLY ?
        access_kind_t::write :
        flags & CL_MEM_HOST_READ_ONLY ?
            access_kind_t::read :
            access_kind_t::none;

    result.kernel = flags & CL_MEM_READ_WRITE ? access_kind_t::read_and_write  :
        flags & CL_MEM_WRITE_ONLY ?
            access_kind_t::write :
            flags & CL_MEM_READ_ONLY ?
                access_kind_t::read :
                access_kind_t::none;
    return result;
}

inline object_t wrap_using_context(handle_t handle, context_t const& context, bool owning = is_owning) noexcept
{
    return wrap(context.platform_handle(), context.handle(), handle, owning);
}

} // namespace detail

inline bool is_buffer(object_t const& object)
{
    return detail::is_buffer(object.handle());
}

inline bool is_image(object_t const& object)
{
    return detail::is_image(object.handle());
}

inline bool is_image_array(object_t const& object)
{
    return detail::is_image_array(object.handle());
}

inline host_and_kernel_access_t::operator flags_t() const { return detail::make_flags(*this); }

inline object_t wrap(
    platform::handle_t platform_handle,
    context::handle_t  context_handle,
    memory::handle_t   handle,
    bool               owning) noexcept
{
    return object_t(platform_handle, context_handle, handle, owning);
}

inline platform_t object_t::platform() const noexcept
{
    return platform::wrap(platform_handle_);
}

inline context_t object_t::context() const noexcept
{
    return context::wrap(platform_handle_, {}, context_handle_, is_not_owning);
}

inline size_t object_t::size_in_bytes() const
{
    return info::get_scalar<CL_MEM_SIZE>(handle());
}

inline void* object_t::data() const
{
    return info::get_scalar<CL_MEM_HOST_PTR>(handle());
}

inline size_t object_t::mapping_count() const
{
    return info::get_scalar<CL_MEM_MAP_COUNT>(handle());
}

inline size_t object_t::reference_count() const
{
    return info::get_scalar<CL_MEM_REFERENCE_COUNT>(handle());
}

inline optional<object_t> object_t::parent() const
{
    auto parent_handle = info::get_scalar<CL_MEM_ASSOCIATED_MEMOBJECT>(handle());
    if (parent_handle == nullptr) {
        return nullopt;
    }
    return wrap(platform_handle_, context_handle_, parent_handle, is_not_owning);
}

inline optional<off_t> object_t::offset_in_parent() const
{

    // Not checking whether there's actually a parent; if there isn't - 0 is returned. But
    // I don't like doing that...
    return info::get_scalar<CL_MEM_OFFSET>(handle());
}

inline bool object_t::is_shared_virtual_mem() const
{
    return info::get_scalar<CL_MEM_USES_SVM_POINTER>(handle());
}

inline bool operator==(object_t const& lhs, object_t const& rhs) noexcept
{
    // TODO: Is it not sufficient to merely compare the handles? Handles should
    // be unique after all
    return lhs.platform_handle() == rhs.platform_handle() and
        lhs.context_handle() == rhs.context_handle() and
        lhs.handle() == rhs.handle();
}

} // namespace memory

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_MEMORY_OBJECT_HPP_
