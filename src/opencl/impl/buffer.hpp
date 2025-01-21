
#ifndef OPENCL_WRAPPERS_IMPL_BUFFER_HPP_
#define OPENCL_WRAPPERS_IMPL_BUFFER_HPP_

#include "../buffer.hpp"
#include "identify.hpp"
#include "memory_object.hpp"

namespace opencl {

namespace buffer {

namespace detail {

inline buffer_t wrap_using_context(memory::handle_t handle, context_t const& context, bool owning = is_owning) noexcept
{
    return wrap(context.platform_handle(), context.handle(), handle, owning);
}

// Can be invoked both for the case of USING_HOST_PTR and for plain copying
inline buffer_t create_with_host_side_storage(
    context_t const& context,
    memory::region_t host_side_storage,
    host_and_kernel_access_t access_spec,
    bool use_host_side_storage)
{
    cl_mem_flags flags = memory::detail::make_flags(access_spec) | (use_host_side_storage ? CL_MEM_USE_HOST_PTR : CL_MEM_COPY_HOST_PTR);
    status_t status { std::numeric_limits<status_t>::min() };
    auto buffer_handle = clCreateBuffer(context.handle(), flags, host_side_storage.size(), host_side_storage.data(), &status);
    throw_if_error_lazy(status, "clCreateBuffer",
        std::string("Creating a buffer ")
        + (use_host_side_storage ? "using " : "copy of ")
        + memory::detail::identify(host_side_storage)
        + " in " + opencl::detail::identify(context));
    return wrap_using_context(buffer_handle, context, is_owning);
}

} // namespace detail

inline buffer_t wrap(
    platform::handle_t platform_handle,
    context::handle_t  context_handle,
    memory::handle_t   handle,
    bool               owning) noexcept
{
    return { platform_handle, context_handle, handle, owning };
}

inline buffer_t create(
    context_t const& context,
    size_t size,
    host_and_kernel_access_t access_spec)
{
    cl_mem_flags flags = memory::detail::make_flags(access_spec);
    status_t status { CL_MEM_OBJECT_ALLOCATION_FAILURE };
    auto buffer_handle = clCreateBuffer(context.handle(), flags, size, nullptr, &status);
    throw_if_error_lazy(status, "clCreateBuffer", "Creating a buffer of size " + std::to_string(size) + " in "
        + opencl::detail::identify(context) );
    return detail::wrap_using_context(buffer_handle, context, is_owning);
}

inline buffer_t create_using(context_t const& context, memory::region_t host_side_storage, host_and_kernel_access_t access_spec)
{
    enum { use_host_side_storage = true };
    return detail::create_with_host_side_storage(context, host_side_storage, access_spec, use_host_side_storage);
}

inline buffer_t create_copy_of(context_t const& context, memory::region_t region_to_copy, host_and_kernel_access_t access_spec)
{
    enum { dont_use_host_side_storage = false };
    return detail::create_with_host_side_storage(context, region_to_copy, access_spec, dont_use_host_side_storage);
}

template <typename T>
buffer_t create_using(context_t const& context, span<T> host_side_storage, host_and_kernel_access_t access_spec)
{
    auto as_region = memory::region_t { host_side_storage.data(), host_side_storage.size() * sizeof(T) };
    return create_using(context, as_region, access_spec);
}

template <typename T>
buffer_t create_copy_of(context_t const& context, span<T> region_to_copy, host_and_kernel_access_t access_spec)
{
    auto as_region = memory::region_t { region_to_copy.data(), region_to_copy.size() * sizeof(T) };
    return create_using(context, as_region, access_spec);
}

inline buffer_t create_sub_buffer(
    buffer_t const& parent,
    memory::subregion_spec_t subregion,
    host_and_kernel_access_t access_spec)
{
    cl_mem_flags flags = memory::detail::make_flags(access_spec);
    status_t status;
    auto new_handle = clCreateSubBuffer(parent.handle(), flags, CL_BUFFER_CREATE_TYPE_REGION, &subregion, &status);
    throw_if_error_lazy(status, "clCreateSubBuffer", "Creating a sub-buffer of " + opencl::detail::identify(parent)
        + " of " + memory::detail::identify(subregion));
    return wrap(parent.platform_handle(), parent.context_handle(), new_handle, is_owning);
}

} // namespace buffer

inline optional<buffer_t> buffer_t::parent() const
{
    auto parent_handle = info::get_scalar<CL_MEM_ASSOCIATED_MEMOBJECT>(handle());
    if (parent_handle == nullptr) {
        return nullopt;
    }
    return buffer::wrap(platform_handle_, context_handle_, parent_handle, is_not_owning);
}

template <> inline buffer_t upcast<buffer_t>(memory::object_t const& memory_object)
{
    if (memory::is_buffer(memory_object)) {
        return buffer::wrap(
            memory_object.platform_handle(), memory_object.context_handle(), memory_object.handle(), is_not_owning);
    }
    throw std::invalid_argument("Attempt to upcast a memory object, which is not an image, into an image_t");
}

inline buffer_t buffer_t::create_sub_buffer(
    memory::subregion_spec_t subregion,
    memory::host_and_kernel_access_t access_spec) const
{
    return buffer::create_sub_buffer(*this, subregion, access_spec);
}

inline memory::subregion_spec_t buffer_t::subregion() const
{
    auto offset = info::get_scalar<CL_MEM_OFFSET>(handle());
    auto size = size_in_bytes();
    return { offset, size };
}


} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_BUFFER_HPP_
