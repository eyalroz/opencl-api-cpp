
#ifndef OPENCL_WRAPPERS_IMPL_BUILDERS_MEMORY_OBJECT_HPP_
#define OPENCL_WRAPPERS_IMPL_BUILDERS_MEMORY_OBJECT_HPP_

#include "../../builders/memory_object.hpp"
#include "../../context.hpp"

namespace opencl {

namespace builders {

namespace detail {
#ifdef CL_VERSION_1_2

template <typename Builder>
Builder& memory_object_t<Builder>::context(context_t const& context)
{
    context_handle_ = context.handle();
    platform_handle_ = context.platform_handle();
    return static_cast<Builder&>(*this);
}

template <typename Builder>
Builder& memory_object_t<Builder>::disable_host_access(access_kind_t access_kind)
{
    switch(access_kind) {
    case access_kind_t::readwrite:
        lower_flags(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE);
        raise_flags(CL_MEM_HOST_NO_ACCESS);
        break;
    case access_kind_t::read:
         if (flags_ & CL_MEM_HOST_READ_ONLY) {
           lower_flags(CL_MEM_HOST_READ_ONLY);
           raise_flags(CL_MEM_HOST_NO_ACCESS);
         }
         else if (flags_ & CL_MEM_READ_WRITE) {
             lower_flags(CL_MEM_READ_WRITE);
             raise_flags(CL_MEM_WRITE_ONLY);
         }
         break;
    case access_kind_t::write:
        if (flags_ & CL_MEM_HOST_READ_ONLY) {
            lower_flags(CL_MEM_HOST_READ_ONLY);
        }
        else if (flags_ & CL_MEM_READ_WRITE) {
            lower_flags(CL_MEM_READ_WRITE);
            raise_flags(CL_MEM_WRITE_ONLY);
        }
        break;
    case access_kind_t::none: break;
    }
    return static_cast<Builder&>(*this);
}

template <typename Builder>
Builder& memory_object_t<Builder>::enable_host_access(access_kind_t access_kind)
{
    switch(access_kind) {
    case access_kind_t::none:
        break;
    case access_kind_t::readwrite:
        lower_flags(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS);
        break;
    case access_kind_t::read:
        lower_flags(CL_MEM_HOST_NO_ACCESS);
        lower_flags(CL_MEM_HOST_WRITE_ONLY);
        break;
    case access_kind_t::write:
        lower_flags(CL_MEM_HOST_NO_ACCESS);
        lower_flags(CL_MEM_HOST_READ_ONLY);
        break;
    }
    return static_cast<Builder&>(*this);
}

template <typename Builder>
Builder& memory_object_t<Builder>::host_access(access_kind_t access_kind)
{
    raise_flags(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS);
    enable_host_access(access_kind);
    return static_cast<Builder&>(*this);
}
#endif // CL_VERSION_1_2

template <typename Builder>
Builder& memory_object_t<Builder>::no_host_access()
{
    return disable_host_access(access_kind_t::read_and_write);
}

template <typename Builder>
Builder& memory_object_t<Builder>::disable_device_access(access_kind_t access_kind)
{
    // Remember the must always be _some_ device access
    switch(access_kind) {
    case access_kind_t::none:
        break;
    case access_kind_t::read:
            if (flags_ & CL_MEM_READ_ONLY) {
                throw std::invalid_argument("Cannot completely disable device access to an OpenCL buffer");
            }
            lower_flags(CL_MEM_READ_WRITE);
            raise_flags(CL_MEM_WRITE_ONLY);
        break;
    case access_kind_t::write:
            if (flags_ & CL_MEM_WRITE_ONLY) {
                throw std::invalid_argument("Cannot completely disable device access to an OpenCL buffer");
            }
            lower_flags(CL_MEM_READ_WRITE);
            raise_flags(CL_MEM_READ_ONLY);
        break;
    case access_kind_t::readwrite:
            throw std::invalid_argument("Cannot completely disable device access to an OpenCL buffer");
    }
    return static_cast<Builder&>(*this);
}

template <typename Builder>
Builder& memory_object_t<Builder>::enable_device_access(access_kind_t access_kind)
{
    // Remember the must always be _some_ device access
    switch(access_kind) {
    case access_kind_t::none:
        break;
    case access_kind_t::read:
        if (flags_ & CL_MEM_WRITE_ONLY) {
            lower_flags(CL_MEM_WRITE_ONLY);
            raise_flags(CL_MEM_READ_WRITE);
        }
        // otherwise it's read-only or read-write, nothing to do
        break;
    case access_kind_t::write:
        if (flags_ & CL_MEM_READ_ONLY) {
            lower_flags(CL_MEM_READ_ONLY);
            raise_flags(CL_MEM_READ_WRITE);
        }
        // otherwise it's write-only or read-write, nothing to do
        break;
    case access_kind_t::readwrite:
        lower_flags(CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY);
        raise_flags(CL_MEM_READ_WRITE);
    }
    return static_cast<Builder&>(*this);
}

template <typename Builder>
Builder& memory_object_t<Builder>::device_access(access_kind_t access_kind)
{
    if (access_kind == access_kind_t::none) {
        throw std::invalid_argument("Cannot completely disable device access to an OpenCL buffer");
    }
    lower_flags(CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY);
    return enable_device_access(access_kind);
}

template <typename Builder>
Builder& memory_object_t<Builder>::no_device_access()
{
    return device_access(access_kind_t::none);
}

template <typename Builder>
Builder& memory_object_t<Builder>::enable_access(side_t side, access_kind_t access_kind)
{
    return (side == side_t::host) ?
        enable_host_access(access_kind) :
        enable_device_access(access_kind);
}

template <typename Builder>
Builder& memory_object_t<Builder>::disable_access(side_t side, access_kind_t access_kind)
{
    return (side == side_t::host) ?
        disable_host_access(access_kind) :
        disable_device_access(access_kind);
}

template <typename Builder>
Builder& memory_object_t<Builder>::access(side_t side, access_kind_t access_kind)
{
    return (side == side_t::host) ?
        host_access(access_kind) :
        device_access(access_kind);
}

template <typename Builder>
Builder& memory_object_t<Builder>::require_allocation(size_t size)
{
    size_ = size;
    raise_flags(CL_MEM_ALLOC_HOST_PTR);
    // Let's be courteous in case the user has previously had this builder use a pointer
    lower_flags(CL_MEM_USE_HOST_PTR);
    host_ptr_ = nullptr;
    return static_cast<Builder&>(*this);
}

template <typename Builder>
Builder& memory_object_t<Builder>::host_side_storage(memory::region_t region)
{
    host_ptr_ = region.data(); size_ = region.size();
    if (size_ == 0) {
        throw std::runtime_error("An OpenCL buffer must have a non-zero size");
    }
    copy_host_ptr_ = nullptr;
    lower_flags(CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR);
    raise_flags(CL_MEM_USE_HOST_PTR);
    return static_cast<Builder&>(*this);
}

template <typename Builder>
Builder& memory_object_t<Builder>::host_region_to_copy(void const * ptr, size_t size)
{
    if (ptr == nullptr) {
        throw std::runtime_error("Null pointer provided for a host memory region to copy into a new OpenCL buffer");
    }
    if (size == 0) {
        throw std::runtime_error("Attempt to create an OpenCL buffer from a size-0 memory buffer");
    }
    // Note: It _is_ legitimate to have CL_MEM_aLLOC_HOST_PTR and specify a region to copy from
    size_ = size;
    copy_host_ptr_ = ptr;
    raise_flags(CL_MEM_COPY_HOST_PTR);
    return static_cast<Builder&>(*this);
}

template <typename Builder>
Builder& memory_object_t<Builder>::host_region_to_copy(memory::const_region_t region)
{
    return host_region_to_copy(region.data(), region.size());
}

template <typename Builder>
Builder& memory_object_t<Builder>::host_region(memory::region_t region)
{
    return host_region(region.data(), region.size());
}

template <typename Builder>
template <typename Container>
Builder& memory_object_t<Builder>::host_region_to_copy(Container const & container)
{
    using value_type = typename Container::value_type;
    static_assert(opencl::detail::is_kinda_like_contiguous_container<Container>::value,
        "Can't copy from a non-contiguous container");
    static_assert(std::is_trivially_copyable<value_type>::value,
        "Container values must be trivially copyable");
    return host_region_to_copy(container.data(), container.size() * sizeof(value_type));
}

template <typename Builder>
template <typename Container>
Builder& memory_object_t<Builder>::host_region(Container const & container)
{
    using value_type = typename Container::value_type;
    static_assert(opencl::detail::is_kinda_like_contiguous_container<Container>::value,
        "Can only use a contiguous container for a host region");
    static_assert(std::is_standard_layout<value_type>::value and std::is_trivial<value_type>::value,
        "Containers serving as host regions must have extra-nice types... standard-layout and trivial");
    return host_region(container.data(), container.size() * sizeof(value_type));
}

template <typename Builder>
Builder& memory_object_t<Builder>::host_region_start(void * ptr, size_t size)
{
    size_ = size;
    host_ptr_ = ptr;
    raise_flags(CL_MEM_COPY_HOST_PTR);
    return *this;
}


template <typename Builder>
Builder& memory_object_t<Builder>::no_accessible_host_storage()
{
    lower_flags(CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR);
    host_ptr_ = nullptr;
    return static_cast<Builder&>(*this);
}

template <typename Builder>
void memory_object_t<Builder>::validate(bool ensure_sufficiency) const noexcept(false)
{
    if (size_ == 0) {
        throw std::runtime_error("An OpenCL buffer must have a non-zero size");
    }
    if (has_flags(CL_MEM_USE_HOST_PTR) and has_flags(CL_MEM_ALLOC_HOST_PTR)) {
        throw std::runtime_error(
            "Cannot both use an existing memory region for the buffer "
            "and allocate a new region for this purpose");
    }
    if (has_flags(CL_MEM_USE_HOST_PTR) and not host_ptr_) {
        throw std::runtime_error(
            "Request to use existing host memory for the buffer, "
            "but the pointer to this memory is null");
    }
    if (not has_flags(CL_MEM_USE_HOST_PTR) and host_ptr_) {
        throw std::runtime_error("Gratuitous host memory pointer specified");
    }
    if (has_flags(CL_MEM_COPY_HOST_PTR) and not copy_host_ptr_) {
        throw std::runtime_error(
            "Request to copy the contents of an existing host memory region into the buffer, "
            "but the pointer to copy source is null");
    }
    if (not has_flags(CL_MEM_COPY_HOST_PTR) and copy_host_ptr_) {
        throw std::runtime_error("Gratuitous copy source pointer specified");
    }
    // Should we check CL_MEM_USE_HOST_PTR against host access flags?

    if (ensure_sufficiency) {
        if (not context_handle_) {
            throw std::runtime_error("Attempt to create an OpenCL buffer without specifying a context");
        }
    }
}

template<typename Builder>
memory::host_and_kernel_access_t memory_object_t<Builder>::get_access_spec() const
{
    return memory::detail::from_flags(flags_);
}

} // namespace detail

} // namespace builders

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_BUILDERS_MEMORY_OBJECT_HPP_
