
#ifndef OPENCL_WRAPPERS_IMPL_BUILDERS_BUFFER_HPP_
#define OPENCL_WRAPPERS_IMPL_BUILDERS_BUFFER_HPP_

#include "../../builders/buffer.hpp"
#include "../../context.hpp"
#include "../../error.hpp"

namespace opencl {

namespace builders {

inline opencl::buffer_t buffer_t::create()
{
    // Should this really be here?
    if (not platform_handle_) {
        platform_handle_ = context::detail::get_platform_handle(context_handle_);
    }

    constexpr auto ensure_sufficiency { true };
    validate(ensure_sufficiency);

    auto ptr =
        has_flags(CL_MEM_COPY_HOST_PTR) ?
            const_cast<void*>(copy_host_ptr_) :
            has_flags(CL_MEM_USE_HOST_PTR) ? host_ptr_ : nullptr;
    // TODO: Add support for properties
    status_t status;
    auto new_buffer_handle = clCreateBuffer(
        context_handle_,
        flags_,
        size_,
        ptr,
        &status);
    throw_if_error_lazy(status, "clCreateBuffer", "Creating an OpenCL buffer with flags "
        + opencl::detail::as_hex(static_cast<unsigned>(flags_)));
    // TODO: Print the string representation of these flags
    return buffer::wrap(platform_handle_, context_handle_, new_buffer_handle, is_owning);
}

inline opencl::buffer_t buffer_t::build() { return create(); }

inline buffer_t& buffer_t::size(size_t size)
{
    size_ = size;
    return *this;
}

template <typename ContiguousContainer>
buffer_t& buffer_t::size_by(ContiguousContainer&& container)
{
    static_assert(opencl::detail::is_kinda_like_contiguous_container<ContiguousContainer>::value,
        "Can't copy from a non-contiguous container");
    size_ = container.size() * sizeof(*container.data());
    return *this;
}

inline void buffer_t::validate(bool) const
{
    // TODO: WRITEME
}

} // namespace builders

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_BUILDERS_BUFFER_HPP_
