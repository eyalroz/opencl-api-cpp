
#ifndef OPENCL_WRAPPERS_IMPL_SHARED_VIRTUAL_MEMORY_HPP_
#define OPENCL_WRAPPERS_IMPL_SHARED_VIRTUAL_MEMORY_HPP_

#include "../shared_virtual_memory.hpp"
#include "../context.hpp"
#include "../event.hpp"
#include "memory_object.hpp"

namespace opencl {
namespace memory {

namespace shared_virtual {

namespace detail {

inline void free(context::handle_t context_handle, void* ptr)
{
    clSVMFree(context_handle, ptr);
    // Can this really not fail? Ok...
}

inline event::handle_t free(queue::handle_t queue_handle, span<void*> svm_region_ptrs)
{
    auto no_callback = nullptr;
    auto no_user_data = nullptr;
    span<event::handle_t> no_events {};
    event::handle_t new_event_handle;
    auto status = clEnqueueSVMFree(
        queue_handle,
        svm_region_ptrs.size(),
        svm_region_ptrs.data(),
        no_callback,
        no_user_data,
        no_events.size(),
        no_events.data(),
        &new_event_handle);
    throw_if_error_lazy(status, "clEnqueueSVMFree",
        (svm_region_ptrs.size() == 1) ?
        "Freeing a shared shared virtual memory region at " + opencl::detail::ptr_as_hex(svm_region_ptrs[0]) :
        "Freeing " + std::to_string(svm_region_ptrs.size()) + " shared virtual memory regions");
    return new_event_handle;
}

inline event::handle_t free(queue::handle_t queue_handle, void *ptr)
{
    span<void *> single_ptr_span { &ptr, 1 };
    return free(queue_handle, single_ptr_span);
}

inline flags_t make_flags(access_spec_t access_spec)
{
    flags_t flags {};
    if (access_spec.atomics)    { flags |= CL_MEM_SVM_ATOMICS; }
    if (access_spec.fine_grain) { flags |= CL_MEM_SVM_FINE_GRAIN_BUFFER; }
    flags |= memory::detail::make_kernel_access_flags(access_spec.access_kind);
    return flags;
}

} // namespace detail

inline region_t allocate(
    context_t const& context,
    size_t size,
    access_spec_t access_spec,
    alignment_t alignment)
{
    auto flags = detail::make_flags(access_spec);
    auto ptr = clSVMAlloc(context.handle(), flags, size, alignment);
    if (not ptr) {
        throw std::runtime_error("Unknown error while invoking clSVMAlloc");
    }
    return { ptr, size };
}

inline region_t allocate(context_t const& context, size_t size)
{
    return allocate(context, size, read_and_write(), default_alignment);
}

inline void free(region_t region) { detail::free(region.data()); }

inline queue::event_t free(queue_t const& queue, span<region_t> svm_regions)
{
    auto svm_pointers = memory::detail::get_ptrs(svm_regions);
    auto event_handle = detail::free(queue.handle(), svm_pointers);
    return queue::event::detail::wrap_by_queue(event_handle, queue, is_owning);
}

inline queue::event_t free(queue_t const& queue, region_t region)
{
    span<region_t> svm_regions { &region, 1 };
    return free(queue, svm_regions);
}

} // namespace shared_virtual

} // namespace memory

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_SHARED_VIRTUAL_MEMORY_HPP_
