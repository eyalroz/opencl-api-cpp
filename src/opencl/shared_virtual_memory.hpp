/**
 * @file
 *
 * @brief Declarations of functions and definitions of types related to OpenCL's
 * "Shared Virtual Memory" mechanism - areas in memory which can be accessed
 * directly, using the same address, from both host-side code and OpenCL kernel/
 * device-side code.
 *
 * @note Shared Virtual Memory is only supported with OpenCL 2.0 or later; for
 * older versions, this file is effectively empty.
 */
#ifndef OPENCL_WRAPPERS_SHARED_VIRTUAL_MEMORY_HPP_
#define OPENCL_WRAPPERS_SHARED_VIRTUAL_MEMORY_HPP_

#ifdef CL_VERSION_2_0

#include "types.hpp"

namespace opencl {

namespace memory {

namespace shared_virtual {

namespace detail {

void free(void* ptr);
event::handle_t free(queue::handle_t queue_handle, void *ptr);

} // namespace detail

region_t allocate(
    context_t const& context,
    size_t size,
    access_spec_t access_spec,
    alignment_t alignment = default_alignment);

// Note: We are not defining a RAII/CADRe class for shared memory regions, because while
// their allocation is immediate, and requires merely a context - their lifetime lasts
// either until the context is released/destroyed, or until a de-allocation on a queue.
// If we were to pass a queue in a closure to be stored with the region object, we
// would also have to retain that queue, i.e. increase its refcount. That _is_ possible,
// but it's not clear that the exercise is worthwhile.

region_t allocate(context_t const& context, size_t size);
void free(region_t region);
queue::event_t free(queue_t const& queue, span<region_t const> svm_regions);
queue::event_t free(queue_t const& queue, region_t region);


/**
 * A memory region allocated as shared virtual. This thin wrapper class
 * behaves just like `region_t` - except that other code can notice, if
 * it so wishes, its nature as shared virtual memory.
 */
class region_t : public memory::region_t {
    using parent_type = memory::region_t;
    using parent_type::parent_type;
};

class const_region_t : public memory::const_region_t {
    using parent_type = memory::const_region_t;
    using parent_type::parent_type;
};

} // namespace shared_virtual

} // namespace memory

namespace svm = memory::shared_virtual;

} // namespace opencl

#endif // CL_VERSION_2_0

#endif // OPENCL_WRAPPERS_SHARED_VIRTUAL_MEMORY_HPP_
