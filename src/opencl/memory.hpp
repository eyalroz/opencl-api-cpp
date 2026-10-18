/**
 * @file
 *
 * @brief various functions, and a few type definitions, in the @ref opencl::memory
 * namespace - relating to memory objects and operations on memory.
 */
#ifndef OPENCL_WRAPPERS_MEMORY_HPP_
#define OPENCL_WRAPPERS_MEMORY_HPP_

#include "types.hpp"

namespace opencl {

namespace memory {

namespace detail {

context::handle_t get_context_handle(handle_t);

/// Takes a container of elements which correspond to a region of memory, or
/// some contiguous storage, and have a `data()` member - and returns a
/// (dynamic) array of those data() pointers.
template <typename RegionContainer>
dynarray<void*> get_ptrs(RegionContainer regions);
template <typename RegionContainer>
dynarray<size_t> get_sizes(RegionContainer regions);
}

struct relative_region_spec_t {
    offset_t offset; // Note: this must be non-negative
    size_t size;
};

// OpenCL buffers do not allow for direct access via pointers;
// hence, if we want to refer to region within a buffer - some
// number of bytes at some offset
struct buffer_region_t {
    buffer_t const& buffer;
    relative_region_spec_t region_spec;
};

// Function should cover as many kinds of copying, and
// specifically OpenCL "read buffer" and "write buffer",
// which are silly names
template <typename Source, typename Destination, typename EventContainer>
queue::event_t copy(Source&& source, Destination&& destination, queue_t const& queue, EventContainer&&);

template <typename Source, typename Destination>
queue::event_t copy(Source&& source, Destination&& destination, queue_t const& queue);

template <typename B>
queue::event_t fill(B&& filled, const_region_t pattern, queue_t const& queue);

// TODO: A variant of set() which takes a single unsigned numeric type as the pattern

template <typename B, typename T>
queue::event_t set(B&& filled, T value, queue_t const& queue);

template<typename B, typename T>
queue::event_t zero(B &&zeroed, queue_t const &queue);

/**
 * Arrange it so that changes to a buffer or image are reflected in a region of host memory, or
 * vice-versa, that changes to a region of host memory apply to the OpenCL buffer as well.
 *
 * @note this function is similar to the @ref mmap for files: Changes to some region of
 * memory translate into changes to a file (or vice-versa).
 *
 * @param queue The command queue on which to place the mapping setup operation, as it
 *    is not executed immediately
 * @param spec specification of what kind of mapping to perform: what part of the buffer or
 *    image, and whether to map for reading buffer contents via the memory region, or for
 *    having changes to the region apply to the buffer as well (mapping for writing). It is
 *    also possible to specify a request to unmap a previously-mapped buffer.
 * @return a pair: the event of the intended completion of the mapping, and the mapped region
 *    (which will be usable after the event has occurred).
 *
 * @note the mapping can be undone using the @ref unmap function.
 */
///@{
/// @param buffer the buffer to map to host memory region
std::pair<queue::event_t, region_t>
map(buffer_t const& buffer, memory::mapping::spec_t const& spec, queue_t const& queue);

/// @param image the image to map to host memory region
std::pair<queue::event_t, region_t>
map(image_t const& image, image::mapping::spec_t const& spec, queue_t const& queue);
///@}

/**
* Break a previously-set-up mapping of a host memory region to a buffer (or a subregion
* thereof).
*
* @note this operation undoes the effect of {@ref map}.
*/
///@{
queue::event_t unmap(memory::object_t const& memory_object, void * mapped_region, queue_t const& queue);
queue::event_t unmap(memory::object_t const& buffer, region_t mapped_region, queue_t const& queue);
///@}

queue::event_t migrate(memory::object_t const& object, queue_t const& queue);

namespace detail {

inline cl_map_flags make_map_flags(mapping::kind_t kind)
{
    using mapping::kind_t;
    switch (kind) {
    case kind_t::read: return CL_MAP_READ;
    case kind_t::write: return CL_MAP_WRITE;
    case kind_t::read_and_write: return CL_MAP_READ | CL_MAP_WRITE;
    case kind_t::invalidate_and_write: return CL_MAP_WRITE_INVALIDATE_REGION;
    default: throw std::invalid_argument("Unknown mapping kind - should not be possible to get here");
    }
}

} // namespace detail

namespace migration {
namespace detail {

flags_t make_flags(destination_t destination, bool maintain_contents);

} // namespace detail
} // namespace migration

} // namespace memory

} // namespace opencl

#endif // OPENCL_WRAPPERS_MEMORY_HPP_
