
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

// TODO: Should we have a map() function?

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

namespace local {
/**
 * @brief an indication of the dynamic local memory allowance of a given kernel.
 *
 * In OpenCL, local memory can be allocated statically using in-kernel variables of
 * fixed size marked `__local`, or dynamically - via a kernel parameter, a pointer
 * marked `__local`. An argument for this parameter is not set by the user; instead,
 * an allowance, a size, of local memory to be allocated is indicated, and the
 * OpenCL driver populates the actual pointer. This structure is necessary for
 * the library automaAs we are not able to distinguish
 * a "simple" size_t or similar argument from a local memory i
 */
struct allowance_t { size_t size; };

} // namespace local

} // namespace memory

} // namespace opencl

#endif // OPENCL_WRAPPERS_MEMORY_HPP_
