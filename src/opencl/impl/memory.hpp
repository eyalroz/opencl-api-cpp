
#ifndef OPENCL_WRAPPERS_IMPL_MEMORY_HPP_
#define OPENCL_WRAPPERS_IMPL_MEMORY_HPP_

#include "../memory.hpp"
#include "miscellany.hpp"
#include "event.hpp"
#include "identify.hpp"
#include "opencl/util/mpark_variant.hpp"

namespace opencl {

namespace detail {

enum : bool {
    is_blocking = true,
    non_blocking = false,
};

} // namespace detail

namespace memory {

namespace detail {

inline context::handle_t get_context_handle(handle_t handle)
{
    return info::get_scalar<CL_MEM_CONTEXT>(handle);
}

using opencl::detail::is_blocking;
using opencl::detail::non_blocking;

template <typename RegionContainer>
dynarray<void*> get_ptrs(RegionContainer regions)
{
    // This could really just be something like:
    // return util::transform<dynarray>(regions, [](auto const& region) { return region.data(); });
    // ... except that we don't have such a utility function
    auto result = make_dynarray<void*>(regions.size());
    for (size_t i = 0; i < regions.size(); ++i) { result[i] = regions[i].data(); }
    return result;
}

template <typename RegionContainer>
dynarray<size_t> get_sizes(RegionContainer regions)
{
    // This could really just be something like:
    // return util::transform<dynarray>(regions, [](auto const& region) { return region.size(); });
    // ... except that we don't have such a utility function
    auto result = make_dynarray<size_t>(regions.size());
    for (size_t i = 0; i < regions.size(); ++i) { result[i] = regions[i].size(); }
    return result;
}

enum class kind_t {
    host,
    memory_object, // this could be a buffer, pipe or image, but with some type erasure
    buffer,
    pipe,
    shared_virtual,
    image,
    unknown
    // Note: Ignoring rectangles
};

inline char const* name_of(kind_t kind)
{
    switch (kind) {
    case kind_t::host: return "host";
    case kind_t::memory_object: return "memory object";
    case kind_t::buffer: return "buffer";
    case kind_t::pipe: return "pipe";
    case kind_t::shared_virtual: return "shared virtual region";
    case kind_t::image: return "image";
    case kind_t::unknown:
    default:
        return "(unknown)";
    }
}

template<kind_t Kind>
using lifted_kind_t = ::std::integral_constant<kind_t, Kind>;

// TODO: Add image support

// TODO: Add image support
template <typename T>
using kind_of_helper_ = std::integral_constant<kind_t,
    std::is_same<T, buffer_t>::value ? kind_t::buffer :
    std::is_same<T, pipe_t>::value ? kind_t::pipe :
    (std::is_same<T, shared_virtual::region_t>::value or
     std::is_same<T, shared_virtual::const_region_t>::value) ? kind_t::shared_virtual :
    ((opencl::detail::has_size_method<T>::value and
     opencl::detail::has_data_method<T>::value) or
     std::is_array<T>::value) ? kind_t::host :  kind_t::unknown
>;

template <typename T>
using kind_of_helper = kind_of_helper_<opencl::detail::remove_cvref_t<T>>;

template <typename T>
constexpr kind_t kind_of()
{
    return kind_of_helper<T>::value;
}

template <typename T>
constexpr kind_t kind_of(T&&) { return kind_of<T>(); }

enum offsets_into_buffer_t : size_t {
    no_offset =  0,
    no_destination_buffer_offset = no_offset,
    no_source_buffer_offset = no_offset
};

static constexpr auto no_image_offset = nullptr;
static constexpr auto no_region = nullptr;

template <typename T>
constexpr kind_t decayed_kind_of()
{
    return
        (kind_of_helper<T>::value == kind_t::buffer or
         kind_of_helper<T>::value == kind_t::pipe) ?
        kind_t::memory_object : kind_of_helper<T>::value;
    // what about images? They're also memory objects
}

} // namespace detail

struct memcpy_args_t {
    void * destination;
    void const * source;
    size_t size;
} ;

inline CL_CALLBACK void memcpy_callback(void* erased_args_owning)
{
    auto memcpy_args = static_cast<memcpy_args_t *>(erased_args_owning);
    std::memcpy(memcpy_args->destination, memcpy_args->source, memcpy_args->size);
    delete memcpy_args;
}

// TODO: Perhaps implement this for containers without value types
template <typename Container>
size_t size_in_bytes_of(Container const& container)
{
    return container.size() * sizeof(typename Container::value_type);
}

namespace detail {

template <typename SourceRegion, typename DestinationRegion>
queue::event_t enqueue_svm_copy(
    SourceRegion source,
    DestinationRegion destination,
    queue_t const& queue,
    span<event::handle_t> event_handles)
{
    if (destination.size() < source.size()) {
        throw std::invalid_argument("Attempt to copy " + shared_virtual::detail::identify_region(source)
            + "into " + shared_virtual::detail::identify_region(destination) + " - which is of smaller size");
    }
    event::handle_t resulting_event_handle;
    auto status = clEnqueueSVMMemcpy(
        queue.handle(),
        is_non_blocking,
        destination.data(),
        source.data(),
        source.size(),
        event_handles.size(),
        event_handles.data(),
        &resulting_event_handle);
    throw_if_error_lazy(status, "clEnqueueSVMMemcpy",
        "Scheduling a shared-virtual-memory copy from " + shared_virtual::detail::identify_region(source) +
        " to " + shared_virtual::detail::identify_region(destination) + " on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::shared_virtual>, // source kind
    lifted_kind_t<kind_t::shared_virtual>, // destination kind
    shared_virtual::const_region_t source,
    shared_virtual::region_t destination,
    queue_t const& queue,
    span<event::handle_t> event_handles)
{
    return enqueue_svm_copy(source, destination, queue, event_handles);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::host>, // source kind
    lifted_kind_t<kind_t::shared_virtual>, // destination kind
    const_region_t source,
    shared_virtual::region_t destination,
    queue_t const& queue,
    span<event::handle_t> event_handles)
{
    return enqueue_svm_copy(source, destination, queue, event_handles);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::shared_virtual>, // source kind
    lifted_kind_t<kind_t::host>, // destination kind
    shared_virtual::const_region_t source,
    region_t destination,
    queue_t const& queue,
    span<event::handle_t> event_handles)
{
    return enqueue_svm_copy(source, destination, queue, event_handles);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::host>,
    lifted_kind_t<kind_t::buffer>,
    const_region_t source, buffer_t const& destination, queue_t const& queue, span<event::handle_t> event_handles)
{
    if (destination.size_in_bytes() < source.size()) {
        throw std::invalid_argument("Attempt to copy a host memory region of size " + std::to_string(source.size()) +
            " to a buffer of smaller size " + std::to_string(destination.size_in_bytes()));
    }

    event::handle_t resulting_event_handle;
    auto status = clEnqueueWriteBuffer(
        queue.handle(),
        destination.handle(),
        non_blocking,
        no_destination_buffer_offset,
        source.size(),
        source.data(),
        event_handles.size(),
        event_handles.data(),
        &resulting_event_handle);
    throw_if_error_lazy(status, "clEnqueueWriteBuffer",
        "Scheduling a copy of a host memory region to an OpenCL buffer on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::buffer>,
    lifted_kind_t<kind_t::host>,
    buffer_t const& source, region_t destination, queue_t const& queue, span<event::handle_t> event_handles)
{
    if (destination.size() < source.size_in_bytes()) {
        throw std::invalid_argument("Attempt to copy a buffer of size " + std::to_string(source.size_in_bytes()) +
            " to a host memory region of smaller size " + std::to_string(destination.size()));
        // throw std::invalid_argument(opencl::detail::format("Attempt to copy a buffer of size ",
        //     source.size_in_bytes(), " to a buffer of smaller size ", destination.size()));
        // oclw_throw(std::invalid_argument, "Attempt to copy a buffer of size ",
        //     source.size_in_bytes(), " to a buffer of smaller size ", destination.size());
    }

    event::handle_t resulting_event_handle;
    auto status = clEnqueueReadBuffer(
        queue.handle(),
        source.handle(),
        non_blocking,
        no_source_buffer_offset,
        source.size_in_bytes(),
        destination.data(),
        event_handles.size(),
        event_handles.data(),
        &resulting_event_handle);
    throw_if_error_lazy(status, "clEnqueueReadBuffer",
        "Scheduling a copy from an OpenCL buffer to a host memory region on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::host>,
    lifted_kind_t<kind_t::host>,
    const_region_t source, region_t destination, queue_t const& queue, span<event::handle_t> event_handles)
{
    if (destination.size() < source.size()) {
        throw std::invalid_argument("Attempt to copy a host memory region of size " + std::to_string(source.size()) +
            " bytes to a host memory region of smaller size " + std::to_string(destination.size()) + " bytes");
    }

    auto memcpy_args_owning = new memcpy_args_t;
    memcpy_args_owning->destination = destination.data();
    memcpy_args_owning->source = source.data();
    memcpy_args_owning->size = source.size();
    event::handle_t resulting_event_handle;
    auto no_mem_objects { 0 };
    auto empty_mem_object_list { nullptr };
    auto status = clEnqueueNativeKernel(
        queue.handle(),
        memcpy_callback,
        memcpy_args_owning,
        sizeof(memcpy_args_owning),
        no_mem_objects,
        empty_mem_object_list,
        nullptr,
        event_handles.size(),
        event_handles.data(),
        &resulting_event_handle);
    throw_if_error_lazy(status, "clEnqueueNativeKernel",
        "Scheduling a copy between two host memory regions on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::buffer>,
    lifted_kind_t<kind_t::buffer>,
    buffer_t const& source, buffer_t const& destination, queue_t const& queue, span<event::handle_t> event_handles)
{
    if (destination.size_in_bytes() < source.size_in_bytes()) {
        throw std::invalid_argument("Attempt to copy a buffer of size "
            + std::to_string(source.size_in_bytes()) + " to a buffer of smaller size " + std::to_string(destination.size_in_bytes()));
    }

    event::handle_t resulting_event_handle;
    auto status = clEnqueueCopyBuffer(
        queue.handle(),
        source.handle(),
        destination.handle(),
        no_source_buffer_offset,
        no_destination_buffer_offset,
        source.size_in_bytes(),
        event_handles.size(),
        event_handles.data(),
        &resulting_event_handle);
    throw_if_error_lazy(status, "clEnqueueCopyBuffer",
        "Scheduling a copy between two OpenCL buffers on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::host>,
    lifted_kind_t<kind_t::image>,
    const_region_t source, image_t const& destination, queue_t const& queue, span<event::handle_t> event_handles)
{
    if (destination.size_in_bytes() < source.size()) {
        throw std::invalid_argument("Attempt to copy a host memory region of size " + std::to_string(source.size()) +
            " to an image of smaller size " + std::to_string(destination.size_in_bytes()));
    }

    static constexpr auto auto_row_pitch = 0;
    static constexpr auto auto_slice_pitch = 0;
    event::handle_t resulting_event_handle;
    auto status = clEnqueueWriteImage(
        queue.handle(),
        destination.handle(),
        non_blocking,
        no_image_offset,
        no_region,
        auto_row_pitch,
        auto_slice_pitch,
        source.data(),
        event_handles.size(),
        event_handles.data(),
        &resulting_event_handle);
    throw_if_error_lazy(status, "clEnqueueWriteBuffer",
        "Scheduling a copy of a host memory region to an OpenCL buffer on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::image>,
    lifted_kind_t<kind_t::host>,
    image_t const& source, region_t destination, queue_t const& queue, span<event::handle_t> event_handles)
{
    if (destination.size() < source.size_in_bytes()) {
        throw std::invalid_argument("Attempt to copy an image taking up " + std::to_string(source.size_in_bytes()) +
            " bytes to a host memory region of smaller size " + std::to_string(destination.size()));
    }

    static constexpr auto auto_row_pitch = 0;
    static constexpr auto auto_slice_pitch = 0;

    event::handle_t resulting_event_handle;
    auto status = clEnqueueReadImage(
        queue.handle(),
        source.handle(),
        non_blocking,
        no_image_offset,
        no_region,
        auto_row_pitch,
        auto_slice_pitch,
        destination.data(),
        event_handles.size(),
        event_handles.data(),
        &resulting_event_handle);
    throw_if_error_lazy(status, "clEnqueueReadImage",
        "Scheduling a copy from " + opencl::detail::identify(source) + " to a host memory region on "
        + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::image>,
    lifted_kind_t<kind_t::image>,
    image_t const& source, image_t const& destination, queue_t const& queue, span<event::handle_t> event_handles)
{
    if (destination.size_in_bytes() < source.size_in_bytes()) {
        throw std::invalid_argument("Attempt to copy " + opencl::detail::identify(source)
            + " of size " + std::to_string(source.size_in_bytes())
            + " to a buffer of smaller size " + std::to_string(destination.size_in_bytes()));
    }

    event::handle_t resulting_event_handle;
        auto status = clEnqueueCopyImage(
            queue.handle(),
            source.handle(),
            destination.handle(),
            no_image_offset,
            no_image_offset,
            no_region,
            event_handles.size(),
            event_handles.data(),
            &resulting_event_handle);
            throw_if_error_lazy(status, "clEnqueueCopyImage",
        "Scheduling a copy of " + opencl::detail::identify(source) + " to " +
             opencl::detail::identify(destination) + "on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::image>,
    lifted_kind_t<kind_t::buffer>,
    image_t const& source, buffer_t const& destination, queue_t const& queue, span<event::handle_t> event_handles)
{
    if (destination.size_in_bytes() < source.size_in_bytes()) {
        throw std::invalid_argument("Attempt to copy " + opencl::detail::identify(source)
            + " of size " + std::to_string(source.size_in_bytes())
            + " to a buffer of smaller size " + std::to_string(destination.size_in_bytes()));
    }

    event::handle_t resulting_event_handle;
    auto status = clEnqueueCopyImageToBuffer(
        queue.handle(),
        source.handle(),
        destination.handle(),
        no_image_offset,
        no_region,
        no_destination_buffer_offset,
        event_handles.size(),
        event_handles.data(),
        &resulting_event_handle);
    throw_if_error_lazy(status, "clEnqueueCopyImageToBuffer",
        "Scheduling a copy of " + opencl::detail::identify(source) + " to " +
        opencl::detail::identify(destination) + "on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}

inline queue::event_t copy(
    lifted_kind_t<kind_t::buffer>,
    lifted_kind_t<kind_t::image>,
    buffer_t const& source, image_t const& destination, queue_t const& queue, span<event::handle_t> event_handles)
{
    if (destination.size_in_bytes() < source.size_in_bytes()) {
        throw std::invalid_argument("Attempt to copy " + opencl::detail::identify(source)
            + " of size " + std::to_string(source.size_in_bytes())
            + " to a buffer of smaller size " + std::to_string(destination.size_in_bytes()));
    }

    event::handle_t resulting_event_handle;
    auto status = clEnqueueCopyBufferToImage(
        queue.handle(),
        source.handle(),
        destination.handle(),
        no_source_buffer_offset,
        no_image_offset,
        no_region,
        event_handles.size(),
        event_handles.data(),
        &resulting_event_handle);
    throw_if_error_lazy(status, "clEnqueueCopyImageToBuffer",
        "Scheduling a copy of " + opencl::detail::identify(source) + " to " +
        opencl::detail::identify(destination) + "on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}


template <typename T>
using is_buffer_builder = std::is_same<opencl::detail::remove_cvref_t<T>, builders::buffer_t>;

} // namespace detail

// Function should cover as many kinds of copying, and
// specifically OpenCL "read buffer" and "write buffer",
// which are silly names. TODO: Move this out of the memory namespace!
template <typename Source, typename Destination, typename EventContainer>
queue::event_t copy(Source&& source, Destination&& destination, queue_t const& queue, EventContainer&& events)
{
    // This function would be soooo much easier with C++17 :-(

    static_assert(not detail::is_buffer_builder<Source>::value, "Source is a buffer builder; try a buffer instead");
    static_assert(not detail::is_buffer_builder<Destination>::value, "Destination is a buffer builder; try a buffer instead");
    constexpr auto source_kind = detail::kind_of<Source>();
    constexpr auto destination_kind = detail::kind_of<Destination>();
    static_assert(source_kind != detail::kind_t::unknown, "Unsupported copy source type");
    static_assert(destination_kind != detail::kind_t::unknown, "Unsupported copy destination type");
    // Notes:
    // 1. No special behavior for pipes, so copying to/from them will probably fail
    // 2. This function does not support copying rects. It probably should...
    auto event_handles = opencl::detail::get_handles(events);
    return detail::copy(
        std::integral_constant<detail::kind_t, source_kind>{},
        std::integral_constant<detail::kind_t, destination_kind>{},
        source, destination, queue, event_handles);
}

template <typename Source, typename Destination>
queue::event_t copy(Source&& source, Destination&& destination, queue_t const& queue)
{
    return copy<Source, Destination>(
        std::forward<Source>(source),
        std::forward<Destination>(destination),
        queue,
        empty_span<event_t>());
}

template<typename B>
queue::event_t fill(B &&filled, const_region_t pattern, queue_t const &queue)
{
    event::handle_t resulting_event_handle;
    auto status = clEnqueueFillBuffer(
        queue.handle(),
        filled.handle(),
        pattern.data(),
        pattern.size(),
        detail::no_offset,
        filled.size(),
        &resulting_event_handle);
    throw_if_error_lazy(status, "clEnqueueFillBuffer", "Filling a buffer with a pattern");
    return event::detail::wrap_by_queue(resulting_event_handle, queue, is_owning);
}

template<typename B, typename T>
queue::event_t set(B &&filled, T value, queue_t const &queue)
{
    const_region_t pattern { &value, sizeof(T) };
    return fill(std::forward<B>(filled), pattern, queue);
}

template<typename B, typename T>
queue::event_t zero(B &&zeroed, queue_t const &queue)
{
    return set(std::move(zeroed), static_cast<T>(0), queue);
}

inline queue::event_t migrate(
    memory::object_t const& object,
    queue_t const& queue,
    memory::destination_t destination,
    bool maintain_contents)
{
    return queue.enqueue_migrate(object, destination, maintain_contents);
}

#if CL_VERSION_1_2
namespace migration {
namespace detail {

inline flags_t make_flags(destination_t destination, bool maintain_contents)
{
    cl_mem_migration_flags result = 0;
    if (destination == destination_t::host) { result |= CL_MIGRATE_MEM_OBJECT_HOST; }
    if (not maintain_contents) { result |= CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED; }
    return result;
}

} // namespace detail
} // namespace migration

#endif // CL_VERSION_1_2

} // namespace memory

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_MEMORY_HPP_
