
#ifndef OPENCL_WRAPPERS_IMPL_QUEUE_HPP_
#define OPENCL_WRAPPERS_IMPL_QUEUE_HPP_

#include "shared_virtual_memory.hpp"
#include "launch.hpp"
#include "memory_object.hpp"

#include "../queue.hpp"
#include "../context.hpp"
#include "../context_device.hpp"
#include "../image.hpp"

#ifndef NDEBUG
#include <algorithm>
#endif

namespace opencl {

inline char const* info::traits_t<info::queue>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case CL_QUEUE_CONTEXT:                            return "context";
    case CL_QUEUE_DEVICE:                             return "device";
    case CL_QUEUE_REFERENCE_COUNT:                    return "reference count";
    case CL_QUEUE_PROPERTIES:                         return "properties bit-field";
#ifdef CL_VERSION_2_0
    case CL_QUEUE_SIZE:                               return "size";
#endif
#ifdef CL_VERSION_2_1
    case CL_QUEUE_DEVICE_DEFAULT:                     return "device-default queue";
#endif
#ifdef CL_VERSION_3_0
    case CL_QUEUE_PROPERTIES_ARRAY:                   return "properties array";
#endif
    default: return nullptr;
    }
}

namespace detail {

OCLW_DEFINE_HANDLE_TRAITS("command queue", queue::handle_t, clReleaseCommandQueue, clRetainCommandQueue);

} // namespace detail

namespace queue {

namespace detail {

// TODO: Support queue properties
inline handle_t create(
    context::handle_t context_handle,
    device::handle_t  device_handle)
{
    status_t status;
#ifdef CL_VERSION_2_0
    static constexpr auto api_function_name = "clCreateCommandQueueWithProperties";
    handle_t result = clCreateCommandQueueWithProperties
#else
    static constexpr const api_function_name = "clCreateCommandQueue";
    handle_t result = clCreateCommandQueue
#endif
        (context_handle, device_handle, nullptr, &status);
    throw_if_error_lazy(status, api_function_name, "Failed creating a command queue for "
        + opencl::detail::identify(device_handle) + " in "
        + opencl::detail::identify(context_handle));
    return result;
}

inline device::handle_t get_device_handle(handle_t handle)
{
    return info::get_scalar<CL_QUEUE_DEVICE>(handle);
}

inline context::handle_t get_context_handle(handle_t handle)
{
    return info::get_scalar<CL_QUEUE_CONTEXT>(handle);
}

inline queue_t create(context_t const& context, device::handle_t device_handle)
{
    auto handle = create(context.handle(), device_handle);
    return wrap(context.platform_handle(), context.handle(), device_handle, handle, is_owning);
}

} // namespace detail

inline queue_t wrap(
    platform::handle_t platform_handle,
    context::handle_t context_handle,
    device::handle_t    device_handle,
    handle_t            handle,
    bool                owning) noexcept
{
    return queue_t { platform_handle, context_handle, device_handle, handle, owning };
}

// TODO: What about properties? :-(
inline queue_t create(context::device_t const& device)
{
    auto handle = detail::create(device.context_handle(), device.handle());
    return wrap(device.platform_handle(), device.context_handle(), device.handle(), handle, is_owning);
}

inline queue_t create(context_t const& context, device_t const& device)
{
    return detail::create(context, device.handle());
}

} // namespace queue

inline void queue_t::flush() const
{
    auto status = clFlush(handle_);
    throw_if_error_lazy(status, "clFlush", "Flushing " + detail::identify(*this));
}

inline void queue_t::finish() const
{
    auto status = clFinish(handle_);
    throw_if_error_lazy(status, "clFinish", "Completing all commands remaining on " + detail::identify(*this));
}

template <typename Source, typename Destination>
queue::event_t queue_t::enqueue_copy(Source&& source, Destination&& destination) const
{
    return memory::copy(std::forward<Source>(source), std::forward<Destination>(destination), *this);
}

template <typename... Ts>
queue::event_t queue_t::enqueue_kernel_launch(
    kernel_t const& kernel,
    kernel::launch_configuration_t launch_config,
    Ts&&... additional_arguments) const
{
    return enqueue_launch(kernel, *this, launch_config, std::forward<Ts>(additional_arguments)...);
}

template <
    typename NativeKernelFunction,
    typename BufferParameterPositionContainer>
queue::event_t enqueue_native_kernel_launch(
    queue_t const& queue,
    NativeKernelFunction kernel_function,
    span<void*> buffers,
    BufferParameterPositionContainer&& buffer_param_positions,
    span<void*> pointers_to_arguments)
{
    static_assert(std::is_function<NativeKernelFunction>::value, "The native kernel must be of a function type");
    using type_erased_native_function = void (CL_CALLBACK *)(void *);
    static_assert(std::is_constructible<type_erased_native_function, NativeKernelFunction>::value,
        "Cannot convert the native kernel function passed to the raw native kernel function type");
    if (buffers.size() != buffer_param_positions.size()) {
        throw std::invalid_argument("Mismatched number of buffers and number of buffer positions in arguments");
    }
    auto pos_to_ptr_into_pointers_to_arguments =
        [&](size_t pos)  { return static_cast<void *>(&pointers_to_arguments[pos]); };
    auto pointers_to_buffer_locations_in_ptrs_to_arguments =
        to_dynarray(buffer_param_positions, pos_to_ptr_into_pointers_to_arguments);

    span<event::handle_t> no_events {};
    queue::event::handle_t result_event_handle;
    auto status = clEnqueueNativeKernel(
        queue.handle(),
        static_cast<type_erased_native_function>(kernel_function), //void
        pointers_to_arguments.data(), pointers_to_arguments.size() * sizeof(void*),
        buffers.size(), buffers.data(),
        pointers_to_buffer_locations_in_ptrs_to_arguments.data(),
        no_events.size(), no_events.data(),
        &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueNativeKernel",
        "Failed enqueueing a native kernel on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(result_event_handle, queue, is_owning);
}

template <
    typename NativeKernelFunction,
    typename BufferParameterPositionContainer>
queue::event_t queue_t::enqueue_native_kernel_launch(
    NativeKernelFunction kernel_function,
    span<void*> buffers,
    BufferParameterPositionContainer&& buffer_param_positions,
    span<void*> pointers_to_arguments) const
{
    return opencl::enqueue_native_kernel_launch(*this,
        kernel_function, buffers,
        std::forward<BufferParameterPositionContainer>(buffer_param_positions),
        pointers_to_arguments);
}

namespace detail {

template <typename Invokable>
static void CL_CALLBACK queue_launched_invoker(void* type_erased_invokable) {
    auto invokable = static_cast<Invokable*>(type_erased_invokable);
    (*invokable)();
}

} // namespace detail

template <typename Invokable>
queue::event_t queue_t::enqueue_pure_host_invokable(Invokable& invokable) const
{
    auto type_erased_invoker = reinterpret_cast<queue::callback_t>(detail::queue_launched_invoker<Invokable>);
    event::handle_t result_event_handle;
    span<event::handle_t> no_events {};
    span<memory::handle_t> no_mem_objects {};
    auto no_mem_object_locations = nullptr;
    span<void*> args = { &invokable, 1 };
    auto status = clEnqueueNativeKernel(
        handle_,
        type_erased_invoker, args.data(), args.size(),
        no_mem_objects.size(), no_mem_objects.data(), no_mem_object_locations,
        no_events.size(), no_events.data(),
        &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueNativeKernel", "Enqueuing a purely native, non-OpenCL-related function");
    return event::detail::wrap_by_queue(result_event_handle, *this, is_owning);
}

inline std::pair<queue::event_t, memory::region_t>
map(buffer_t const& buffer, memory::mapping::spec_t const& spec, queue_t const& queue, bool blocking)
{
    event::handle_t result_event_handle;
    status_t status;
    span<event::handle_t> no_events {};
    auto subregion = spec.subregion.value_or(memory::subregion_spec_t{0, buffer.size_in_bytes()});
    auto mapping_ptr = clEnqueueMapBuffer(
        queue.handle(),
        buffer.handle(),
        blocking,
        memory::detail::make_map_flags(spec.kind),
        subregion.origin, subregion.size,
        no_events.size(), no_events.data(),
        &result_event_handle, &status);
    throw_if_error_lazy(status, "clEnqueueMapBuffer", "Mapping " + opencl::detail::identify(buffer)
            + "to host memory on " + opencl::detail::identify(queue));
    return {
        event::detail::wrap_by_queue(result_event_handle, queue, is_owning),
        memory::region_t { mapping_ptr, spec.subregion->size }
    };
}

//
// template <
//     typename NativeKernelFunction,
//     typename BufferContainer,
//     typename BufferParameterPositionContainer,
//     typename ArgumentsContainer>
// queue::event_t queue_t::enqueue_native_kernel_launch(
//     NativeKernelFunction kernel_function,
//     BufferContainer&& buffers,
//     BufferParameterPositionContainer&& buffer_param_positions,
//     ArgumentsContainer&& arguments);

inline std::pair<queue::event_t, memory::region_t> queue_t::enqueue_map(
    buffer_t const& buffer, memory::mapping::spec_t const& spec) const
{
    return map(buffer, spec, *this, is_non_blocking);
}

inline queue::event_t unmap(memory::object_t const& memory_object, void* mapping, queue_t const& queue)
{
    event::handle_t result_event_handle;
    span<event::handle_t> no_events {};
    auto status = clEnqueueUnmapMemObject(
        queue.handle(),  memory_object.handle(), mapping,
        no_events.size(), no_events.data(), &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueUnmapMemObject", "Unmapping " + detail::identify(memory_object));
    return event::detail::wrap_by_queue(result_event_handle, queue, is_owning);
}

inline queue::event_t queue_t::enqueue_unmap(memory::object_t const& memory_object, void* mapping) const
{
    return unmap(memory_object, mapping, *this);
}

inline std::pair<queue::event_t, memory::region_t>
map(image_t const& image, image::mapping::spec_t const& spec, queue_t const& queue, bool blocking)
{
    event::handle_t event_handle;
    status_t status;
    span<event::handle_t> no_events {};
    auto mapping_ptr = clEnqueueMapImage(queue.handle(), image.handle(), blocking,
        memory::detail::make_map_flags(spec.kind),
        spec.box ? nullptr : spec.box->origin.data(),
        spec.box ? nullptr : spec.box->extents.data(),
        (image.dimensionality() > 1 and spec.pitches) ? const_cast<size_t*>(&((*spec.pitches)[0])) : nullptr,
        (image.dimensionality() > 1 and spec.pitches) ? const_cast<size_t*>(&((*spec.pitches)[1])) : nullptr,
        no_events.size(),
        no_events.data(),
        &event_handle,
        &status) CL_API_SUFFIX__VERSION_1_0;
    throw_if_error_lazy(status, "clEnqueueMapImage", "");
    auto mapping = memory::region_t { mapping_ptr, image.size_in_bytes() };
    return std::make_pair(event::detail::wrap_by_queue(event_handle, queue, is_owning), mapping );
}

inline std::pair<queue::event_t, memory::region_t>
queue_t::enqueue_map( image_t const& image, image::mapping::spec_t const& spec) const
{
    return map(image, spec, *this, is_non_blocking);
}

template <typename EventContainer>
queue::event_t enqueue_marker(queue_t const& queue, EventContainer&& events)
{
#ifndef NDEBUG
    if (events.empty()) {
        throw std::invalid_argument("Attempt to enqueue a marker for the conclusion of no events");
    }
#endif
    auto event_handles = detail::get_handles(events);
    event::handle_t result_event_handle;
    auto status = clEnqueueMarkerWithWaitList(
        queue.handle(), event_handles.size(), event_handles.data(), &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueMarkerWithWaitList",
        "Enqueueing a marker for the conclusion of " + std::to_string(event_handles.size())
        + "events on " + detail::identify(queue));
    return event::detail::wrap_by_queue(result_event_handle, queue, is_owning);
}

template <typename EventContainer>
queue::event_t queue_t::enqueue_marker(EventContainer&& events) const
{
    return opencl::enqueue_marker(*this, events);
}

// Note: A barrier is a marker which also enforces execution order: No command
// after the barrier can begin to execute until the commands before the barrier on
// the queue, and the events in the list, have concluded.
template <typename EventContainer = span<event_t>>
queue::event_t enqueue_barrier(queue_t const& queue, EventContainer&& events)
{
    auto event_handles = detail::get_handles(events);
    event::handle_t result_event_handle;
    auto status = clEnqueueBarrierWithWaitList(
        queue.handle(), event_handles.size(), event_handles.data(), &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueBarrierWithWaitList",
        "Enqueueing a barrier operation (" +
        (event_handles.empty() ? "without waiting for " : " and waiting for " + std::to_string(event_handles.size()))
        + "events on " + detail::identify(queue));
    return event::detail::wrap_by_queue(result_event_handle, queue, is_owning);
}

inline queue::event_t queue_t::enqueue_barrier() const
{
    return opencl::enqueue_barrier(*this);
}

#ifdef cl_khr_external_memory

namespace queue {

namespace detail {

template <typename Container>
event_t enqueue_acquire_external_memory(queue_t const& queue, Container const& memory_object_handles)
{
    event::handle_t result_event_handle;
    span<event::handle_t> no_events {};
    auto status = clEnqueueAcquireExternalMemObjectsKHR(
        queue.handle(),
        memory_object_handles.size(),
        memory_object_handles.data(),
        no_events.size(), no_events.data(),
        &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueAcquireExternalMemObjectsKHR", "Enqueueing acquisition of "
        + std::to_string(memory_object_handles.size) + " on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(result_event_handle, queue, is_owning);
}

template <typename Container>
event_t enqueue_release_external_memory(queue_t queue, Container const& memory_object_handles)
{
    span<event::handle_t> no_events {};
    event::handle_t result_event_handle;
    auto status = clEnqueueReleaseExternalMemObjectsKHR(
        queue.handle(),
        memory_object_handles.size(),
        memory_object_handles.data(),
        no_events.size(),
        no_events.data(),
        &result_event_handle);
    if (not is_success(status)) {
#ifndef OCLW_DESTRUCTOR_NOEXCEPT
        if (not std::uncaught_exception()))
#endif
        throw runtime_error(
            status, "clEnqueueReleaseExternalMemObjectsKHR",
            "Scheduling the release of " + std::to_string(memory_object_handles.size())
            + " external memory objects on" + opencl::detail::identify(queue));
    }
    return event::detail::wrap_by_queue(result_event_handle, queue, is_owning);
}

inline queue_t by_handle(handle_t handle, bool owning = false)
{
    auto device_handle = get_device_handle(handle);
    auto context_handle = get_context_handle(handle);
    auto platform_handle = context::detail::get_platform_handle(context_handle);
    return wrap(platform_handle, context_handle, device_handle, handle, owning);
}

} // namespace detail
} // namespace queue

template <typename Container>
queue::event_t queue_t::enqueue_acquire_external_memory(Container const& memory_objects) const
{
    auto mem_object_handles = detail::get_handles(memory_objects);
    return queue::detail::enqueue_acquire_external_memory(*this, mem_object_handles);
}

template <typename Container>
queue::event_t queue_t::enqueue_release_external_memory(Container const& memory_objects) const
{
    auto mem_object_handles = detail::get_handles(memory_objects);
    return queue::detail::enqueue_release_external_memory(*this, mem_object_handles);
}

#endif // cl_khr_external_memory

inline queue::event_t queue_t::enqueue_free(void* ptr) const
{
    auto new_event_handle = memory::shared_virtual::detail::free(handle(), ptr);
    return event::detail::wrap_by_queue(new_event_handle, *this, is_owning);
}

inline queue::event_t queue_t::enqueue_free(memory::region_t region) const
{
    return enqueue_free(region.data());
}

inline queue::event_t queue_t::enqueue_free(memory::shared_virtual::region_t svm_region) const
{
    auto ptr = svm_region.data();
    span<void*> single_svm_pointer = { &ptr, 1 };
    auto no_callback = nullptr;
    auto no_user_data = nullptr;
    span<event::handle_t> no_events {};
    event::handle_t result_event_handle;
    auto status = clEnqueueSVMFree(
        handle(),
        single_svm_pointer.size(),
        single_svm_pointer.data(),
        no_callback, no_user_data,
        no_events.size(),
        no_events.data(),
        &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueSVMFree", "Failed enqueuing a free of "
        + memory::shared_virtual::detail::identify_region(svm_region) + " on " + detail::identify(*this));
    return event::detail::wrap_by_queue(result_event_handle, *this, is_owning);
}

inline queue::event_t queue_t::enqueue_map(memory::shared_virtual::region_t svm_region, memory::mapping::kind_t kind) const
{
    auto flags = memory::detail::make_map_flags(kind);
    event::handle_t result_event_handle;
    span<event::handle_t> no_events {};
    auto status = clEnqueueSVMMap(
        handle(),
        is_non_blocking,
        flags,
        svm_region.data(),
        svm_region.size(),
        no_events.size(),
        no_events.data(),
        &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueSVMMap", "Failed enqueuing a mapping of "
        + memory::shared_virtual::detail::identify_region(svm_region) + " for host access on "
        + detail::identify(*this));
    return event::detail::wrap_by_queue(result_event_handle, *this, is_owning);
}

inline queue::event_t queue_t::enqueue_map(memory::shared_virtual::region_t svm_region, memory::mapping::spec_t spec) const
{
    if (spec.subregion) {
        // Is this actually supported?
        auto sr = svm_region.subregion(spec.subregion->origin, spec.subregion->size);
        svm_region = memory::shared_virtual::region_t { sr.data(), sr.size() };
    }
    return enqueue_map(svm_region, spec.kind);
}


inline queue::event_t queue_t::enqueue_unmap(memory::shared_virtual::region_t svm_region) const
{
    event::handle_t result_event_handle;
    span<event::handle_t> no_events {};
    auto status = clEnqueueSVMUnmap(
        handle(), svm_region.data(), no_events.size(), no_events.data(), &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueSVMUnmap", "Failed enqueuing an unmap of "
        + memory::shared_virtual::detail::identify_region(svm_region) + " for host access on "
        + detail::identify(*this));
    return event::detail::wrap_by_queue(result_event_handle, *this, is_owning);
}

inline queue::event_t queue_t::enqueue_fill(memory::shared_virtual::region_t svm_region, memory::region_t pattern) const
{
    event::handle_t result_event_handle;
    span<event::handle_t> no_events {};
    auto status = clEnqueueSVMMemFill(
        handle_,
        svm_region.data(),
        pattern.data(),
        pattern.size(),
        svm_region.size(),
        no_events.size(),
        no_events.data(),
        &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueSVMMemFill", "Failed enqueuing a memory fill of "
        + memory::shared_virtual::detail::identify_region(svm_region) + " on "
        + detail::identify(*this));
    return event::detail::wrap_by_queue(result_event_handle, *this, is_owning);
}

template <typename T>
queue::event_t queue_t::enqueue_fill(memory::shared_virtual::region_t svm_region, T const& pattern) const
{
    static_assert(std::is_trivially_copyable<T>::value, "Attempt to use a non-trivially-copyable type as the fill pattern");
    auto pattern_ = memory::region_t { &pattern, sizeof(T) };
    return enqueue_fill(svm_region, pattern_);
}

inline queue::event_t queue_t::enqueue_fill(image_t const& image, image::box_spec_t box, void const* color) const
{
    event::handle_t event_handle;
    span<event::handle_t> no_events {};
    auto status = clEnqueueFillImage(
        handle_, image.handle(), color,
        box.origin.data(), box.extents.data(),
        no_events.size(), no_events.data(), &event_handle);
    // TODO: Consider saying something about the box. ALso, if the error regards exceeding image dimensions,
    // we could identify this and report the relevant error.
    throw_if_error_lazy(status, "clEnqueueFillImage", "");
    return event::detail::wrap_by_queue(event_handle, *this, is_owning);
}

namespace detail {

#if CL_VERSION_2_0
template <typename Container>
queue::event_t enqueue_migrate(
    bool_constant<false> migrating_svms,
    queue_t const& queue,
    Container && svms,
    memory::destination_t destination,
    bool maintain_contents)
{
    (void) migrating_svms;
    event::handle_t result_event_handle;
    auto flags = memory::migration::detail::make_flags(destination, maintain_contents);
    auto ptrs = memory::detail::get_ptrs(svms);
    auto sizes = memory::detail::get_sizes(svms);

    span<event::handle_t> no_events {};
    auto status = clEnqueueSVMMigrateMem(
        queue.handle(),
        ptrs.size(),
        const_cast<const void**>(ptrs.data()),
        sizes.data(),
        flags,
        no_events.size(),
        no_events.data(),
        &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueSVMMigrateMem", "Failed scheduling a migration of "
        + std::to_string(svms.size()) + " SVM regions on " + identify(queue));
    return event::detail::wrap_by_queue(result_event_handle, queue, is_owning);
}
#endif // CL_VERSION_2_0

template <typename Container>
queue::event_t enqueue_migrate(
    bool_constant<true> migrating_objects,
    queue_t const& queue,
    Container&& memory_objects,
    memory::destination_t destination,
    bool maintain_contents)
{
    (void) migrating_objects;
    event::handle_t result_event_handle;
    auto flags = memory::migration::detail::make_flags(destination, maintain_contents);
    auto handles = get_handles(memory_objects);

    span<event::handle_t> no_events {};
    auto status = clEnqueueMigrateMemObjects(
        queue.handle(),
        memory_objects.size(),
        handles.data(),
        flags,
        no_events.size(),
        no_events.data(),
        &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueMigrateMemObjects", "Failed scheduling a migration of "
        + std::to_string(memory_objects.size()) + " memory objects on " + identify(queue));
    return event::detail::wrap_by_queue(result_event_handle, queue, is_owning);
}

} // namespace detail

template <typename Container, typename>
queue::event_t queue_t::enqueue_migrate(
    Container&& objects_or_regions,
    memory::destination_t destination,
    bool maintain_contents) const
{
    using migrated_type = typename Container::value_type;
    static constexpr auto migrating_objects = std::is_base_of<memory::object_t, migrated_type>::value;
#if CL_VERSION_2_0
    static constexpr auto migrating_svms = std::is_same<memory::shared_virtual::region_t, migrated_type>::value;
    static_assert(migrating_objects or migrating_svms, "Attempt to migrate entities other than "
        "memory objects or shared virtual memory regions");
#endif
    static_assert(migrating_objects, "Attempt to migrate entities other than memory objects");
    return detail::enqueue_migrate(
        detail::bool_constant<migrating_objects>{},
        *this, std::forward<Container>(objects_or_regions), destination, maintain_contents);
}

#if CL_VERSION_2_0

inline queue::event_t queue_t::enqueue_migrate(
    memory::shared_virtual::region_t const& svm_region,
    memory::destination_t destination,
    bool maintain_contents) const
{
    span<memory::shared_virtual::region_t const> single_svm { &svm_region, 1 };
    detail::bool_constant<false> migrate_svms;
    return detail::enqueue_migrate(migrate_svms, *this, single_svm, destination, maintain_contents);
}

#endif // CL_VERSION_2_0

inline queue::event_t queue_t::enqueue_migrate(
    memory::object_t const& memory_object,
    memory::destination_t destination,
    bool maintain_contents) const
{
    span<memory::object_t const> single_object { &memory_object, 1 };
    detail::bool_constant<true> migrate_objects;
    return detail::enqueue_migrate(migrate_objects, *this, single_object, destination, maintain_contents);
}

inline platform_t queue_t::platform() const noexcept
{
    return platform::wrap(platform_handle_);
}

inline context_t queue_t::context() const noexcept
{
    auto device_handles = context::detail::get_device_handles(context_handle_);
    return context::wrap(platform_handle_, std::move(device_handles), context_handle_, is_not_owning);
}

inline context::device_t queue_t::device() const noexcept
{
    return context::device::wrap(platform_handle_, context_handle_, device_handle_);
}

inline size_t queue_t::capacity() const
{
    return info::get_scalar<CL_QUEUE_SIZE>(handle_);
}

inline bool queue_t::in_order_execution() const
{
    auto properties = info::get_scalar<CL_QUEUE_PROPERTIES>(handle_);
    return not (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
}

inline bool queue_t::profiling_enabled() const
{
    auto properties = info::get_scalar<CL_QUEUE_PROPERTIES>(handle_);
    return properties & CL_QUEUE_PROFILING_ENABLE;
}

/// Is this queue designated as the default queue for its device?
inline bool queue_t::is_device_default() const
{
    return info::get_scalar<CL_QUEUE_DEVICE_DEFAULT>(handle_) == handle_;
}

inline bool operator==(queue_t const& lhs, queue_t const& rhs) noexcept
{
    // TODO: Is it not sufficient to merely compare the handles? Handles should
    // be unique after all
    return lhs.platform_handle() == rhs.platform_handle() and
        lhs.context_handle() == rhs.context_handle() and
        lhs.device_handle() == rhs.device_handle() and
        lhs.handle() == rhs.handle();
}

template <typename Source, typename Destination>
queue::event_t queue_t::enqueue_copy_box(
    Source&&,
    box_spec_t,
    Destination&&,
    box_target_spec_t) const
{
    static_assert(false, "Not yet implemented");
    throw std::runtime_error("Not yet implemented");
}

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_QUEUE_HPP_
