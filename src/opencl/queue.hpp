#ifndef OPENCL_WRAPPERS_QUEUE_HPP_
#define OPENCL_WRAPPERS_QUEUE_HPP_

#include "types.hpp"

namespace opencl {

namespace queue {

namespace detail {

device::handle_t get_device_handle(handle_t handle);
context::handle_t get_context_handle(handle_t handle);

} // namespace detail

queue_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    device::handle_t    device_handle,
    handle_t            handle,
    bool                owning) noexcept;

queue_t create(context::device_t const& device);
queue_t create(context_t const& context, device_t const& device);

} // namespace queue

template <
    typename NativeKernelFunction,
    typename BufferParameterPositionContainer>
queue::event_t enqueue_native_kernel_launch(
    queue_t const& queue,
    NativeKernelFunction kernel_function,
    span<void*> buffers,
    BufferParameterPositionContainer&& buffer_param_positions,
    span<void*> pointers_to_arguments);

template <typename EventContainer>
queue::event_t enqueue_marker(queue_t const& queue, EventContainer&& events = {});

// Note: A barrier is a marker which also enforces execution order: No command
// after the barrier can begin to execute until the commands before the barrier on
// the queue, and the events in the list, have concluded.
template <typename EventContainer>
queue::event_t enqueue_barrier(queue_t const& queue, EventContainer&& events = {});

// TODO: Add support for queue properties
class queue_t {
public:
    using handle_type = queue::handle_t;
    using event_type = queue::event_t;

protected:
    platform::handle_t platform_handle_;
    context::handle_t context_handle_;
    device::handle_t device_handle_;
    handle_type handle_;
    detail::handle_ownership_token_t<queue_t> ownership_token_;

public:
    platform::handle_t platform_handle() const noexcept { return platform_handle_; }
    context::handle_t context_handle() const noexcept { return context_handle_; }
    device::handle_t device_handle() const noexcept { return device_handle_; }
    handle_type handle() const noexcept { return handle_; }

    platform_t platform() const noexcept;
    context_t context() const noexcept;
    context::device_t device() const noexcept;
    bool owning() const noexcept { return ownership_token_.owning(); }

protected:
    queue_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        device::handle_t device_handle,
        queue::handle_t handle,
        bool owning) noexcept
    :
        platform_handle_(platform_handle), context_handle_(context_handle), device_handle_(device_handle),
        handle_(handle), ownership_token_(owning, handle) {}

public:
    friend queue_t queue::wrap(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        device::handle_t  device_handle,
        queue::handle_t   handle,
        bool              owning) noexcept;

    size_t capacity() const;
    bool in_order_execution() const;
    bool profiling_enabled() const;
    bool is_device_default() const;

    // enqueue methods
    // TODO: Move into an enqueue gadget

    // Note: this must cover the following kinds of endpoints:
    //
    //   - non-OpenCL memory region
    //   - Buffer
    //   - Image, 2D
    //   - Image, 3D
    //
    // and the possible combinations are 3 + 4 + 3 + 3 = 13 different
    // OpenCL API functions
    //
    // ... but not also a specification of a rectangle/region. Although that might be
    // relegated to copy parameters only.
    template <typename Source, typename Destination>
    event_type enqueue_copy(Source&& source, Destination&& destination) const;

    // TODO: Consider unifying with enqueue_copy - either as a method or just the implementation
    template <typename Source, typename Destination>
    event_type enqueue_copy_box(
        Source&& source,
        box_spec_t source_box,
        Destination&& destination,
        box_target_spec_t box_target) const;

    /// Q: Why don't these methods take a blocking/non-blocking parameter?
    /// A: Because that's stupid. It's a queue. Enqueuing things is not blocking. If
    ///    you want to wait for the operation to conclude, just wait on the resulting
    ///    event like with every other queue operation.
    std::pair<event_type, memory::region_t>
    enqueue_map(buffer_t const& buffer, memory::mapping::spec_t const& spec = {}) const;
    std::pair<event_type, memory::region_t>
    enqueue_map(image_t const& image, image::mapping::spec_t const & spec) const;
    event_type enqueue_unmap(memory::object_t const& memory_object, void* mapping) const;

    /**
     * A barrier ensures all commands enqueued before it have completed execution before any command
     * enqueued after it begins execution. By default, OpenCL queues have this behavior for
     * every command, and barriers are unnecessary; but a queue may be configured for out-of-order
     * execution, in which case it is only the barriers which enforce such order generally, rather
     * than w.r.t. explicitly-specified events.
     */
    event_type enqueue_barrier() const;

    /**
     * Unelike a barrier, a marker only enforces execution order on some specific events:
     * Whatever depends on the marker, depends transitively on the markers' dependents and will
     * not begin execution until those dependents have concluded.
     */
    template <typename EventContainer = span<event_t>>
    event_type enqueue_marker(EventContainer&& events = {}) const;

#ifdef cl_khr_external_memory
    template <typename Container = span<memory::object_t>>
    event_type enqueue_acquire_external_memory(Container const& memory_objects) const;
    template <typename Container = span<memory::object_t>>
    event_type enqueue_release_external_memory(Container const& memory_objects) const;
#endif

    // event_type enqueue_kernel_launch(kernel_t const& kernel, kernel::launch_configuration_t launch_config);
    template <typename... Ts>
    event_type enqueue_kernel_launch(
        kernel_t const& kernel,
        kernel::launch_configuration_t launch_config,
        Ts&&... additional_arguments) const;

    template <
        typename NativeKernelFunction,
        typename BufferParameterPositionContainer>
    event_type enqueue_native_kernel_launch(
        NativeKernelFunction kernel_function,
        span<void*> buffers,
        BufferParameterPositionContainer&& buffer_param_positions,
        span<void*> pointers_to_arguments) const;

    // TODO: Implement the following:
    // template <typename F, typename... Params>
    // event_type enqueue_native_kernel_launch(queue_t const& queue, F kernel_function, Params&&... params) const;


    template <typename Invokable>
    event_type enqueue_pure_host_invokable(Invokable& invokable) const;

    event_type enqueue_free(void* ptr) const;
    event_type enqueue_free(memory::region_t region) const;
    event_type enqueue_free(memory::shared_virtual::region_t svm_region) const;
    event_type enqueue_map(memory::shared_virtual::region_t svm_region, memory::mapping::kind_t kind) const;
    event_type enqueue_map(memory::shared_virtual::region_t svm_region, memory::mapping::spec_t spec = {}) const;
    event_type enqueue_unmap(memory::shared_virtual::region_t svm_region) const;
    event_type enqueue_fill(memory::shared_virtual::region_t svm_region, memory::region_t pattern) const;
    template <typename T>
    event_type enqueue_fill(memory::shared_virtual::region_t svm_region, T const& pattern) const;
    event_type enqueue_fill(image_t const& image, image::box_spec_t box, void const* color) const;

    /**
    * Migrate memory objections to the device or the host
    *
    * @tparam Container Any C++ container, with value type being either a memory object
    * or shared virtual memory region
    *
    * @param[in] objects_or_regions
    *     The memory objects or shared virtual memory regions to migrate.
    * @param[in] destination
    *     A choice between migrating to the device on a queue of which the migration
    *    is enqueued, and migrating to host memory.
    * @param[in] maintain_contents
    *     When true, the buffer/memory region content is usable after the migration;
    *     when false, the content is undefined; that makes the migration cheaper and
    *     is intended for when an overwrite is expected bool maintain_content;
    */
    template <typename Container, typename = detail::void_t<typename Container::value_type>>
    event_type enqueue_migrate(
        Container&& objects_or_regions,
        memory::destination_t destination,
        bool maintain_contents = true) const;

    event_type enqueue_migrate(
        memory::object_t const& memory_object,
        memory::destination_t destination,
        bool maintain_contents = true) const;

#if CL_VERSION_2_0
    event_type enqueue_migrate(
        memory::shared_virtual::region_t const& svm_region,
        memory::destination_t destination,
        bool maintain_contents = true) const;
#endif // CL_VERSION_2_0

    void flush() const;
    void finish() const;
}; // class queue_t

inline bool operator==(queue_t const& lhs, queue_t const& rhs) noexcept;
inline bool operator!=(queue_t const& lhs, queue_t const& rhs) noexcept { return not (lhs == rhs); }

} // namespace opencl

#endif // OPENCL_WRAPPERS_QUEUE_HPP_
