
#ifndef OPENCL_WRAPPERS_IMPL_EVENT_HPP_
#define OPENCL_WRAPPERS_IMPL_EVENT_HPP_

#include "../event.hpp"
#include "../queue.hpp"
#include "../context.hpp"
#include "ownership_token_t.hpp"

#ifndef NDEBUG
#endif

namespace opencl {
namespace detail {

OCLW_DEFINE_HANDLE_TRAITS("event", event::handle_t, clReleaseEvent, clRetainEvent);

} // namespace detail

namespace event {
namespace detail {

inline context::handle_t get_context_handle(handle_t handle)
{
    context::handle_t context_handle;
    auto status = clGetEventInfo(handle, CL_EVENT_CONTEXT, sizeof(context_handle), &context_handle, nullptr);
    throw_if_error_lazy(status, "clGetEventInfo", "To obtain an event's context");
    return context_handle;
}

} // namespace detail
} // namespace event

namespace queue {
namespace event {
namespace detail {

// Like @ref @wrap, but when what you have in hand is a command queue - which is
// how we very typically get event handles
inline event_t wrap_by_queue(opencl::event::handle_t handle, queue_t const& queue, bool owning)
{
    return wrap(
        queue.platform_handle(),
        queue.context_handle(),
        queue.device_handle(),
        queue.handle(),
        handle,
        owning);
}

inline queue::handle_t get_queue_handle(opencl::event::handle_t handle)
{
    queue::handle_t queue_handle;
    auto status = clGetEventInfo(handle, CL_EVENT_COMMAND_QUEUE, sizeof(queue_handle), &queue_handle, nullptr);
    throw_if_error_lazy(status, "clGetEventInfo", "To obtain an event's queue");
    return queue_handle;
}

} // namespace detail

inline event_t wrap(
    platform::handle_t        platform_handle,
    context::handle_t         context_handle,
    device::handle_t          device_handle,
    queue::handle_t           queue_handle,
    opencl::event::handle_t   handle,
    bool                      owning) noexcept
{
    return event_t { platform_handle, context_handle, device_handle, queue_handle, handle, owning };
}

// TODO: What about properties? :-(
inline event_t from_handle(opencl::event::handle_t handle)
{
    auto context_handle = opencl::event::detail::get_context_handle(handle);
    auto platform_handle = context::detail::get_platform_handle(context_handle);
    auto queue_handle = detail::get_queue_handle(handle);
    auto device_handle = queue::detail::get_device_handle(queue_handle);
    return wrap(platform_handle, context_handle, device_handle, queue_handle, handle, is_owning);
}

} // namespace event

} // namespace queue

namespace event {

namespace detail {

using queue::event::detail::wrap_by_queue;

inline handle_t create(context::handle_t context_handle)
{
    status_t status;
    auto event_handle = clCreateUserEvent(context_handle, &status);
    throw_if_error_lazy(status, "clCreateUserEvent", "in " + opencl::detail::identify(context_handle));
    return event_handle;
}

} // namespace detail

inline user_event_t wrap_user_event(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    event::handle_t     handle,
    bool                owning) noexcept
{
    return user_event_t { platform_handle, context_handle, handle, owning };
}

inline user_event_t create(context_t const& context)
{
    auto handle = detail::create(context.handle());
    return wrap_user_event(context.platform_handle(), context.handle(), handle, is_owning);
}

// TODO: Perhaps a create_on_queue() ? Or just let that be a queue.hpp method?

namespace detail {

inline char const* name_of(execution_status_t status)
{
    using es = execution_status_t;
    switch(status) {
    case es::queued: return "queued";
    case es::submitted: return "submitted to device";
    case es::running: return "running";
    case es::complete: return "completed execution";
    default:
        throw std::invalid_argument("Invalid execution status + " + std::to_string(static_cast<int>(status)));
    }
}

} // namespace detail

} // namespace event

inline platform_t event_t::platform() const noexcept
{
    return platform::wrap(platform_handle_);
}

inline context_t event_t::context() const noexcept
{
    // TODO: Fix event gathering issue
    return context::wrap(platform_handle_, {}, context_handle_, is_not_owning);
}

inline event::execution_status_t event_t::execution_status() const
{
    auto raw_status = info::get_scalar<CL_EVENT_COMMAND_EXECUTION_STATUS>(handle_);
    return static_cast<event::execution_status_t>(raw_status);
}

namespace queue {

inline context::device_t event_t::device() const noexcept
{
    return context::device::wrap(platform_handle_, context_handle_, device_handle_);
}

inline queue_t event_t::queue() const noexcept
{
    return wrap(platform_handle_, context_handle_, device_handle_, queue_handle_, is_not_owning);
}

inline command_type_t event_t::command_type() const
{
    return static_cast<command_type_t>(info::get_scalar<CL_EVENT_COMMAND_TYPE>(handle_));
}

} // namespace queue

namespace detail {

inline void wait_for(span<event::handle_t> event_handles)
{
    auto status = clWaitForEvents(event_handles.size(), event_handles.data());
    throw_if_error_lazy(status, "clWaitForEvents", "Waiting for " + std::to_string(event_handles.size()) + " events");
}

inline status_t wait_for(event::handle_t const &event_handle)
{
    return clWaitForEvents(1, &event_handle);
}

} // namespace detail

inline void wait(event_t const &event)
{
    auto handle = event.handle();
    auto status = clWaitForEvents(1, &handle);
    throw_if_error_lazy(status, "clWaitForEvents", "Waiting for " + detail::identify(event));
}

template <typename EventContainer>
void wait_for(EventContainer const& events)
{
    auto event_handles = detail::get_handles(events);
    detail::wait_for(event_handles);
}

inline bool operator==(event_t const& lhs, event_t const& rhs) noexcept
{
    // TODO: Is it not sufficient to merely compare the handles? Handles should be unique after all
    return lhs.platform_handle() == rhs.platform_handle() and
        lhs.context_handle() == rhs.context_handle() and
        lhs.handle() == rhs.handle();
}


enum class execution_status_t {
    queued                 = CL_QUEUED,
    in_queue = queued, enqueued = queued,
    submitted              = CL_SUBMITTED,
    submitted_to_device    = submitted,
    running                = CL_RUNNING,
    in_progress = running, in_flight = running,
    complete               = CL_COMPLETE,
    completed = complete, done = complete, executed = complete
};



inline void user_event_t::set_execution_status(event::execution_status_t execution_status) const
{
    auto status = clSetUserEventStatus(handle_, static_cast<cl_int>(execution_status));
    throw_if_error_lazy(status, "clSetUserEventStatus",
        "Setting the execution status of " + detail::identify(*this) + " to " + event::detail::name_of(execution_status));
}

template <typename Invokable>
void CL_CALLBACK reimbue_with_type_and_invoke(void* type_erased_invokable)
{
    auto invokable = static_cast<Invokable*>(type_erased_invokable);
    (*invokable)();
}

template <typename F>
void user_event_t::set_callback(event::execution_status_t trigger, F& invokable) const
{
    return set_callback(trigger, reimbue_with_type_and_invoke<F>, &invokable);
}

inline void user_event_t::set_callback(
    event::execution_status_t trigger,
    event::raw_callback_t callback,
    void* user_data) const
{
    auto status = clSetEventCallback(handle_, static_cast<cl_int>(trigger), callback, user_data);
    throw_if_error_lazy(status, "clSetEventCallback",
        "setting a callback for " + detail::identify(*this)
        + " triggered at " + event::detail::name_of(trigger));
}


} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_EVENT_HPP_
