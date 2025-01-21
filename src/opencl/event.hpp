#ifndef OPENCL_WRAPPERS_EVENT_HPP_
#define OPENCL_WRAPPERS_EVENT_HPP_

#include "types.hpp"
#include "context.hpp"
#include "queue.hpp"

namespace opencl {

void wait_for(event_t const& event);

template <typename EventContainer>
void wait_for(EventContainer const& events);

namespace event {

event_t from_handle(handle_t handle);
user_event_t create();

user_event_t wrap_user_event(
    platform::handle_t        platform_handle,
    context::handle_t         context_handle,
    handle_t                  handle,
    bool                      owning) noexcept;

using raw_callback_t = void (CL_CALLBACK *)(
    cl_event event,
    cl_int   event_command_status,
    void *   user_data);


namespace detail {
inline char const* name_of(execution_status_t status);

} // namespace detail

} // namespace event

namespace queue {
namespace event {

event_t wrap(
    platform::handle_t        platform_handle,
    context::handle_t         context_handle,
    device::handle_t          device_handle,
    queue::handle_t           queue_handle,
    opencl::event::handle_t   handle,
    bool                      owning) noexcept;

namespace detail {
event_t wrap_by_queue(opencl::event::handle_t handle, queue_t const& queue, bool owning);
}

}
}

// An abstract class for user events and queue events
class event_t  {
public:
    using handle_type = opencl::event::handle_t;

protected:
    platform::handle_t platform_handle_;
    context::handle_t context_handle_;
    handle_type handle_;
    detail::handle_ownership_token_t<event_t> ownership_token_;

public:
    platform::handle_t platform_handle() const noexcept { return platform_handle_; }
    context::handle_t context_handle() const noexcept { return context_handle_; }
    event::handle_t handle() const noexcept { return handle_; }
    bool owning() const noexcept { return ownership_token_.owning(); }

    platform_t platform() const noexcept;
    context_t context() const noexcept;

protected:
    event_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        event::handle_t handle,
        bool owning) noexcept
    :
        platform_handle_(platform_handle), context_handle_(context_handle), handle_(handle), ownership_token_(owning, handle)
    { }

public:
    // get-info-based methods

    event::execution_status_t execution_status() const;

    //clGetEventProfilingInfo-related methods
}; // class event_t

// TODO: Why should user events be constructible without already setting their callback?
class user_event_t : public event_t {
public:
    void set_execution_status(event::execution_status_t) const;

    friend user_event_t event::wrap_user_event(
        platform::handle_t  platform_handle,
        context::handle_t   context_handle,
        event::handle_t     handle,
        bool                owning) noexcept;

    using event_t::event_t;

    user_event_t(user_event_t const& other) = delete;
    user_event_t(user_event_t&& other) = default;

    // Q: Why is the invokable non-const?
    // A: Because, generally, the invokable may change its own state. But TBH, I'm not 100%
    //    sure that argument is strong enough.
    template <typename Invokable>
    void set_callback(event::execution_status_t trigger, Invokable& invokable) const;

    void set_callback(event::execution_status_t trigger, event::raw_callback_t callback, void* user_data) const;
};

namespace queue {
class event_t : public opencl::event_t {
public:
    using parent_type = opencl::event_t;

protected:
    device::handle_t device_handle_;
    queue::handle_t queue_handle_;

public:
    device::handle_t device_handle() const noexcept { return device_handle_; }
    queue::handle_t queue_handle() const noexcept { return queue_handle_; }

    context::device_t device() const noexcept;
    queue_t queue() const noexcept;

protected:
    event_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        device::handle_t device_handle,
        queue::handle_t queue_handle,
        opencl::event::handle_t handle,
        bool owning) noexcept
    :
        parent_type(platform_handle, context_handle, handle, owning),
        device_handle_(device_handle), queue_handle_(queue_handle)
    { }

public:
    friend event_t event::wrap(
        platform::handle_t        platform_handle,
        context::handle_t         context_handle,
        device::handle_t          device_handle,
        queue::handle_t           queue_handle,
        opencl::event::handle_t   handle,
        bool                      owning) noexcept;

    // get-info-based methods

    command_type_t command_type() const;
}; // class event_t

} // namespace queue

inline bool operator==(event_t const& lhs, event_t const& rhs) noexcept;
inline bool operator!=(event_t const& lhs, event_t const& rhs) noexcept { return not (lhs == rhs); }

} // namespace opencl

#endif // OPENCL_WRAPPERS_EVENT_HPP_
