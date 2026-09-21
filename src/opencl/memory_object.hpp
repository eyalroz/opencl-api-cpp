/**
 * @file
 *
 * @brief Definition of the @ref memory::object_t class, declarations
 * of its named constructor idioms, comparison operators and some related
 * utility functions.
 */
#ifndef OPENCL_WRAPPERS_MEMORY_OBJECT_HPP_
#define OPENCL_WRAPPERS_MEMORY_OBJECT_HPP_

#include "types.hpp"
#include "impl/ownership_token_t.hpp"
#include "util/region.hpp"
#include "util/optional.hpp"

namespace opencl {

namespace memory {

bool is_buffer(object_t const&);
bool is_image(object_t const&);
bool is_image_array(object_t const&);

object_t wrap(
    platform::handle_t platform_handle,
    context::handle_t  context_handle,
    memory::handle_t   handle,
    bool               owning) noexcept;

/// A common base class for OpenCL buffers, images and pipes - all of involve some
/// memory opaquely held OpenCL
class object_t {
public:
    using handle_type = handle_t;

protected:
    platform::handle_t platform_handle_;
    context::handle_t context_handle_;
    handle_type handle_;
    opencl::detail::handle_ownership_token_t<object_t> ownership_token_;

    memory::region_t host_region_ { nullptr }; // We could use optional, but this is nullable, so...

public:
    platform::handle_t platform_handle() const noexcept { return platform_handle_; }
    context::handle_t context_handle() const noexcept { return context_handle_; }
    memory::handle_t handle() const noexcept { return handle_; }
    bool owning() const noexcept { return ownership_token_.owning(); }

    platform_t platform() const noexcept;
    context_t context() const noexcept;

protected:
    object_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        memory::handle_t handle,
        bool owning) noexcept
    :
        platform_handle_(platform_handle), context_handle_(context_handle), handle_(handle),
        ownership_token_(owning, handle) {}

public:
    friend object_t wrap(
        platform::handle_t platform_handle,
        context::handle_t  context_handle,
        memory::handle_t   handle,
        bool               owning) noexcept;

    // methods based on getinfo
    size_t size_in_bytes() const;
    void* data() const; // on host...
	size_t mapping_count() const;
    size_t reference_count() const;
    optional<object_t> parent() const; // returns nullopt if this buffer is not a sub-buffer // image created from buffer
    optional<off_t> offset_in_parent() const; // returns nullopt if this buffer is not a sub-buffer
    bool is_shared_virtual_mem() const;

    object_t(object_t const&) = delete;
    object_t(object_t &&) = default;
    object_t& operator=(object_t const&) = delete;
    void friend swap(object_t& a, object_t& b) noexcept
    {
        std::swap(a.platform_handle_, b.platform_handle_);
        std::swap(a.context_handle_, b.context_handle_);
        std::swap(a.handle_, b.handle_);
        // This is the non-trivial part:
        std::swap(a.ownership_token_, b.ownership_token_);
    }
    object_t& operator=(object_t && other) noexcept
    {
       swap(*this, other);
        return *this;
    }
}; // class object_t

inline bool operator==(object_t const& lhs, object_t const& rhs) noexcept;
inline bool operator!=(object_t const& lhs, object_t const& rhs) noexcept { return not (lhs == rhs); }

} // namespace memory

// Notes:
// - The unmap enqueue is _always_ non-blocking.
// - See the two map() functions, for buffer_t and image_t
queue::event_t unmap(memory::object_t const& memory_object, void* mapping, queue_t const& queue);

template <typename T>
T upcast(memory::object_t const& memory_object);

} // namespace opencl

#endif // OPENCL_WRAPPERS_MEMORY_OBJECT_HPP_
