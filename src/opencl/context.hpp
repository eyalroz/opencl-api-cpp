#ifndef OPENCL_WRAPPERS_CONTEXT_HPP_
#define OPENCL_WRAPPERS_CONTEXT_HPP_

#include "buffer.hpp"
#include "types.hpp"
#include "util/optional.hpp"

namespace opencl {

namespace context {

namespace detail {

platform::handle_t get_platform_handle(handle_t handle);
dynarray<device::handle_t> get_device_handles(handle_t handle);
context_t by_handles(platform::handle_t platform_handle, handle_t handle, bool owning);
context_t by_handle(handle_t handle, bool owning);

} // namespace detail

context_t wrap(
    platform::handle_t                    platform_handle,
    optional<dynarray<device::handle_t>>  device_handles,
    handle_t                              handle,
    bool                                  owning) noexcept;

context_t by_handle(handle_t handle, bool owning = false);

context_t create(opencl::device_t device);
template <template <typename> class Container>
context_t create(Container<opencl::device_t> const& devices);
/// Create a context with all devices (possibly of a given type) on a given platform
context_t create_with_all_devices(platform_t const& platform, optional<opencl::device::type_t> device_type = {});

} // namespace context

// TODO: Add support for context properties
class context_t {
public:
    using handle_type = context::handle_t;

protected:
    // Perhaps indicate if the context was created for all devices of a type, multiple devices or a single device
    platform::handle_t platform_handle_;

    // nullopt does not mean there are no devices, only that we haven't obtained them yet;
    // lazy construction of this class.
    // NOTE: Not thread-safe
    mutable optional<dynarray<device::handle_t>> device_handles_;
    context::handle_t handle_;
    detail::handle_ownership_token_t<context_t> ownership_token_;

public: // non-mutators
    platform::handle_t platform_handle() const noexcept { return platform_handle_; }
    span<const device::handle_t> device_handles() const;
    context::handle_t handle() const noexcept { return handle_; }

    platform_t platform() const noexcept;

    size_t num_devices() const;
    // TODO: Should we replicate the devices, or provide a gadget which observes the
    // device handles we have in here? Perhaps something like a lazy map, keyed by device ID?
    // the devices() method would assure it's initialized, then the user could just perform
    // a regular map lookup
    dynarray<context::device_t> devices() const;
    context::device_t get_device(device::index_t index) const;
    context::device_t first_device() const; // A poor man's "default device"

protected:
    context_t(
        platform::handle_t platform_handle,
        context::handle_t handle,
        bool owning) noexcept
   :
        platform_handle_(platform_handle), device_handles_(nullopt),
        handle_(handle), ownership_token_(owning, handle) { }

    context_t(
        platform::handle_t platform_handle,
        optional<dynarray<device::handle_t>> device_handles,
        context::handle_t handle,
        bool owning) noexcept
   :
        platform_handle_(platform_handle), device_handles_(std::move(device_handles)),
        handle_(handle), ownership_token_(owning, handle) { }

public:
    friend context_t context::wrap(
        platform::handle_t                    platform_handle,
        optional<dynarray<device::handle_t>>  device_handles,
        context::handle_t                     handle,
        bool                                  owning) noexcept;

    // image requirements and such

    // the notify function, destruction handle

    // buffer creation of all sorts
    buffer_t create_buffer(size_t size, memory::host_and_kernel_access_t access_spec = memory::full_access()) const;
    buffer_t create_buffer_using(memory::region_t host_side_storage, memory::host_and_kernel_access_t access_spec = memory::full_access()) const; // CL_MEM_USE_HOST_PTR
    buffer_t create_buffer_copy_of(memory::region_t region_to_copy, memory::host_and_kernel_access_t access_spec = memory::full_access()) const; // CL_MEM_COPY_HOST_PTR

    template <typename T> buffer_t create_buffer_using(span<T> host_side_storage, memory::host_and_kernel_access_t access_spec = memory::full_access()) const; // CL_MEM_USE_HOST_PTR and utilize sizeof T
    template <typename T> buffer_t create_buffer_copy_of(span<T> region_to_copy, memory::host_and_kernel_access_t access_spec = memory::full_access()) const; // CL_MEM_COPY_HOST_PTR and utilize sizeof T

    queue_t create_queue(device_t const& device) const;

    image::sampler_t create_sampler(bool normalized_coordinates, image::sampler::addressing_mode_t, image::sampler::filtering_mode_t) const;

    memory::region_t allocate_shared_virtual(size_t size) const;
    memory::region_t allocate_shared_virtual(
        size_t size,
        memory::shared_virtual::access_spec_t access_spec,
        memory::alignment_t alignment) const;

    //TODO: A method utilizing CL_MEM_ALLOC_HOST_PTR
}; // class context_t

inline bool operator==(const context_t& lhs, const context_t& rhs) noexcept
{
    return lhs.platform_handle() == rhs.platform_handle() and lhs.handle() == rhs.handle();
}

inline bool operator!=(const context_t& lhs, const context_t& rhs) noexcept { return not (lhs == rhs); }

} // namespace opencl

#endif // OPENCL_WRAPPERS_CONTEXT_HPP_
