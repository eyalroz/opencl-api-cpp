
#ifndef OPENCL_WRAPPERS_IMPL_CONTEXT_HPP_
#define OPENCL_WRAPPERS_IMPL_CONTEXT_HPP_

#include "device.hpp"
#include "../context.hpp"
#include "../buffer.hpp"
#include "../context_device.hpp"
#include "../sampler.hpp"
#include "../shared_virtual_memory.hpp"
#include "platform.hpp"
#include "info.hpp"

#ifndef NDEBUG
#include <algorithm>
#endif

namespace opencl {

inline char const* info::traits_t<info::context>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case CL_CONTEXT_PLATFORM:                         return "platform handle";
#ifdef CL_VERSION_1_2
    case CL_CONTEXT_INTEROP_USER_SYNC:                return "interop user sync";
#endif
    case CL_CONTEXT_REFERENCE_COUNT:                  return "reference count";
    case CL_CONTEXT_DEVICES:                          return "devices";
    case CL_CONTEXT_PROPERTIES:                       return "properties";
#ifdef CL_VERSION_1_1
    case CL_CONTEXT_NUM_DEVICES:                      return "number of devices";
#endif
    default: return nullptr;
    }
}

namespace detail {

OCLW_DEFINE_HANDLE_TRAITS("context", context::handle_t, clReleaseContext, clRetainContext);

} // namespace detail

inline context::device_t context_t::first_device() const { return get_device(0); }

namespace context {

namespace detail {

inline size_t get_num_devices(handle_t handle)
{
    return info::get_scalar<CL_CONTEXT_NUM_DEVICES>(handle);
}

inline value_or_status_t<dynarray<device::handle_t>>
get_device_handles_nothrow(handle_t handle)
{
    return info::get_array_nothrow<CL_CONTEXT_DEVICES>(handle);
}

inline dynarray<device::handle_t> get_device_handles(handle_t handle)
{
    return info::get_array<CL_CONTEXT_DEVICES>(handle);
}

inline value_or_status_t<platform::handle_t> get_platform_handle_nothrow(handle_t handle)
{
    auto properties = info::get_array_nothrow<CL_CONTEXT_PROPERTIES>(handle);
    if (not properties) { return { {}, properties.status }; }
    // Note that the size may be 0!
    auto maybe_platform = info::detail::lookup_property<cl_platform_id>(properties->data(), CL_CONTEXT_PLATFORM);
    if (maybe_platform) { return { *maybe_platform, status::success }; }
    // Damn OpenCL! We have to do it the hard and riciulous way now
    auto device_handles = get_device_handles_nothrow(handle);
    if (not device_handles) { return { {}, device_handles.status }; }
    return info::get_scalar_nothrow<CL_DEVICE_PLATFORM>(device_handles->front());
}

inline platform::handle_t get_platform_handle(handle_t handle)
{
    auto properties = info::get_array<CL_CONTEXT_PROPERTIES>(handle);
    // Note that the size may be 0!
    auto maybe_platform = info::detail::lookup_property<cl_platform_id>(properties.data(), CL_CONTEXT_PLATFORM);
    if (maybe_platform) { return *maybe_platform; }
    // Damn OpenCL! We have to do it the hard and riciulous way now
    auto device_handles = get_device_handles(handle);
    return info::get_scalar<CL_DEVICE_PLATFORM>(device_handles.front());
}

inline context_t by_handles(platform::handle_t platform_handle, handle_t handle, bool owning)
{
    auto device_handles = get_device_handles(handle);
    return wrap(platform_handle, std::move(device_handles), handle, owning);
}

inline context_t by_handle(handle_t handle, bool owning = false)
{
    platform::handle_t platform_handle = get_platform_handle(handle);
    return by_handles(platform_handle, handle, owning);
}

inline handle_t create(
    cl_context_properties const * marshalled_properties,
    span<device::handle_t> device_handles)
{
    if (device_handles.empty()) {
        throw std::invalid_argument("Attempt to create an OpenCL context with no devices");
    }
    status_t status;
    auto result = clCreateContext(
        marshalled_properties,
        device_handles.size(), device_handles.data(),
        // void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
        // void* user_data,
        nullptr, nullptr,
        &status);
    // TODO: Consider printing more text here
    throw_if_error_lazy(status, "clCreateContext", "Failed creating a context with "
        + std::to_string(device_handles.size()) + " devices");
    return result;
}

inline handle_t create(
    platform::handle_t platform_handle,
    span<device::handle_t> device_handles)
{
    auto single_prop = info::detail::single_property<cl_context_properties>(CL_CONTEXT_PLATFORM, platform_handle);
    return create(single_prop.data(), device_handles);
}

inline handle_t create_from_all_devices(
    cl_context_properties const * marshalled_properties,
    cl_device_type raw_device_type)
{
    status_t status;
    static constexpr auto no_callback = nullptr;
    static constexpr auto no_user_data = nullptr;
#ifndef NDEBUG
    auto platform_handle = info::detail::lookup_property<platform::handle_t>(marshalled_properties, CL_CONTEXT_PLATFORM);
    if (not platform_handle) {
        throw std::invalid_argument("Conntext create a context for a device type without specifying the platform");
    }
#endif
    auto result = clCreateContextFromType(
        marshalled_properties,
        raw_device_type,
        no_callback, no_user_data,
        &status);
    throw_if_error_lazy(status, "clCreateContextFromType",
        std::string{"Failed creating a context with the devices of type "}
        + opencl::device::type_name(raw_device_type) + " devices");
    return result;
}

inline handle_t create_from_all_devices(platform::handle_t platform_handle, cl_device_type raw_device_type)
{
    auto single_prop = info::detail::single_property<cl_context_properties>(CL_CONTEXT_PLATFORM, platform_handle);
    return create_from_all_devices(single_prop.data(), raw_device_type);
}

} // namespace detail

inline context_t wrap(
    platform::handle_t                   platform_handle,
    optional<dynarray<device::handle_t>> device_handles,
    handle_t                             handle,
    bool                                 owning) noexcept
{
    return context_t { platform_handle, std::move(device_handles), handle, owning };
}

// TODO: What about properties? :-(
template <template <typename> class Container>
context_t create(Container<opencl::device_t> const& devices)
{
    if (devices.empty()) {
        throw std::invalid_argument("Attempt to create an OpenCL context with no devices");
    }
    auto get_handle = [&](device_t const& dev) { return dev.handle(); };
    auto dev_handles = to_dynarray<device::handle_t>(devices, get_handle);
    auto handle = detail::create(dev_handles);
    auto shared_common_platform_handle = devices[0].platform_handle();
#ifndef NDEBUG
    if (devices.size() > 1) {
        auto shares_common_platform = [&](opencl::device_t const & dev) {
            return dev.platform_handle() == shared_common_platform_handle;
        };
        if (not std::all_of(devices.begin() + 1, devices.end(), shares_common_platform)) {
            throw std::invalid_argument("Attempt to create a context using devices from different platforms");
        }
    }
#endif
    return wrap(shared_common_platform_handle,std::move(dev_handles), handle, is_owning);
}

inline context_t create(opencl::device_t device)
{
    auto device_handle = device.handle();
    auto single_handle = span<opencl::device::handle_t>{ &device_handle, 1};
    auto context_handle = detail::create(device.platform_handle(), single_handle);
    return wrap(device.platform_handle(), {}, context_handle, is_owning);
}

inline context_t create_with_all_devices(platform_t const& platform, optional<opencl::device::type_t> device_type)
{
    cl_device_type raw_type = device_type ? opencl::detail::to_underlying(*device_type) : CL_DEVICE_TYPE_ALL;
    auto handle = detail::create_from_all_devices(platform.handle(), raw_type);
    auto platform_handle = detail::get_platform_handle(handle);
    return wrap(platform_handle, {}, handle, is_owning);
}

} // namespace context

inline span<device::handle_t const> context_t::device_handles() const
{
    if (not device_handles_) {
        device_handles_ = context::detail::get_device_handles(handle_);
    }
    return *device_handles_;
}

inline size_t context_t::num_devices() const
{
    return context::detail::get_num_devices(handle_);
}

inline context::device_t context_t::get_device(device::index_t index) const
{
//    auto handles_up_to_index = context::detail::get_array_info_parameter<device::handle_t>(handle_, CL_CONTEXT_DEVICES, index + 1);
//    auto device_handle = handles_up_to_index[index];
    auto device_handle = device_handles()[index];
    return context::device::wrap(platform_handle_, handle_, device_handle);
}

inline platform_t context_t::platform() const noexcept { return platform::wrap(platform_handle()); }

inline dynarray<context::device_t> context_t::devices() const
{
    auto const& actual_handles = device_handles();
    auto handle_to_dev = [&](device::handle_t device_handle) {
        return context::device::wrap(platform_handle_, handle_, device_handle);
    };
    return to_dynarray<context::device_t>(actual_handles, handle_to_dev);
}


inline buffer_t context_t::create_buffer(size_t size, buffer::host_and_kernel_access_t access_spec) const
{
    return buffer::create(*this, size, access_spec);
}

inline buffer_t context_t::create_buffer_using(memory::region_t host_side_storage, buffer::host_and_kernel_access_t access_spec) const
{
    return buffer::create_using(*this, host_side_storage, access_spec);
}
inline buffer_t context_t::create_buffer_copy_of(memory::region_t region_to_copy, buffer::host_and_kernel_access_t access_spec) const
{
    return buffer::create_copy_of(*this, region_to_copy, access_spec);
}

template <typename T>
buffer_t context_t::create_buffer_using(span<T> host_side_storage, buffer::host_and_kernel_access_t access_spec) const
{
    return buffer::create_using(*this, host_side_storage, access_spec);
}

template <typename T>
buffer_t context_t::create_buffer_copy_of(span<T> region_to_copy, buffer::host_and_kernel_access_t access_spec) const
{
    return buffer::create_copy_of(*this, region_to_copy, access_spec);
}

inline queue_t context_t::create_queue(device_t const& device) const
{
    return queue::create(*this, device);
}

inline image::sampler_t context_t::create_sampler(
    bool normalized_coordinates,
    image::sampler::addressing_mode_t addressing_mode,
    image::sampler::filtering_mode_t filtering_mode) const
{
    return image::sampler::create(*this, normalized_coordinates, addressing_mode, filtering_mode);
}

inline memory::region_t context_t::allocate_shared_virtual(size_t size) const
{
    return memory::shared_virtual::allocate(*this, size);
}

inline memory::region_t context_t::allocate_shared_virtual(
    size_t size,
    memory::shared_virtual::access_spec_t access_spec,
    memory::alignment_t alignment) const
{
    return memory::shared_virtual::allocate(*this, size, access_spec, alignment);
}

inline bool operator==(const context_t& lhs, const context_t& rhs) noexcept
{
    return lhs.platform_handle() == rhs.platform_handle() and lhs.handle() == rhs.handle();
}

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_CONTEXT_HPP_
