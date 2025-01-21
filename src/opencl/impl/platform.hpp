
#ifndef OPENCL_WRAPPERS_IMPL_PLATFORM_HPP_
#define OPENCL_WRAPPERS_IMPL_PLATFORM_HPP_

#include "../device.hpp"
#include "../platform.hpp"
#include "../error.hpp"
#include "../util/dynarray.hpp"

#include <algorithm>

namespace opencl {

inline char const* info::traits_t<info::platform>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case  CL_PLATFORM_PROFILE:                  return "profile";
    case  CL_PLATFORM_VERSION:                  return "version";
    case  CL_PLATFORM_NAME:                     return "name";
    case  CL_PLATFORM_VENDOR:                   return "vendor";
    case  CL_PLATFORM_EXTENSIONS:               return "extensions";
#ifdef CL_VERSION_2_1
    case  CL_PLATFORM_HOST_TIMER_RESOLUTION:    return "host time resolution";
#endif
#ifdef CL_VERSION_3_0
    case  CL_PLATFORM_NUMERIC_VERSION:          return "numeric version";
    case  CL_PLATFORM_EXTENSIONS_WITH_VERSION:  return "versioned extensions";
#endif
    default: return nullptr;
    }
}

namespace device {

inline char const* type_name(type_t type)
{
    switch (type) {
    case type_t::default_: return "default";
    case type_t::cpu: return "CPU";
    case type_t::gpu: return "GPU";
    case type_t::accelerator: return "accelerator";
#ifdef CL_VERSION_1_2
    case type_t::custom: return "custom";
#endif
    default: throw std::invalid_argument("Unknown OpenCL device type");
    }
}

inline char const* type_name(cl_device_type raw_type)
{
    return raw_type == CL_DEVICE_TYPE_ALL ? "all" : type_name(static_cast<type_t>(raw_type));
}

} // namespace device

namespace platform {

inline platform_t wrap(handle_t handle, optional<index_t> index) noexcept
{ return { handle, std::move(index) }; }

namespace detail {

inline value_or_status_t<size_t> count_nothrow()
{
    cl_uint count_;
    auto status = clGetPlatformIDs(0, nullptr, &count_);
    if (not is_success(status)) { return { {}, status }; }
    return { count_, {} };
}

} // namespace detail

inline size_t count()
{
    auto count_ = detail::count_nothrow();
    return_or_throw_if_error(count_, "clGetPlatformIDs", "Obtaining the number of platforms");
}

} // namespace platform

namespace platform {
namespace detail {

inline status_t get_handles_nothrow(span<handle_t> platform_handle_storage)
{
    return clGetPlatformIDs(platform_handle_storage.size(), platform_handle_storage.data(), nullptr);
}

inline void get_handles(span<handle_t> platform_handle_storage)
{
    auto status = get_handles_nothrow(platform_handle_storage);
    throw_if_error(status, "clGetPlatformIDs",
        "Obtaining the first " + std::to_string(platform_handle_storage.size()) + " OpenCL platform"
        + (platform_handle_storage.size() > 1 ? "s": ""));
}

inline handle_t get_handle_of_first()
{
    handle_t handle;
    span<handle_t> single_handle_span = {&handle, 1};
    get_handles(single_handle_span);
    return handle;
}

inline dynarray<handle_t> get_handles(size_t count)
{
    auto result = make_dynarray<handle_t>(count);
    get_handles(result);
    return result;
}

inline dynarray<handle_t> get_handles()
{
    auto count_ = count();
    auto result = make_dynarray<handle_t>(count_);
    get_handles(result);
    return result;
}

inline dynarray<handle_t> get_handles_nothrow()
{
    auto count_ = count_nothrow();
    auto result = make_dynarray<handle_t>(count_);
    get_handles(result);
    return result;
}

inline value_or_status_t<index_t> get_index_of_nothrow(handle_t handle)
{
    auto handles = get_handles_nothrow();
    auto iter = std::find(handles.begin(), handles.end(), handle);
    if (iter == handles.end()) { return { {}, status::invalid_value }; }
    return { static_cast<index_t>(iter - handles.begin()), status::success };
}

inline index_t get_index_of(handle_t handle)
{
    auto index_ = get_index_of_nothrow(handle);
    if (not index_) {
        throw std::invalid_argument("There is no OpenCL platform with handle " + opencl::detail::ptr_as_hex(handle));
    }
    return *index_;
}

inline std::chrono::nanoseconds host_timer_resolution(handle_t handle)
{
    auto raw = info::get_scalar<CL_PLATFORM_HOST_TIMER_RESOLUTION>(handle);
    return std::chrono::nanoseconds{ raw };
}

} // namespace detail

inline platform_t get(index_t index)
{
    auto count_ = count();
    if (index >= count_) {
        throw std::invalid_argument("Platform with index " + std::to_string(index) + " requested, but only "
            + std::to_string(count_) + " platforms are available");
    }
    auto handles = detail::get_handles(index + 1);
    return wrap(handles[index], index);
}

/*! \brief Gets the first available platform, returning it by value.
 *
 * \return Returns a valid platform if one is available.
 *         If no platform is available will return a null platform.
 * Throws an exception if no platforms are available
 * or an error condition occurs.
 * Wraps clGetPlatformIDs(), returning the first result.
 */
inline platform_t default_() { return wrap(detail::get_handle_of_first(), index_t{0}); }

inline timestamp_t get_host_time(platform_t const& platform)
{
    return device::get_host_time_for(platform.default_device());
}

namespace detail {

inline device::index_t num_devices(handle_t platform_handle, cl_device_type raw_device_type)
{
    enum { unused = 0 };
    device::index_t num_devices;
    auto status = clGetDeviceIDs(platform_handle, raw_device_type, unused, nullptr, &num_devices);
    if (status == CL_DEVICE_NOT_FOUND) {
        // Apparently, some OpenCL drivers throw this exception when you specify a kind of device they don't have
        return 0;
    }
    throw_if_error_lazy(status, "clGetDeviceIDs", "Determining the number of devices on " + opencl::detail::identify(platform_handle));
    return num_devices;
}

// Gets handles for the first @ref handles.size() devices of a given type;
// if there are less devices - this will fail; if there are more devices -
// that rest are ignored.
inline void get_device_handles(
    handle_t                platform_handle,
    cl_device_type          raw_type,
    span<device::handle_t>& handles)
{
    device::index_t num_obtained;
    auto status = clGetDeviceIDs(
        platform_handle,
        raw_type,
        handles.size(),
        handles.data(),
        &num_obtained);
    if (status == CL_DEVICE_NOT_FOUND) {
        // Apparently, some OpenCL drivers throw this exception when you specify a kind of device they don't have
        handles = span<device::handle_t>{};
        return;
    }
    throw_if_error_lazy(status, "clGetDeviceIDs", "Failed obtaining " + std::to_string(handles.size()) + " devices"
                        + device::type_name(raw_type) + " devices for " + opencl::detail::identify(platform_handle));
    if (num_obtained < handles.size()) {
        throw std::invalid_argument("A request for the handles of " + std::to_string(handles.size())
            + " devices "
            + (raw_type == CL_DEVICE_TYPE_ALL ? "" : "of type " + std::string{device::type_name(raw_type)})
            + "on " + opencl::detail::identify(platform_handle) + " returned only "
            + std::to_string(num_obtained) + " such devices");
    }
}

inline void get_device_handles(
    handle_t                platform_handle,
    span<device::handle_t>& handles)
{
    return get_device_handles(platform_handle, CL_DEVICE_TYPE_ALL, handles);
}

inline dynarray<device::handle_t> get_device_handles(
    handle_t            platform_handle,
    cl_device_type      type,
    device::index_t     num_devices)
{
    // Note we're relying on device handles being trivially-constructible
    auto handles = make_dynarray<device::handle_t>(num_devices);
    get_device_handles(platform_handle, type, handles);
    return handles;
}

inline dynarray<device::handle_t> get_device_handles(
    handle_t            platform_handle,
    device::index_t     num_devices)
{
    return get_device_handles(platform_handle, CL_DEVICE_TYPE_ALL, num_devices);
}

inline dynarray<device::handle_t> get_device_handles(
    handle_t            platform_handle,
    device::type_t      type,
    device::index_t     num_devices)
{
    return get_device_handles(platform_handle, static_cast<cl_device_type>(type), num_devices);
}

} // namespace detail

} // namespace platform

inline context_t platform_t::full_context(optional<device::type_t> type) const
{
    return context::create_with_all_devices(*this, type);
}


inline device::index_t platform_t::num_devices(device::type_t type) const
{
    return platform::detail::num_devices(handle_, static_cast<cl_device_type>(type));
}

inline device::index_t platform_t::num_devices() const
{
    return platform::detail::num_devices(handle_, CL_DEVICE_TYPE_ALL);
}

inline std::string platform_t::name() const
{
    return info::get_string<CL_PLATFORM_NAME>(handle_);
}

inline std::string platform_t::vendor() const
{
    return info::get_string<CL_PLATFORM_VENDOR>(handle_);
}

inline std::string platform_t::version_string() const
{
    return info::get_string<CL_PLATFORM_VERSION>(handle_);
}

inline platform::index_t platform_t::index() const
{
    if (not index_) { index_ = platform::detail::get_index_of(handle_); }
    return *index_;
}

inline std::string platform_t::opencl_profile() const
{
    return info::get_string<CL_PLATFORM_PROFILE>(handle_);
}

inline dynarray<std::string> platform_t::extensions() const
{
    // TODO:
    auto concatenated = info::get_string<CL_PLATFORM_EXTENSIONS>(handle_);
    auto num_spaces = std::count_if(concatenated.begin(), concatenated.end(), isspace);
    auto iss = std::istringstream{concatenated};
    auto generator = [&](size_t) { std::string str{}; iss >> str; return str; };
    return generate_dynarray<std::string>(num_spaces+1, generator);
}

#if CL_VERSION_2_1
inline std::chrono::nanoseconds platform_t::host_timer_resolution() const
{
    return platform::detail::host_timer_resolution(handle_);
}
#endif // CL_VERSION_2_1

#if CL_VERSION_3_0
inline dynarray<version::named_t> platform_t::supported_extensions() const
{
    auto raw = info::get_array<CL_PLATFORM_EXTENSIONS_WITH_VERSION>(handle_);
    return to_dynarray(raw, make_named_version);
}
#endif // CL_VERSION_3_0

inline dynarray<device_t> platform_t::devices(device::type_t type) const
{
    device::index_t num_devices_ = num_devices(type);
    auto device_handles = platform::detail::get_device_handles(handle_, type, num_devices_);
    auto device_by_index = [&](device::index_t i) { return device::wrap(handle_, device_handles[i]); };
    return generate_dynarray<device_t>(num_devices_, device_by_index);
}

inline dynarray<device_t> platform_t::devices() const
{
    device::index_t num_devices_ = num_devices();
    auto device_handles = platform::detail::get_device_handles(handle_, CL_DEVICE_TYPE_ALL, num_devices_);
    auto device_by_index = [&](device::index_t i) { return device::wrap(handle_, device_handles[i]); };
    return generate_dynarray<device_t>(num_devices_, device_by_index);
}

inline device_t platform_t::get_device(device::type_t type, device::index_t index) const
{
    auto num_devices_ = num_devices(type);
    if (num_devices_ == 0) {
        throw std::invalid_argument(std::string("Request for a ") + device::type_name(type)
            + " device on " + opencl::detail::identify(*this) + " which has no devices of this type");
    }
    auto num_handles_to_obtain = index + 1;
    if (num_devices_ < num_handles_to_obtain) {
        throw std::invalid_argument("Request for the " + std::string(type_name(type))
            + " device of index " + std::to_string(index) +
            " when there are only " + std::to_string(num_devices_) + " such devices on "
            + detail::identify(*this));
    }
    auto device_handles = platform::detail::get_device_handles(handle_, type, num_handles_to_obtain);
    return device::wrap(handle_, device_handles[index]);
}

inline device_t platform_t::get_device(device::index_t index) const
{
    auto num_devices_ = num_devices();
    if (num_devices_ == 0) {
        throw std::invalid_argument("Request for a device on " + detail::identify(*this)
            + " which has no devices");
    }
    auto num_handles_to_obtain = index + 1;
    if (num_devices_ < num_handles_to_obtain) {
        throw std::invalid_argument("Request for the device of index " + std::to_string(index) +
            " when there are only " + std::to_string(num_devices_) + " such devices on "
            + detail::identify(*this));
    }
    auto device_handles = platform::detail::get_device_handles(handle_, CL_DEVICE_TYPE_ALL, num_handles_to_obtain);
    return device::wrap(handle_, device_handles[index]);
}

inline device_t platform_t::default_device(device::type_t type) const
{
   return get_device(type, 0);
}

inline void platform_t::unload_compiler() const
{
    auto status = clUnloadPlatformCompiler(handle_);
    throw_if_error_lazy(status, "clUnloadPlatformCompiler", "Failed unloading the compiler of " + detail::identify(*this));
}

#if defined(CL_VERSION_3_0) || defined(cl_khr_extended_versioning)
inline version_t platform_t::version() const
{
#if defined(CL_VERSION_3_0)
    static constexpr auto info_param = CL_PLATFORM_NUMERIC_VERSION;
#else
    static constexpr auto info_param = CL_PLATFORM_NUMERIC_VERSION_KHR;
#endif
    auto raw_version = info::get_scalar<info_param>(handle_);
    return make_version(raw_version);
}
#endif

inline dynarray<platform_t> platforms()
{
    // auto status = clGetPlatformIDs(count_, handles_, nullptr);
    // throw_if_error_lazy(status, "clGetPlatformIDs", "Obtaining platform handles for the " + std::to_string(count_) + " available platforms");

    // auto count_ = platform::count();
    // auto handles_ = new platform::handle_t[count_];
    // platform::detail::get(span<platform::handle_t>{handles_,count_});
    auto handles = platform::detail::get_handles();
    auto wrap_handle_by_index = [&handles](size_t i) { return platform::wrap(handles[i]); };
    return generate_dynarray<platform_t>(handles.size(), wrap_handle_by_index);
}

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_PLATFORM_HPP_
