#ifndef OPENCL_WRAPPERS_PLATFORM_HPP_
#define OPENCL_WRAPPERS_PLATFORM_HPP_

#include "types.hpp"
#include "util/dynarray.hpp"

#ifdef CL_VERSION_2_1
#include <chrono>
#endif

namespace opencl {

namespace device {

char const* type_name(type_t type);

} // namespace device

namespace platform {

size_t count();

/*! \brief Gets the first available platform, returning it by value.
 *
 * \return Returns a valid platform if one is available.
 *         If no platform is available will return a null platform.
 * Throws an exception if no platforms are available
 * or an error condition occurs.
 * Wraps clGetPlatformIDs(), returning the first result.
 */
platform_t default_();
platform_t get(index_t index);
platform_t wrap(handle_t handle, optional<index_t> index = {}) noexcept;

} // namespace platform

// TODO: Replace this with an index-getter-backed facade, which
// in turn will use `platform::count()` and `platform::get()`
dynarray<platform_t> platforms();

// TODO: Should a platform hold its index? It's mostly (only?) useful for error reporting.
class platform_t {
public:
    using handle_type = platform::handle_t;
    using index_type = platform::index_t;

protected:
    handle_type handle_;
    mutable optional<index_type> index_;

    platform_t(handle_type handle, optional<index_type> index = {})
    : handle_(handle), index_(std::move(index)) { }

public:
    friend platform_t platform::wrap(handle_type handle, optional<index_type> index) noexcept;

    // Platforms aren't created or destroyed, so no need to restrict copying
    platform_t(platform_t const& other) noexcept = default;
    platform_t(platform_t&& other) noexcept = default;
    platform_t& operator=(platform_t const&) noexcept = default;
    platform_t& operator=(platform_t&&) noexcept = default;
    ~platform_t() noexcept = default;

    context_t full_context(optional<device::type_t> type) const;

    index_type  index() const;
    handle_type handle() const noexcept { return handle_; }
    std::string opencl_profile() const;
    std::string name() const;
#if defined(CL_VERSION_3_0) || defined(cl_khr_extended_versioning)
    version_t version() const;
#endif
    std::string version_string() const;
    std::string vendor() const;
    dynarray<std::string> extensions() const; // TODO: This is so wasteful :-(
#ifdef CL_VERSION_2_1
    std::chrono::nanoseconds host_timer_resolution() const;
#endif
#ifdef CL_VERSION_3_0
    dynarray<version::named_t> supported_extensions() const;
#endif

    // TODO: Add methods for KHR-extension-specific parameters:
    // CL_PLATFORM_COMMAND_BUFFER_CAPABILITIES_KHR
    // CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR
    // CL_PLATFORM_SEMAPHORE_TYPES_KHR
    // CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR
    // CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR
    // CL_PLATFORM_ICD_SUFFIX_KHR

    device::index_t num_devices(device::type_t type) const;
    device::index_t num_devices() const; /// of all types

    dynarray<device_t> devices(device::type_t type) const; // TODO: make this get_devices
    dynarray<device_t> devices() const;
    // TODO: Perhaps drop this and just use .devices(type)[index] ? Maybe even make devices be lazy?
    device_t get_device(device::type_t type, device::index_t index) const;
    device_t get_device(device::index_t index) const;
    device_t default_device(device::type_t type = device::type_t::default_) const;

#if defined(CL_HPP_USE_DX_INTEROP)
    // get Direct3D devices - which are not OpenCL devices
#endif

    void unload_compiler() const;
}; // class platform_t

inline bool operator==(const platform_t& lhs, const platform_t& rhs) noexcept { return lhs.handle() == rhs.handle(); }
inline bool operator!=(const platform_t& lhs, const platform_t& rhs) noexcept { return not (lhs == rhs); }

} // namespace opencl

#endif // OPENCL_WRAPPERS_PLATFORM_HPP_
