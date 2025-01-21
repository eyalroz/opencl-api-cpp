#ifndef OPENCL_WRAPPERS_DEVICE_HPP_
#define OPENCL_WRAPPERS_DEVICE_HPP_

#include "types.hpp"

namespace opencl {

namespace device {

class capabilities_t;

device_t wrap(platform::handle_t, handle_t) noexcept;

synchronized_timestamps_t get_synchronized_time_with(device_t const& device);
timestamp_t get_host_time_for(device_t const& device);

} // namespace device

struct vendor_t {
    vendor_id_t id;
#ifdef CL_VERSION_2_0
    std::string name;
#endif
};

// A non-owning, copyable, reference type! ... unlike most other wrapper types
class device_t {
public:
    using handle_type = device::handle_t;

protected:
    platform::handle_t  platform_handle_;
    handle_type         handle_;

public:
    platform::handle_t platform_handle() const noexcept { return platform_handle_; }
    device::handle_t handle() const noexcept { return handle_; }

    platform_t platform() const noexcept;

protected:
    device_t(platform::handle_t platform_handle, device::handle_t handle) noexcept
    : platform_handle_(platform_handle), handle_(handle) { }

public:
    friend device_t device::wrap(platform::handle_t, device::handle_t) noexcept;

    // TODO: Should I delete instead of move the rvalue-reference methods?

    device_t& operator=(device_t const&) noexcept = default;
    device_t& operator=(device_t&&) noexcept = default;
    device_t(device_t const& other) noexcept = default;
    device_t(device_t&& other) noexcept = default;

    // methods based on getinfo
    device::type_t type() const;
    std::string name() const;
    vendor_t vendor() const;
    device::capabilities_t capabilities() const noexcept;
#ifdef CL_VERSION_1_2
    optional<device_t> parent() const;
    bool is_subdevice() const;

    /// Create a (logical) partition of this device into sub-devices
    dynarray<device_t> partition(device::partition::spec_t const& spec) const;
#endif // CL_VERSION_1_2

    /// @return the device's index among the devices  on its platform
    ///@{
    /// @p among_which the type of devices among which to compute the index
    device::index_t index_in_platform() const;
    device::index_t index_in_platform(device::type_t among_which) const;
    ///@}

    // get host clock value
    // get synchronized host & device clock values

    // create sub devices, possibly with extended properties

    // TODO: Maybe drop the whole context-device charade?
    std::pair<context_t, context::device_t>  in_self_context() const;

    context_t create_context() const; // ... with just this device in it

}; // class device_t

inline bool operator==(const device_t& lhs, const device_t& rhs) noexcept
{
    return lhs.platform_handle() == rhs.platform_handle() and lhs.handle() == rhs.handle();
}

inline bool operator!=(const device_t& lhs, const device_t& rhs) noexcept { return not (lhs == rhs); }

} // namespace opencl

#endif // OPENCL_WRAPPERS_DEVICE_HPP_
