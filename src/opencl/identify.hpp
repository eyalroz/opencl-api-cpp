/**
 * @file
 *
 * @brief Declaration of functions producing strings for identifying
 * various OpenCL-related entities, for use (mostly) in error messages.
 *
 * @note None of the `identify()` functions are `noexcept`, because they create,
 * and return `std::string`'s; otherwise; but otherwise, they _are_ `noexcept`,
 * in the sense of not calling any code that is may throw, nor throwing
 * themselves. If there was some popular string class which involved dynamic
 * on-stack allocation, or perhaps was compositional and could reside on the
 * stack that way - we could wrap our exceptions around it instead of `std::string`
 * and be completely noexcept.
 *
 * @todo Perhaps we should templatize the identify functions? The declarations are
 * rather redundant.
 */
#ifndef OPENCL_WRAPPERS_IDENTIFY_HPP_
#define OPENCL_WRAPPERS_IDENTIFY_HPP_

#include "types.hpp"

#ifdef NDEBUG
#define OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT false
#else
#define OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT true
#endif

namespace opencl {

namespace detail {
std::string identify(platform_t const&);
std::string identify(device_t const&);
std::string identify(context_t const&);
std::string identify(context::device_t const&);
std::string identify(memory::object_t const&);
std::string identify(buffer_t const&);
std::string identify(pipe_t const&);
std::string identify(kernel_t const&);
std::string identify(program_t const&);
std::string identify(program::source_t const&);
std::string identify(program::with_intermediate_language_t const&);
std::string identify(program::with_binaries_t const&);
std::string identify(queue_t const&);
std::string identify(event_t const&);
std::string identify(image::sampler_t const&);
std::string identify(info::detail::kernel_device_handle_pair_t const&);
std::string identify(info::detail::program_device_handle_pair_t const& program_and_device);
std::string identify(memory::local::allowance_t const& local_mem_allowance);
std::string identify(info::detail::kernel_param_index_pair_t const& kernel_and_param_index);
} // namespace detail

// Handle identifier declarations
namespace detail {

enum {
    do_not_obtain_details = false,
    do_obtain_details = true
};

inline std::string identify(platform::handle_t handle, bool obtain_details = OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT);
inline std::string identify(device::handle_t handle, bool obtain_details = OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT);
inline std::string identify(context::handle_t handle, bool obtain_details = OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT);
inline std::string identify(queue::handle_t handle, bool obtain_details = OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT);
inline std::string identify(memory::handle_t handle, bool obtain_details = OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT);
inline std::string identify(event::handle_t handle, bool obtain_details = OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT);
inline std::string identify(program::handle_t handle, bool obtain_details = OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT);
inline std::string identify(kernel::handle_t handle, bool obtain_details = OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT);
inline std::string identify(image::sampler::handle_t handle, bool obtain_details = OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT);

} // namespace detail

namespace context {
namespace device {
namespace detail {
std::string identify(device_t const&);
} // namespace detail
} // namespace device
} // namespace context


namespace memory {
namespace detail {
std::string identify(region_t region);
} // namespace detail

namespace shared_virtual {
namespace detail {
template <typename Region>
std::string identify_region(Region region);
} // namespace detail
} // namespace shared_virtual

} // namespace memory

namespace nd_range {
namespace detail {
std::string identify(nd_range::workgroup_dimensions_t const& dimensions);
} // namespace detail
} // namespace nd_range

} // namespace opencl

#undef OCLW_IDENTIFY_OBTAIN_DETAILS_DEFAULT

#endif // OPENCL_WRAPPERS_IDENTIFY_HPP_
