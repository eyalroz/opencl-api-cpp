/**
 * @file
 *
 * @brief Functions producing strings for identifying various
 * OpenCL-related entities, for use in error messages.
 *
 */
#ifndef OPENCL_WRAPPERS_IMPL_IDENTIFY_HPP_
#define OPENCL_WRAPPERS_IMPL_IDENTIFY_HPP_

#include "../identify.hpp"
#include "device.hpp"
#include "platform.hpp"
#include "queue.hpp"
#include "context.hpp"
#include "kernel.hpp"
#include "program.hpp"
#include "info.hpp"

namespace opencl {
namespace device {
namespace detail {

// TODO: Maybe support also identifying by index within the platform?
// and given the index, drop the pointer?

inline std::string identify(handle_t handle, std::string const& name)
{
	return "device '" + name + "' at " + opencl::detail::ptr_as_hex(handle);
}

inline std::string identify(platform::handle_t platform_handle, handle_t handle, std::string const& name)
{
	return identify(handle, name) + " on " + opencl::detail::identify(platform_handle);
}

} // namespace detail
} // namespace device

namespace context {
namespace detail {
inline std::string identify(platform::handle_t platform_handle, handle_t context_handle)
{
	return "context " + opencl::detail::ptr_as_hex(context_handle) + " in " + opencl::detail::identify(platform_handle);
}

} // namespace detail
} // namespace context

namespace memory {
namespace detail {

inline std::string identify(const_region_t region)
{
	return std::string("memory region at ") + opencl::detail::ptr_as_hex(region.data())
		+ " of size " + std::to_string(region.size());
}

inline std::string identify(region_t region)
{
	return identify(static_cast<const_region_t>(region));
}

inline std::string identify(subregion_spec_t subregion)
{
	return std::to_string(subregion.size) + " bytes at offset " + std::to_string(subregion.origin);
}

} // namespace detail

namespace shared_virtual {
namespace detail {

template <typename Region>
std::string identify_region(Region region)
{
	using tp_ = opencl::detail::remove_cvref_t<Region>;
	static constexpr bool is_region = std::is_base_of<memory::region_t, tp_>::value or std::is_base_of<memory::const_region_t, tp_>::value;
	static constexpr bool is_svm = std::is_base_of<region_t, tp_>::value or std::is_base_of<const_region_t, tp_>::value;
	static_assert(is_region, "Only use this function on memory regions");
	return (is_svm ? "SVM " : "") + memory::detail::identify(region);
}

} // namespace detail
} // namespace shared_virtual

} // namespace memory

namespace kernel {
namespace detail {

inline std::string identify(kernel::handle_t handle, const std::string& name)
{
	return "kernel '" + name + "' at " + opencl::detail::ptr_as_hex(handle);
}

inline std::string identify(
	platform::handle_t platform_handle,
	context::handle_t context_handle,
	program::handle_t program_handle,
	kernel::handle_t handle,
	const std::string& name)
{
	auto program = program::detail::wrap(platform_handle, context_handle, program_handle, is_not_owning);
	return identify(handle, name) + " in " + opencl::detail::identify(program);
}

} // namespace detail
} // namespace kernel

namespace nd_range {
namespace detail {
inline std::string identify(nd_range::workgroup_dimensions_t const& dimensions)
{
	std::ostringstream oss;
	oss << '(';
	bool first = true;
	for (auto dim : dimensions) {
		if (first) { first = false;} else { oss << ','; }
		oss << dim;
	}
	oss << ')';
	return oss.str();
}
} // namespace detail
} // namespace nd_range

// TODO: identify() for user_event_t and queue::event_t

namespace detail {

inline std::string identify(platform_t const & platform)
{
	auto platform_name = info::get_string_nothrow<CL_PLATFORM_NAME>(platform.handle());
	// Note: We can't directly get the index
	auto maybe_index = platform::detail::get_index_of_nothrow(platform.handle());
	return "platform "
		+ (maybe_index ? std::to_string(*maybe_index) : "")
		+ (platform_name ? " (" : "")
		+ (platform_name ? *platform_name : "")
		+ (platform_name ? ")" : "")
		+ " at " + ptr_as_hex(("C"));
}

inline std::string identify(context_t const & context)
{
	return identify(context.handle(), do_not_obtain_details) + " on " + identify(context.platform_handle());
}

inline std::string identify(device_t const& device)
{
#ifndef NDEBUG
	auto name = info::get_string_nothrow<CL_DEVICE_NAME>(device.handle());
	if (name) {
		return device::detail::identify(device.platform_handle(), device.handle(), *name);
	}
#endif
	return "device at " + ptr_as_hex(device.handle()) + " on " + identify(device.platform());
}

inline std::string identify(context::device_t const& device)
{
#ifndef NDEBUG
	auto name = info::get_string_nothrow<CL_DEVICE_NAME>(device.handle());
#endif
	return identify(device.handle(), do_not_obtain_details)
#ifndef NDEBUG
		+ (name ? " (" + *name + ')' : "")
#endif
		+ " in " + identify(device.context());
}

inline std::string identify(buffer_t const& buffer)
{
	return std::string("buffer ") + ptr_as_hex(buffer.handle());
}

inline std::string identify(memory::object_t const& object)
{
	return "memory object at " + ptr_as_hex(object.handle()) + " in " + identify(object.context());
}

inline std::string identify(pipe_t const& pipe)
{
	return identify(pipe.handle()) + " in " + identify(pipe.context_handle()) + " in " + identify(pipe.platform_handle());
}

inline std::string identify(kernel_t const& kernel)
{
#ifndef NDEBUG
	try {
		return kernel::detail::identify(kernel.handle(), kernel.name());
	} catch(std::exception const&) { }
#endif
	// TODO: Avoid this duplication; take a boolean in the handle identifier function
	return "kernel at " + ptr_as_hex(kernel.handle());
}

// Used when obtaining device-specific compiled/built kernel information, see info.hpp
inline std::string identify(info::detail::kernel_device_handle_pair_t const& kernel_and_device)
{
	return identify(kernel_and_device.kernel_) + ", on " + identify(kernel_and_device.device_);
}

// Used when obtaining device-specific program build information, see info.hpp
inline std::string identify(info::detail::program_device_handle_pair_t const& program_and_device)
{
	return "build information of " + identify(program_and_device.program_) + ", targeting " + identify(program_and_device.device_);
}

// Used when obtaining device-specific compiled/built kernel information, see info.hpp
inline std::string identify(info::detail::kernel_param_index_pair_t const& kernel_and_param_index)
{
	return "parameter " + std::to_string(kernel_and_param_index.index_) + " of " + identify(kernel_and_param_index.kernel_);
}


inline std::string identify(memory::local::allowance_t const& local_mem_allowance)
{
	return "local memory allowance of " + std::to_string(local_mem_allowance.size) + " bytes";
}

inline std::string identify(program_t const& program)
{
	return identify(program.handle(), do_not_obtain_details) + " in " + identify(program.context());
}

inline std::string identify(program::source_t const& program)
{
	return identify(program.handle(), do_not_obtain_details) + " with sources, in " + identify(program.context());
}

inline std::string identify(program::with_binaries_t const& program)
{
	return identify(program.handle(), do_not_obtain_details)  + " with binaries, in " + identify(program.context());
}

inline std::string identify(program::with_kernels_t const& program)
{
	return identify(program.handle(), do_not_obtain_details)  + " with kernels,  in " + identify(program.context());
}

inline std::string identify(queue_t const& queue)
{
	return identify(queue.handle(), do_not_obtain_details)  + " on " + identify(queue.device());
}

inline std::string identify(event_t const& event)
{
	// TODO: Differentiate between queue event and user event here? Add an obtain_details flag parameter?
	return identify(event.handle(), do_not_obtain_details)  + " in " + identify(event.context());
}

inline std::string identify(image::sampler_t const& sampler)
{
	return identify(sampler.handle(), do_not_obtain_details) + " in " + identify(sampler.context());
}

} // namespace detail

// Handle identifier implementations
namespace detail {

inline std::string identify(platform::handle_t handle, bool obtain_details)
{
	return obtain_details ? identify(platform::wrap(handle, {})) : "platform " + ptr_as_hex(handle);
}

inline std::string identify(device::handle_t handle, bool obtain_details)
{
	auto result = "device at " + ptr_as_hex(handle);
	if (not obtain_details) { return result; }
	auto platform_handle = info::get_scalar_nothrow<CL_DEVICE_PLATFORM>(handle);
	if (not platform_handle) { return result; }
	auto device = device::wrap(*platform_handle, handle);
	return identify(device);
}

inline std::string identify(context::handle_t handle, bool obtain_details)
{
	auto result = "context at " + ptr_as_hex(handle);
	if (not obtain_details) { return result; }
	auto platform_handle = context::detail::get_platform_handle_nothrow(handle);
	if (not platform_handle) { return result; }
	return context::detail::identify(*platform_handle, handle);
}

inline std::string identify(kernel::handle_t handle, bool obtain_details)
{
	auto result = "kernel at " + ptr_as_hex(handle);
	if (not obtain_details) { return result; }
	auto program_handle  = info::get_scalar_nothrow<CL_KERNEL_PROGRAM>(handle);
	if (not program_handle) { return result;}
	auto context_handle  = info::get_scalar_nothrow<CL_PROGRAM_CONTEXT>(*program_handle);
	if (not context_handle) { return result; }
	auto platform_handle = context::detail::get_platform_handle_nothrow(*context_handle);
	auto name = info::get_string_nothrow<CL_KERNEL_FUNCTION_NAME>(handle).value_or("");
	return kernel::detail::identify(*platform_handle, *context_handle, *program_handle, handle, name);
}

inline std::string identify(queue::handle_t handle, bool obtain_details)
{
	auto result = "queue at " + ptr_as_hex(handle);
	if (not obtain_details) { return result; }
	auto context_handle  = info::get_scalar_nothrow<CL_QUEUE_CONTEXT>(handle);
	if (not context_handle) { return result; }
	auto device_handle  = info::get_scalar_nothrow<CL_QUEUE_DEVICE>(handle);
	if (not device_handle) { return result; }
	auto platform_handle = context::detail::get_platform_handle_nothrow(*context_handle);
	if (not platform_handle) { return result; }
	auto dev = device::wrap(*platform_handle, *device_handle);
	auto context = context::wrap(*platform_handle, {}, *context_handle, is_not_owning);
	return result + " on " + identify(dev) + " within " + identify(context);
}

inline std::string identify(memory::handle_t handle, bool obtain_details)
{
	auto result = std::string("memory object ") + ptr_as_hex(handle);
	if (not obtain_details) { return result; }
	auto context_handle = info::get_scalar_nothrow<CL_MEM_CONTEXT>(handle);
	if (not context_handle) { return result; }
	auto platform_handle = context::detail::get_platform_handle_nothrow(*context_handle);
	if (not platform_handle) { return result; }
	auto memory_object = memory::wrap(*platform_handle, *context_handle, handle, is_not_owning);
	return identify(memory_object);
}

inline std::string identify(event::handle_t handle, bool obtain_details)
{
	auto result = "event at " + ptr_as_hex(handle);
	if (not obtain_details) { return result; }
	auto context_handle  = info::get_scalar_nothrow<CL_EVENT_CONTEXT>(handle);
	if (not context_handle) { return result; }
	auto platform_handle = context::detail::get_platform_handle_nothrow(*context_handle);
	if (not platform_handle) { return result; }
	auto queue_handle = info::get_scalar_nothrow<CL_EVENT_COMMAND_QUEUE>(handle);
	if (not queue_handle) {
		auto user_event = event::wrap_user_event(*platform_handle, *context_handle, handle, is_not_owning);
		return identify(user_event);
	}
	auto device_handle  = info::get_scalar_nothrow<CL_QUEUE_DEVICE>(*queue_handle);
	if (not device_handle) { return result; }
	auto queue_event = queue::event::wrap(*platform_handle, *context_handle, *device_handle, *queue_handle, handle, is_not_owning);
	return identify(queue_event);
}

inline std::string identify(program::handle_t handle, bool obtain_details)
{
	auto result = "program at " + ptr_as_hex(handle);
	if (not obtain_details) { return result; }
	auto context_handle  = info::get_scalar_nothrow<CL_PROGRAM_CONTEXT>(handle);
	if (not context_handle) { return result; }
	auto platform_handle = context::detail::get_platform_handle_nothrow(*context_handle);
	if (not platform_handle) { return result; }
	auto program = program::detail::wrap(*platform_handle, *context_handle, handle, is_not_owning);
	return identify(program);
}

inline std::string identify(image::sampler::handle_t handle, bool obtain_details)
{
	auto result = "image sampler at " + ptr_as_hex(handle);
	if (not obtain_details) { return result; }
	auto context_handle  = info::get_scalar_nothrow<CL_SAMPLER_CONTEXT>(handle);
	if (not context_handle) { return result; }
	auto platform_handle = context::detail::get_platform_handle_nothrow(*context_handle);
	if (not platform_handle) { return result; }
	auto sampler = image::sampler::wrap(*platform_handle, *context_handle, handle, is_not_owning);
	return identify(sampler);
}

} // namespace detail

} // namespace opencl


#endif // OPENCL_WRAPPERS_IMPL_IDENTIFY_HPP_
