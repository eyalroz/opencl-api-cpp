#include <opencl.hpp>
std::string get_default_device_name(opencl::platform_t const& platform)
{
	auto device = platform.default_device();
	return device.name();
}
