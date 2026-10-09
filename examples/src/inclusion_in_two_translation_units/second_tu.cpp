#include <opencl.hpp>
std::string get_default_device_name(opencl::platform_t const& platform)
{
	auto device = platform.default_device();
	return device.name();
}

/**
 * Two nice functions
 *
 *@param[in] x  The x parameter.
 */
///@{
///@param[in] y The y parameter.
int foo(int x, int y);

int bar(int x, int y);
///@}