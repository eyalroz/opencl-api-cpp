#include <opencl.hpp>

#include <iostream>

std::string get_default_device_name(opencl::platform_t const& platform);

int main() 
{
	auto count = opencl::platform::count();
	if (count > 0) {
		auto platform = opencl::platform::get(0);
		auto device_name = get_default_device_name(platform);
		std::cout << "Default device name for platform 0 is " << device_name << '\n';
	}
	std::cout << "SUCCESS\n";
}
