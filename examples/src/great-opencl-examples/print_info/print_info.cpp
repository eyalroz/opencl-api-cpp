#include <opencl.hpp>
#include <iostream>
#include <cstdlib>

[[noreturn]] void die(char const* message)
{
    std::cerr << message << std::endl;
    exit(EXIT_FAILURE);
}

int main()
{
    auto platforms = opencl::platforms();
    if (platforms.empty()) { die("No OpenCL platforms available!"); }
    auto device = platforms.front().default_device();
    auto capabilities = device.capabilities();
    auto name = device.name();
    auto vendor = device.vendor().name;
    auto version = device.capabilities().supported_opencl_version();
    auto workItems = capabilities.maximum_workgroup_dimensions();
    auto workGroups = capabilities.maximum_workgroup_size();
    auto computeUnits = capabilities.num_compute_units();
    auto globalMemory = capabilities.global_memory_size();
    auto localMemory = capabilities.local_memory_size();

    std::cout << "OpenCL Device Info:"
        << "\nName: " << name
        << "\nVendor: " << vendor
        << "\nVersion: " << std::string{version}
        << "\nMax size of work-items: (" << workItems[0] << "," << workItems[1] << "," << workItems[2] << ")"
        << "\nMax size of work-groups: " << workGroups
        << "\nNumber of compute units: " << computeUnits
        << "\nGlobal memory size (bytes): " << globalMemory
        << "\nLocal memory size per compute unit (bytes): " << localMemory/computeUnits
        << std::endl;
}