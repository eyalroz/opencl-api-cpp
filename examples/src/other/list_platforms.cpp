#include <iostream>
#define CL_TARGET_OPENCL_VERSION 300
#include <opencl.hpp>
#include "../enumerate.hpp"

int main() {
    auto platforms = opencl::platforms();
    for (auto index_and_platform : enumerate(platforms)) {
        auto const& platform = index_and_platform.item_;
        std::cout
            << "Platform " << index_and_platform.index << ":\n"
            << "Name:       " << platform.name() << '\n'
            << "Vendor:     " << platform.vendor() << '\n'
            << "Version:    " << platform.version_string() << '\n'
            << "Profile:    " << platform.opencl_profile() << '\n'
            << "Extensions: ";
        for (auto const& extension_name : platform.extensions()) {
            std::cout << extension_name << ' ';
        }
        std::cout << "\n";
    }
}