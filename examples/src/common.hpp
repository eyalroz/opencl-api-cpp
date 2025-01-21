/**
 * @file
 *
 * @brief Common header for many/most OpenCL API wrappers example programs.
 */
#ifndef EXAMPLES_COMMON_HPP_
#define EXAMPLES_COMMON_HPP_


#include <string>
#include <iostream>
#ifdef __GNUC__
#include <cxxabi.h>
#endif

#include <opencl.hpp>

#include <fstream>
#include <cmath>
#include <cstdlib>
#include <iomanip>

#if __GNUC__
template <typename>
[[gnu::warning("type printed for your convenience")]]
bool your_type_was_() { return true; }

#define print_type_of(_x) your_type_was_<decltype(_x)>()
#endif

inline const char* ordinal_suffix(int n)
{
	static constexpr char suffixes [4][5] = {"th", "st", "nd", "rd"};
	auto ord = n % 100;
	if (ord / 10 == 1) { ord = 0; }
	ord = ord % 10;
	return suffixes[ord > 3 ? 0 : ord];
}

template <typename N = int>
std::string xth(N n) { return std::to_string(n) + ordinal_suffix(n); }

std::ostream& operator<<(std::ostream& os, opencl::nd_range::dimensions_t const& dims)
{
	os << '[';
	for (std::size_t i = 0; i < dims.size()-1; ++i) {
		os << dims[i] << ' ';
	}
	if (dims.size() > 0) {
		os << dims[dims.size() - 1];
	}
	os << ']';
	return os;
}

std::ostream& operator<<(std::ostream& os, opencl::kernel::launch_configuration_t const& lc)
{
	os << '[';
	os << "dimensions: workgroup: " << lc.dimensions.workgroup << "; ";
	os << "overall: " << lc.dimensions.overall << "; ";
	os << ']';
	return os;
}

std::ostream& operator<<(std::ostream& os, opencl::version_t const& version)
{
	return os << std::string{version};
}

[[noreturn]] inline bool die_(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

#define assert_(cond) \
{ \
	auto evaluation_result = (cond); \
	if (not evaluation_result) \
		die_("Assertion failed at line " + std::to_string(__LINE__) + ": " #cond); \
}

// Note: This will only work correctly for positive values
template <typename U1, typename U2>
typename std::common_type<U1,U2>::type div_rounding_up(U1 dividend, U2 divisor)
{
	return dividend / divisor + !!(dividend % divisor);
}

inline opencl::platform::index_t resolve_platform_index(int argc, char *argv[])
{
	opencl::platform::index_t platform_index = (argc == 1) ? 0 : std::strtol(argv[1], nullptr, 10);
	if (opencl::platform::count() <= platform_index) {
		std::cerr << "This system only has " << opencl::platform::count()
		  << " platforms, cannot use a platform with index " << platform_index << ".\n";
		exit(EXIT_FAILURE);
	}
	return platform_index;
}

#endif // EXAMPLES_COMMON_HPP_
