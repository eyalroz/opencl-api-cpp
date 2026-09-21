/**
 * @file
 *
 * @brief An importation of an C++-17-like @ref opencl::optional class and related definitions.
 *
 * @note When compiling with C++17 or later, the actual @ref std::optional class is used.
 */
#ifndef OPENCL_WRAPPERS_UTIL_OPTIONAL_HPP_
#define OPENCL_WRAPPERS_UTIL_OPTIONAL_HPP_

#if __cplusplus >= 201703L
#include <optional>
#include <any>
namespace opencl {
using std::optional;
using std::nullopt_t;
using std::nullopt;
} // namespace opencl
#else
#include "optional_lite.hpp"
namespace opencl {
using nonstd::optional;
using nonstd::nullopt_t;
using nonstd::nullopt;
} // namespace opencl
#endif // __cplusplus >= 201703L

#endif // OPENCL_WRAPPERS_UTIL_OPTIONAL_HPP_
