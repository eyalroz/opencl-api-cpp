/**
 * @file
 *
 * @brief An importation of an C++-17-like @ref opencl::variant class and related definitions.
 *
 */
#ifndef OPENCL_WRAPPERS_UTIL_VARIANT_HPP_
#define OPENCL_WRAPPERS_UTIL_VARIANT_HPP_

#if __cplusplus >= 201703L
#include <variant>
#include <any>
namespace opencl {
using std::variant;
using std::monostate;
} // namespace opencl
#else
#include "mpark_variant.hpp"
namespace opencl {
using mpark::variant;
using mpark::monostate;
using mpark::visit;
} // namespace opencl
#endif // __cplusplus >= 201703L

#endif // OPENCL_WRAPPERS_UTIL_VARIANT_HPP_
