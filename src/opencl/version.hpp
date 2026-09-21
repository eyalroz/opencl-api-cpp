/**
 * @file
 *
 * @brief Definitions relating to representation and manipulation of versions - of
 * the The OpenCL standard itself and of versioned entities represented in it.
 */
#ifndef OPENCL_WRAPPERS_VERSION_HPP_
#define OPENCL_WRAPPERS_VERSION_HPP_

#include "types.hpp"

namespace opencl {

struct version_t {
    using component_type = version::component_t;

    enum { num_major_digits = 5, num_minor_digits = 4, num_patch_digits = 4 };

    component_type major;
    component_type minor;
    component_type patch;

    constexpr bool is_valid() const;
    component_type& operator[](size_t index);
    component_type const& operator[](size_t index) const;
    explicit operator std::string() const;
    // raw_version_t as_combined_number() const
};

constexpr bool operator==(version_t const& lhs, version_t const& rhs);
constexpr bool operator!=(version_t const& lhs, version_t const& rhs);
constexpr bool operator> (version_t const& lhs, version_t const& rhs);
constexpr bool operator< (version_t const& lhs, version_t const& rhs);
constexpr bool operator>=(version_t const& lhs, version_t const& rhs);
constexpr bool operator<=(version_t const& lhs, version_t const& rhs);

constexpr version_t make_version(
    version::component_t major,
    version::component_t minor = 0,
    version::component_t patch = 0);
// The version string must have the format MAJOR[.MINOR[.PATCH]],
// possibly with an "OpenCL " prefix (case is ignored); other formats
// will either be rejected or interpreted incorrectly.
version_t make_version(char const* version_string);
version_t make_version(std::string const& version_string);

template <typename I>
version_t make_version(span<I> version_elements_array);

// TODO: Just have a make_version for string_view's; but for that, we'd need
// a strntol / natoi function


#if CL_VERSION_3_0
struct name_and_version_t {
    std::string name;
    version_t version;
};
#endif // CL_VERSION_3_0

namespace version {

#if CL_VERSION_3_0
using named_t = name_and_version_t;
#endif
version_t decompose(raw_version_t raw_version);

} // namespace version

#if CL_VERSION_3_0
version::named_t make_named_version(cl_name_version const& raw_name_and_version);
#endif // CL_VERSION_3_0

namespace spir {

struct version_t {
    using component_type = version::component_t;

    component_type major;
    component_type minor;
};

constexpr bool operator==(version_t const& lhs, version_t const& rhs);
constexpr bool operator!=(version_t const& lhs, version_t const& rhs);
constexpr bool operator> (version_t const& lhs, version_t const& rhs);
constexpr bool operator< (version_t const& lhs, version_t const& rhs);
constexpr bool operator>=(version_t const& lhs, version_t const& rhs);
constexpr bool operator<=(version_t const& lhs, version_t const& rhs);

constexpr version_t make_version(
    version::component_t major,
    version::component_t minor = 0);

// The version string must have the format MAJOR[.MINOR[.PATCH]],
// possibly with an "OpenCL " prefix (case is ignored); other formats
// will either be rejected or interpreted incorrectly.
version_t make_version(char const* version_string);

} // namespace spir

} // namespace opencl

#endif //OPENCL_WRAPPERS_VERSION_HPP_
