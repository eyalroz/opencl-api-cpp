#ifndef OPENCL_WRAPPERS_IMPL_VERSION_HPP_
#define OPENCL_WRAPPERS_IMPL_VERSION_HPP_

#include "../version.hpp"

namespace opencl {

constexpr bool version_t::is_valid() const
{
    return
        (major >= 0 and major < (1 << num_major_digits)) and
        (minor >= 0 and minor < (1 << num_minor_digits)) and
        (patch >= 0 and patch < (1 << num_patch_digits));
}

inline version_t::component_type& version_t::operator[](size_t index)
{
    switch (index) {
    case 0: return major;
    case 1: return minor;
    case 2: return patch;
    default: throw std::invalid_argument("Invalid version component index " + std::to_string(index));
    }
}

inline version_t::component_type const& version_t::operator[](size_t index) const
{
    return (*const_cast<version_t*>(this))[index];
}

inline version_t::operator std::string() const
{
    // TODO: Make this less inefficient, perhaps at the price of some verbosity
    return std::to_string(major) + '.' + std::to_string(minor) + '.' + std::to_string(patch);
}

/*raw_version_t compatibility_t::as_combined_number() const
{
#ifndef NDEBUG
    if (not is_valid()) {
      throw std::logic_error("Attempt to convert an invalid OpenCL version into a single numeric value");
    }
#endif
    return
          major << (num_patch_digits + num_minor_digits)
        | minor << num_patch_digits
        | patch;
}*/

constexpr bool operator==(version_t const& lhs, version_t const& rhs)
{
    return lhs.major == rhs.major and lhs.minor == rhs.minor and lhs.patch == rhs.patch;
}
constexpr bool operator!=(version_t const& lhs, version_t const& rhs) { return not (lhs == rhs); }
constexpr bool operator> (version_t const& lhs, version_t const& rhs)
{
    return lhs.major > rhs.major
        or (lhs.major == rhs.major and lhs.minor > rhs.minor)
        or (lhs.major == rhs.major and lhs.minor == rhs.minor and lhs.patch > rhs.patch);
}
constexpr bool operator< (version_t const& lhs, version_t const& rhs) { return rhs > lhs; }
constexpr bool operator>=(version_t const& lhs, version_t const& rhs) { return (lhs > rhs) or (lhs == rhs); }
constexpr bool operator<=(version_t const& lhs, version_t const& rhs) { return (lhs < rhs) or (lhs == rhs); }


constexpr version_t make_version(version::component_t major, version::component_t minor, version::component_t patch)
{
    return {major, minor, patch};
}

inline version_t make_version(char const* version_string)
{
    // Strip possible prefix
    {
        // Computing the length is actually redundant, but - it helps us avoid
        // a compiler warning about advancing version_string by too much - when
        // the compiler knows the lengths at compile-time, but does not do what
        // `strncasecmp()` does exactly.
        auto len = std::strlen(version_string);
        static constexpr auto prefix = "OpenCL ";
        auto length_of_prefix = std::strlen(prefix);
        if (len >= length_of_prefix and strncasecmp(version_string, prefix, length_of_prefix) == 0) {
            version_string += length_of_prefix;
        }
    }
    version_t result { 0, 0, 0 };
    if (not isdigit(*version_string)) {
        throw std::invalid_argument("Unsupported OpenCL version string");
    }
    enum { decimal_basis = 10 };
    result.major = std::strtol(version_string, nullptr, decimal_basis);
    auto first_period = std::strchr(version_string, '.');
    if (not first_period) { return result; }
    if (not isdigit(*(first_period + 1))) {
        throw std::invalid_argument("Unsupported OpenCL version string");
    }
    result.minor = std::strtol(first_period + 1, nullptr, decimal_basis);
    auto second_period = std::strchr(first_period + 1, '.');
    if (not second_period) { return result; }
    if (not isdigit(*(second_period + 1))) {
        throw std::invalid_argument("Unsupported OpenCL version string");
    }
    result.patch = std::strtol(second_period + 1, nullptr, decimal_basis);
    return result;
}

inline version_t make_version(std::string const& version_string)
{
    return make_version(version_string.c_str());
}

template <typename I>
version_t make_version(span<I> version_elements_array)
{
    if (version_elements_array.size() != 3) {
        throw std::logic_error("Device OpenCL version query returned an unexpected number of elements: "
            + std::to_string(version_elements_array.size()));
    }
    version_t result;
    result.major = version_elements_array[0];
    result.minor = version_elements_array[1];
    result.patch = version_elements_array[2];
    return result;
}

namespace version {

inline version_t decompose(raw_version_t raw_version)
{
    using version::component_t;
    auto major = static_cast<component_t>((raw_version >> 22) & 0x3FF);
    auto minor = static_cast<component_t>((raw_version >> 12) & 0x3FF);
    auto patch = static_cast<component_t>(raw_version & 0xFFF);
    return { major, minor, patch };
}

} // namespace version

namespace spir {

constexpr version_t make_version(version::component_t major, version::component_t minor) { return {major, minor}; }

constexpr bool operator==(version_t const& lhs, version_t const& rhs)
{
    return lhs.major == rhs.major and lhs.minor == rhs.minor;
}
constexpr bool operator!=(version_t const& lhs, version_t const& rhs) { return not (lhs == rhs); }
constexpr bool operator> (version_t const& lhs, version_t const& rhs)
{
    return lhs.major > rhs.major
        or (lhs.major == rhs.major and lhs.minor > rhs.minor)
        or (lhs.major == rhs.major and lhs.minor == rhs.minor);
}
constexpr bool operator< (version_t const& lhs, version_t const& rhs) { return rhs > lhs; }
constexpr bool operator>=(version_t const& lhs, version_t const& rhs) { return (lhs > rhs) or (lhs == rhs); }
constexpr bool operator<=(version_t const& lhs, version_t const& rhs) { return (lhs < rhs) or (lhs == rhs); }

inline version_t make_version(char const* version_string)
{
    enum { decimal_basis = 10 };
    {
        static constexpr auto prefix = "SPIR ";
        auto length_of_prefix = std::strlen(prefix);
        if (strncasecmp(version_string, prefix, length_of_prefix) == 0) {
            version_string += length_of_prefix;
        }
    }
    version_t result { 0, 0 };
    if (not std::isdigit(*version_string)) {
        throw std::invalid_argument("Unsupported SPIR version string");
    }
    result.major = std::strtol(version_string, nullptr, decimal_basis);
    auto first_period = std::strchr(version_string, '.');
    if (not first_period) { return result; }
    if (not isdigit(*(first_period + 1))) {
        throw std::invalid_argument("Unsupported SPIR version string");
    }
    result.minor = std::strtol(first_period + 1, nullptr, decimal_basis);
    return result;
}

} // namespace spir

#if CL_VERSION_3_0
inline version::named_t make_named_version(cl_name_version const& raw_name_and_version)
{
    return { raw_name_and_version.name, version::decompose(raw_name_and_version.version) };
}
#endif

template <template <typename> class InContainer, typename NamedVersion, template <typename> class OutContainer>
OutContainer<std::string> strip_versions(InContainer<version::named_t> const& names_and_versions)
{
    static_assert(
        std::is_same<NamedVersion, cl_name_version>::value or
        std::is_same<NamedVersion, version::named_t>::value, "Unexpected name-and-version element type");
    OutContainer<std::string> result;
    for (auto const& nv : names_and_versions) { result.emplace_back(nv.name); }
    return result;
}

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_VERSION_HPP_
