#ifndef OPENCL_OPTIONS_LINK_HPP_
#define OPENCL_OPTIONS_LINK_HPP_

#include "common.hpp"

namespace opencl {
namespace program {

namespace link {
struct options_t : common_options_t {
public:
    std::string render() const { return detail::render(*this); }
    // SPIR Compilation Options

#ifdef CL_KHR_SPIR_EXTENSION_VERSION
    optional<spir::version_t> inputs_spir_version;
    bool input_binaries_are_spir() const { return inputs_spir_version.has_value(); }
#endif
}; // class link::options_t

namespace detail {
inline std::string as_option(spir::version_t const& spir_version)
{
    return std::to_string(spir_version.major) + "." + std::to_string(spir_version.minor);
}
} // namespace detail


namespace options {

/// A trivial named constructor idiom, for the convenience of
/// not having to remember the type name and for consistence
/// with other create() functions
inline options_t create() { return options_t {}; }

}

} // namespace link

namespace detail {

template<>
inline void render<link::options_t>(link::options_t const& opts, std::ostream& os)
{
    auto delim = common_options_t::delimiter();

#if CL_KHR_SPIR_EXTENSION_VERSION
    if (opts.input_binaries_are_spir()) {
        os << "-x spir" << delim;
        os << "-spir-std=" << link::detail::as_option(*opts.inputs_spir_version) << delim;
    }
#endif

    render<common_options_t>(opts, os);
}

} // namespace detail

} // namespace program
} // namespace opencl

#endif // OPENCL_OPTIONS_LINK_HPP_
