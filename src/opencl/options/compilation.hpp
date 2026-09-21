/**
 * @file
 *
 * @brief Definition of the @ref program::compilation::options_t class, and
 * declaration of its named constructor idiom and other related functions.
 */
#ifndef OPENCL_OPTIONS_COMPILATION_HPP_
#define OPENCL_OPTIONS_COMPILATION_HPP_

#include "common.hpp"
#include <unordered_set>
#include <unordered_map>

namespace opencl {
namespace program {

enum class warning_policy_t {
    suppress,
    emit,
    emit_and_continue = emit,
    treat_as_errors
};

// Note: The compilation target device(s) are _not_ part of the compilation options.
namespace compilation {
struct options_t : common_options_t {
public:
    std::string render() const { return detail::render(*this); }

    /// Preprocessor macros to have the compiler define, without specifying a particular value
    std::unordered_set<std::string> no_value_defines;

    /// Preprocessor macros to have the compiler define to specific values
    std::unordered_map<std::string,std::string> valued_defines;

    bool quote_definitions { false };

    /**
     * A sequence of directories to be searched for headers. These paths are searched _after_ the
     * list of headers given to nvrtcCreateProgram.
     *
     * @note The members here are `std::string`'s rather than `const char*` or `std::string_view`'s,
     * since this class is a value-type, and cannot rely someone else keeping these strings alive.
     *
     * @todo In C++17, consider making the elements `std::filesystem::path`'s.
     */
    std::vector<std::string> additional_include_paths;

    optional<bool> implicitly_convert_fp64_constants_to_fp32 { };

    optional<bool> denormals_may_be_treated_as_zero { };

#ifdef CL_VERSION_1_2
    optional<bool> precise_fp32_rounding_in_division_and_sqrt { };
#endif

    // Optimization-related options

    optional<bool> disable_all_optimizations { };

#if CL_VERSION <= CL_VERSION_1_1
    optional<bool> assume_strict_aliasing_rules { };
#endif

#if CL_VERSION_2_0
    optional<bool> grid_size_must_be_divisible_by_workgroup_size { };
#endif

#if CL_VERSION_2_1
    optional<bool> sub_workgroups_must_make_independent_progress { };
#endif

    // The following options control compiler behavior regarding floating-point arithmetic.

    optional<bool> enable_fused_multiply_add { };

    optional<bool> ignore_signedness_of_fp_zero { };

    // Should we support the following: it only aggregates other options
    // optional<bool> unsafe_math_optimizations { };

    optional<bool> assume_fp_values_are_finite { };

    // Aggregation of the previous two and defines __FAST_RELAXED_MATH__
    // optional<bool> fast_relaxed_math { };

    // Options regarding warning emission

    warning_policy_t warning_policy { warning_policy_t::emit };

    //  Options Controlling the OpenCL C Version

    optional<version_t> language_version { };

    bool use_opencl_cpp { false };

#if CL_VERSION_1_2
    optional<bool> store_kernel_argument_info_in_executable { };
#endif

    // Debugging-related options

#if CL_VERSION_2_0
    bool produce_debugging_info { false };
#endif


}; // class options_t

namespace options {

/// A trivial named constructor idiom, for the convenience of
/// not having to remember the type name and for consistence
/// with other create() functions
inline options_t create() { return options_t {}; }

}

namespace detail {

inline std::string as_option(version_t const& version)
{
#ifndef NDEBUG
    if ((version.major == 1 and version.minor == 0) or
        (version.major == 2 and version.minor != 0)) {
        throw std::invalid_argument("Unsupported OpenCL C language version - compilation should fail");
        }
#endif
    return "CL" + std::to_string(version.major) + "." + std::to_string(version.minor);
}

} // namespace detail
} // namespace compilation

namespace detail {

template<>
inline void render<compilation::options_t>(compilation::options_t const& opts, std::ostream& os)
{
    auto delim = common_options_t::delimiter();

    for (auto const& def : opts.no_value_defines) { os << "-D" << def << delim; }
    for (auto const& def : opts.valued_defines) {
        os << "-D" << def.first << '=';
        if (opts.quote_definitions) { os << '"'; }
        os << def.second;
        if (opts.quote_definitions) { os << '"'; }
        os << delim;
    }

    if (opts.implicitly_convert_fp64_constants_to_fp32.value_or(false)) { os << "-cl-single-precision-constant" << delim; }
#ifdef CL_VERSION_1_2
    if (opts.precise_fp32_rounding_in_division_and_sqrt.value_or(false)) { os << "-cl-fp32-correctly-rounded-divide-sqrt" << delim; }
#endif

    if (opts.disable_all_optimizations.value_or(false)) { os << "-cl-opt-disable " << delim; }
#if CL_VERSION <= CL_VERSION_1_1
    if (opts.assume_strict_aliasing_rules.value_or(false)) { os << "-cl-strict-aliasing" << delim; }
#endif
#if CL_VERSION_2_0
    if (opts.grid_size_must_be_divisible_by_workgroup_size.value_or(false)) { os << "-cl-uniform-work-group-size" << delim; }
#endif
#if CL_VERSION_2_1
    if (opts.sub_workgroups_must_make_independent_progress.value_or(false)) { os << "-cl-no-subgroup-ifp" << delim; }
#endif
    // The following options control compiler behavior regarding floating-point arithmetic.

    if (opts.enable_fused_multiply_add.value_or(false)) { os << "-cl-mad-enable" << delim; }

    switch (opts.warning_policy) {
    case warning_policy_t::suppress : os << "-w" << delim; break;
    case warning_policy_t::treat_as_errors : os << "-Werror" << delim; break;
    default: break; // do nothing
    }

    //  Options Controlling the programming language variant to expect

    if (opts.use_opencl_cpp) {
        if (opts.language_version) {
            throw std::invalid_argument(
                "OpenCL compilation options specify both the use of OpenCL C++ and an OpenCL C version");
        }
        os << "-cl-std=CLC++" << delim;
    }
    if (opts.language_version) {
        os << "-cl-std=" << compilation::detail::as_option(*opts.language_version) << ' ';
    }

    // Debugging-related options

#if CL_VERSION_1_2
    if (opts.store_kernel_argument_info_in_executable.value_or(false)) { os << "-cl-kernel-arg-info" << delim; }
#endif
    if (opts.produce_debugging_info) { os << "-g" << delim; }

    render<common_options_t>(opts, os);
}

} // namespace detail

} // namespace program
} // namespace opencl

#endif // OPENCL_OPTIONS_COMPILATION_HPP_

