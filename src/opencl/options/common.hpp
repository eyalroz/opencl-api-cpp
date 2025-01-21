#ifndef OPENCL_OPTIONS_COMMON_HPP_
#define OPENCL_OPTIONS_COMMON_HPP_

#include "../types.hpp"

#include <sstream>
#include <vector>

namespace opencl {
namespace program {

struct common_options_t;

namespace detail {

template <typename Options>
std::string render(const Options& opts);

template<typename Options>
void render(Options const& opts, std::ostream& os);

} // namespace detail

struct common_options_t {
    std::string render() const { return detail::render(*this); }
    static constexpr char delimiter() noexcept { return ' '; }

    std::vector<std::string> additional;

    optional<bool> denormals_may_be_treated_as_zero { };
#if CL_VERSION_2_1
    optional<bool> sub_workgroups_must_make_independent_progress { };
#endif

    // The following options control compiler behavior regarding floating-point arithmetic.
    optional<bool> enable_fused_multiply_add { };
    optional<bool> ignore_signedness_of_fp_zero { };

    // Aggregation of of denormals-are-zero, enable-fma and ignore-sign-of-zero - and possibly other optimizations
    optional<bool> unsafe_math_optimizations { };

    optional<bool> assume_fp_values_are_finite { };

    // Aggregation of unsafe-math-opts and, assume-fp-finite and also defines __FAST_RELAXED_MATH__
    // optional<bool> fast_relaxed_math { };
}; // class common_options_t

namespace detail {

template<>
inline void render<common_options_t>(common_options_t const& opts, std::ostream& os)
{
    auto delim = common_options_t::delimiter();

    if (opts.denormals_may_be_treated_as_zero.value_or(false)) { os << "-cl-denorms-are-zero" << delim; }
    if (opts.enable_fused_multiply_add.value_or(false)) { os << "-cl-mad-enable" << delim; }
    if (opts.ignore_signedness_of_fp_zero.value_or(false)) { os << "-cl-no-signed-zeros" << delim; }
    if (opts.unsafe_math_optimizations.value_or(false)) { os << "-cl-unsafe-math-optimizations" << delim; }
    if (opts.assume_fp_values_are_finite.value_or(false)) { os << "-cl-finite-math-only"; }

    // Should we support the following?: It only aggregates mad-enable, no-signed-zeros, denorms-are-zero
    // and finite-math-only
    // if (opts.fast_relaxed_math.value_or(false)) { os << "-cl-fast-relaxed-math " << delim; }

#if CL_VERSION_2_1
    if (opts.sub_workgroups_must_make_independent_progress.value_or(false)) { os << "-cl-no-subgroup-ifp" << delim; }
#endif

    for (auto const& opt : opts.additional) { os << opt << delim; }
}


template <typename Options>
std::string render(const Options& opts)
{
    std::ostringstream oss;
    detail::render(opts, oss);
    if (oss.tellp() > 0) {
        // Remove the last, excessive, delimiter
        oss.seekp(-1,oss.cur);
    }
    return oss.str();
}

} // namespace detail
} // namespace program
} // namespace opencl

#endif // OPENCL_OPTIONS_COMMON_HPP_
