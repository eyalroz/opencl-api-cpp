/**
 * @file
 *
 * @brief Miscellaneous utility code for the OpenCL C++ wrappers - which is not
 * specific to OpenCL.
 */
#ifndef OPENCL_WRAPPERS_UTIL_MISCELLANY_HPP
#define OPENCL_WRAPPERS_UTIL_MISCELLANY_HPP

#include "string_view.hpp"
#include <string>
#include <vector>

#ifndef STRINGIFY
#define STRINGIFY(_q) #_q
#endif

namespace opencl {
namespace detail {

template <typename I, typename I2 = I>
constexpr I round_up(I x, I2 modulus) noexcept
{
    static_assert(std::is_integral<I>::value and std::is_integral<I2>::value,
        "Attempt to round-up using non-integral types");
    static_assert(std::is_signed<I>::value or std::is_unsigned<I2>::value,
        "Attempt to implicitly introduce signedness; make an explicit cast");
    return (x % modulus == 0) ? x : x + (modulus - x%modulus);
}

constexpr bool lxor(bool lhs, bool rhs) noexcept { return (lhs and not rhs) or (rhs and not lhs); }

/// return the sequence of tokens within a delimited string - each token a view into the
/// original rather a copy.
inline CONSTEXPR_CPP20 std::vector<string_view> split(string_view str, char separator)
{
    std::vector<string_view> result{};
    struct { std::string::size_type start, end; } token;
    token.start = 0;
    do {
        token.end = str.find(separator, token.start);
        if (token.end == std::string::npos) {
            result.emplace_back(str.substr(token.start));
            break;
        }
        result.emplace_back(str.substr(token.start, token.end - token.start)); // No need for a move - it's a string view; a string will be constructed here
        token.start = token.end + 1; // past the separator
    } while (true);
    return result;
}

#ifdef CL_VERSION_1_2

inline dynarray<char> join(span<string_view const> strings, char separator)
{
    auto sum_of_lengths = std::accumulate(strings.begin(), strings.end(), size_t{0},
        [](size_t sol, string_view sv) { return sol + sv.length(); });
    auto result = make_dynarray<char>(sum_of_lengths + strings.size());
    size_t pos = 0;
    for (auto sv : strings) {
        if (pos != 0) {
            result[pos++] = separator;
        }
        sv.copy(&result[pos], sv.length());
        pos += sv.length();
    }
    return result;
}
#endif

} // namespace detail
} // namespace opencl

#endif //OPENCL_WRAPPERS_UTIL_MISCELLANY_HPP
