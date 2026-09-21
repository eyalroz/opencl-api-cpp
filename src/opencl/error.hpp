/**
 * @file
 *
 * @brief Facilities for exception-based handling of OpenCL errors, including
 * the definition of @ref opencl::runtime_error, wrapping `std::runtime_error`.
 *
 */
#ifndef OPENCL_WRAPPERS_ERROR_HPP_
#define OPENCL_WRAPPERS_ERROR_HPP_

#include "types.hpp"
#include "util/string_view.hpp"

#include <string>
#include <stdexcept>

#define oclw_throw(_exception_type, ...) throw _exception_type(opencl::detail::format(__VA_ARGS__))

namespace opencl {
namespace detail {

inline char const * ordinal_suffix(int n)
{
    static char constexpr suffixes [4][5] = {"th", "st", "nd", "rd"};
    auto ord = n % 100;
    if (ord / 10 == 1) { ord = 0; }
    ord = ord % 10;
    return suffixes[ord > 3 ? 0 : ord];
}

template <typename N = int>
std::string xth(N n) { return std::to_string(n) + ordinal_suffix(n); }

} // namespace detail

namespace status {

/**
 * Aliases for OpenCL status codes
 */
enum named_t {
	success              = CL_SUCCESS,
	invalid_value        = CL_INVALID_VALUE
};

///@cond
constexpr bool operator==(const status_t& lhs, const named_t& rhs) noexcept { return lhs == static_cast<status_t>(rhs); }
constexpr bool operator!=(const status_t& lhs, const named_t& rhs) noexcept { return lhs != static_cast<status_t>(rhs); }
constexpr bool operator==(const named_t& lhs, const status_t& rhs) noexcept { return static_cast<status_t>(lhs) == rhs; }
constexpr bool operator!=(const named_t& lhs, const status_t& rhs) noexcept { return static_cast<status_t>(lhs) != rhs; }
///@endcond

} // namespace status

/// Determine whether the API call returning the specified status had succeeded
///@{
constexpr bool is_success(status_t status)  { return status == static_cast<status_t>(status::success); }
///@}

/// @brief Determine whether the API call returning the specified status had failed
///@{
constexpr bool is_failure(status_t status)  { return not is_success(status); }
///@}

string_view description_of(status_t status);

namespace detail {

char const* description_of_cstr(status_t status);

template <bool UpperCase = false, typename I>
std::string as_hex(I x, unsigned num_hex_digits = 2 * sizeof(I));

// TODO: Perhaps find a way to avoid the extra function, so that as_hex() can
// be called for pointer types as well? Would be easier with boost's uint<T>...
template <typename T, bool UpperCase = false>
std::string ptr_as_hex(T const * ptr);

} // namespace detail

/**
 * A (base?) class for exceptions raised by OpenCL API code; these errors are thrown by
 * essentially all OpenCL API wrappers upon failure.
 *
 * An OpenCL API error can be constructed with either just an error code (= status
 * code), or a code plus an additional, arbitrary, message.
 *
 * @note The exception uses std::string's rather than string_view's, so as to avoid
 * scope/lifetime issues.
 */
class runtime_error final : public std::runtime_error {
public:
	///@cond
	runtime_error(status_t error_code) :
		std::runtime_error(detail::description_of_cstr(error_code)), code_(error_code), api_function_name_(nullptr)
	{ }
	// I wonder if I should do this the other way around
	runtime_error(status_t error_code, char const* api_function_name, const std::string& what_arg = "") :
		std::runtime_error(std::string(api_function_name)
			+ ": " + what_arg + (what_arg.empty() ? "" : ": ") + detail::description_of_cstr(error_code)),
		code_(error_code), api_function_name_(api_function_name)
	{ }
	///@endcond
	explicit runtime_error(status::named_t error_code) :
		runtime_error(static_cast<status_t>(error_code)) { }
	runtime_error(status::named_t error_code, char const* api_function_name, const std::string& what_arg = "") :
		runtime_error(static_cast<status_t>(error_code), api_function_name, what_arg) { }

protected:
	runtime_error(status_t error_code, char const* api_function_name, std::runtime_error&& err) :
		std::runtime_error(std::move(err)), code_(error_code), api_function_name_(api_function_name)
	{ }

public:
	/// Construct a runtime error which will not produce the default description for the error code,
	/// but rather only the specified message.
	static runtime_error with_message_override(
		status_t error_code, char const* api_function_name, std::string const& complete_what_arg)
	{
		return { error_code, api_function_name, std::runtime_error(complete_what_arg) };
	}

	/// Obtain the OpenCL status code which resulted in this error being thrown.
	status_t code() const { return code_; }

	/// Obtain the OpenCL API function which failed and produced the error status
	char const* api_function_name() const { return api_function_name_; }

private:
	status_t code_;
	char const* api_function_name_;
};

/// A macro for only throwing an error if we've failed - which also ensures no string
/// is constructed unless we actually need to throw
#define throw_if_error_lazy(status_expr__, api_function_name__, ... ) \
do { \
	auto evaluated_status__ = status_expr__; \
	if (::opencl::is_failure(evaluated_status__)) { \
		throw ::opencl::runtime_error(evaluated_status__, api_function_name__, __VA_ARGS__); \
	} \
} while(false)

#define return_or_throw_if_error(value_or_status_expr__, api_function_name__, ... ) \
do { \
	auto evaluated_vos = std::move(value_or_status_expr__); \
	if (evaluated_vos) { return *evaluated_vos; } \
	else { \
		throw ::opencl::runtime_error(evaluated_vos.status, api_function_name__, __VA_ARGS__); \
	} \
} while(false)


/**
 * Do nothing... unless the status indicates an error, in which case
 * a @ref opencl::runtime_error exception is thrown
 *
 * @note Using these functions means the string will (almost certainly) be constructed,
 * hence you might want to use the @ref throw_if_error_lazy macro instead
 *
 * @param status should be @ref status::success  - otherwise an exception is thrown
 * @param api_function_name the OpenCL C API function which was invoked and returned @p status
 * @param message An extra description message to add to the exception
 */
///@{
inline void throw_if_error(status_t status, char const* api_function_name, const std::string& message) noexcept(false);
inline void throw_if_error(status_t status, char const* api_function_name, std::string&& message) noexcept(false);
///@}

/**
 * Does nothing - unless the status indicates an error, in which case
 * a @ref opencl::runtime_error exception is thrown
 *
 * @note Using these functions means the string will (almost certainly) be constructed,
 * hence you might want to use the @ref throw_if_error_lazy macro instead
 *
 * @param status should be @ref opencl::status::success - otherwise an exception is thrown
 */
inline void throw_if_error(status_t status) noexcept(false)
{
	if (is_failure(status)) { throw runtime_error(status); }
}

} // namespace opencl

#endif // OPENCL_WRAPPERS_ERROR_HPP_
