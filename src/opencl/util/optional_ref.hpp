/**
 * @file
 *
 * @brief An implementation of a simplistic optional-reference class
 * (as optional<T&> is problematic semantically); see [Foonathan's blog
 * entry](https://www.foonathan.net/2018/07/optional-reference/#content)
 * about those.
 */
#ifndef OPENCL_WRAPPERS_OPTIONAL_REF_HPP_
#define OPENCL_WRAPPERS_OPTIONAL_REF_HPP_

#include "optional.hpp"

namespace opencl {
/**
 * A class for optionally holding a reference to a value of some type.
 *
 * This class complements the C++17 standard library's optional<T> class,
 * or the backported optional implementation in `optional.hpp`: Those
 * optionals' contained type must not be a reference. The C++ standards
 * committee could not agree on how references-in-optionals are to be
 * handled, and refrained from defining any meaning from them - until
 * C++26. See this discussion of the matter by Jonathan Muller (2018):
 *
 * @url https://www.foonathan.net/2018/07/optional-reference/
 *
 * @note If you are wondering: "What did you choose to do with assignment,
 * then?" - the answer is: Nothing! To avoid any confusion and frustrate
 * everyone equitably, one cannot assign to an optional_ref - one needs
 * to either assign to the value in it, or to `rebind()` it.
 */
template<typename T>
struct optional_ref {
	optional_ref &operator=(const optional_ref &other) = default;

	optional_ref &operator=(optional_ref &&other) = default;

	optional_ref &operator=(const T &value) = delete;

	optional_ref &operator=(const T &&value) = delete;

	optional_ref() noexcept: ptr_(nullptr)
	{ }

	optional_ref(T& v) noexcept : ptr_(&v) 
	{ }

	optional_ref(const optional_ref &other) = default;

	// The following voodoo will allow an optional_ref<const S> to be constructible from an optional<S>,
	// as well as optional_ref<S> from optional<S>
	template <typename U, typename = std::enable_if<std::is_same<typename std::remove_cv<T>::type, U>::value> >
	optional_ref(optional<U> const &opt) noexcept : ptr_( opt ? &(*opt) : nullptr )
	{ }

	optional_ref(nullopt_t) noexcept : ptr_(nullptr) { }

	~optional_ref() noexcept = default;

	T& value() const
	{ return *ptr_; }

	T& value_or(T& fallback_ref) const
	{
		return has_value() ? value() : fallback_ref;
	}

	T& operator*() noexcept { return *ptr_; }
	const T& operator*() const noexcept { return *ptr_; }
	T* operator->() noexcept { return ptr_; }
	const T* operator->() const noexcept { return ptr_; }

	bool has_value() const noexcept
	{ return ptr_ != nullptr; }

	operator bool() const noexcept
	{ return has_value(); }

	void reset() noexcept
	{ ptr_ = nullptr; }

protected:
	T* ptr_;
};

template <typename T>
bool operator==(optional_ref<T> const &lhs, optional_ref<T> const &rhs) noexcept
{ return lhs.has_value() == rhs.has_value() and (not lhs.has_value() or *lhs == *rhs); }

template <typename T>
bool operator!=(optional_ref<T> const &lhs, optional_ref<T> const &rhs) noexcept
{ return not (lhs == rhs); }

} // namespace opencl

#endif //OPENCL_WRAPPERS_OPTIONAL_REF_HPP_
