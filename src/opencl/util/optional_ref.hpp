/**
 * @file
 *
 * @brief An implementation of a simplistic optional-reference class
 * (as optional<T&> is problematic semantically)
 */
#ifndef OPENCL_WRAPPERS_OPTIONAL_REF_HPP_
#define OPENCL_WRAPPERS_OPTIONAL_REF_HPP_

#include "optional.hpp"

namespace opencl {

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

} // namespace opencl

#endif //OPENCL_WRAPPERS_OPTIONAL_REF_HPP_
