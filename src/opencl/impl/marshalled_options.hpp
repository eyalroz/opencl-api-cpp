#ifndef OPENCL_WRAPPERS_IMPL_MARSHALLED_OPTIONS_HPP
#define OPENCL_WRAPPERS_IMPL_MARSHALLED_OPTIONS_HPP

#error "here"
#include "launch.hpp"

namespace opencl {

/**
 * Mechanism for finalizing options into a format readily usable by the
 * OpenCL compilation, linking and build functions.
 *
 * @note Don't create instances of this struct yourself unless you have to;
 * use "unmarshalled" options structs, and call their `marshal()` methods as
 * necessary. If you must create them - use the `push_back()` method
 * repeatedly until done with all options.
 */
struct marshalled_options_t {
	using size_type = cl_uint; // Seems reasonable; although... maybe make it size_t?

protected:
	::std::array<option_t, max_num_options> option_buffer;
	::std::array<void*, max_num_options> value_buffer;
	size_type count_ { 0 };
public:
	void push_back(option_t option)
	{
		if (count_ >= max_num_options) {
			throw ::std::invalid_argument("Attempt to push back the same option a second time");
			// If each option is pushed back at most once, the count cannot exist the number
			// of possible options. In fact, it can't even reach it because some options contradict.
			//
			// Note: This check will not catch all repeat push-backs, nor the case of conflicting
			// options - the cuLink methods will catch those. We just want to avoid overflow.
		}
		option_buffer[count_] = option;
		count_++;
	}
protected:
	template <typename I>
	void* process_value(typename ::std::enable_if<::std::is_integral<I>::value, I>::type value)
	{
		return reinterpret_cast<void*>(static_cast<uintptr_t>(value));
	}

	template <typename T>
	void* process_value(T* value)
	{
		return static_cast<void*>(value);
	}

	void* process_value(bool value) { return process_value<int>(value ? 1 : 0); }

	void* process_value(caching_mode_t<memory_operation_t::load> value)
	{
		using ut = typename ::std::underlying_type<caching_mode_t<memory_operation_t::load>>::type;
		return process_value(static_cast<ut>(value));
	}

public:
	/**
	 * This method (alone) is used to populate this structure.
	 *
	 * @note The class is not a standard container, and this method cannot be
	 * reversed or undone, i.e. there is no `pop_back()` or `pop()`.
	 */
	template <typename T>
	void push_back(option_t option, T value)
	{
		push_back(option);
		process_value(value);
		// Now set value_buffer[count-1]...
		value_buffer[count_-1] = process_value(value);
	}

	const option_t* options() const { return option_buffer.data(); }
	const void * const * values() const { return value_buffer.data(); }
	size_type count() const { return count_; }
};

} // namespace opencl

#endif //OPENCL_WRAPPERS_IMPL_MARSHALLED_OPTIONS_HPP
