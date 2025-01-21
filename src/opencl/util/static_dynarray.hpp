/**
 * @file
 *
 * @brief Contains the @ref opencl::static_dynarray class
 *
 * @note There is no OpenCL-specific code in this file; the class is usable
 * entirely independently of the OpenCL APIs and GPUs in general. The only
 * mention of OpenCL is the namespace.
 */

#ifndef OPENCL_WRAPPERS_STATIC_STATIC_DYNARRAY_HPP_
#define OPENCL_WRAPPERS_STATIC_STATIC_DYNARRAY_HPP_

#include "type_traits.hpp"
#include "index_sequence.hpp"

#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace opencl {

/**
 * A class combining features of `std::array`, `static_dynarray` and `static_vector`: 
 * Fixed capacity set at compile-time; size set at construction time; and no resizing.
 *
 * @todo mark noexcept carefully where possible
 *
 * @todo add support for capacity 0
 *
 * @todo add support for construction with an initializer list
 *
 * @tparam T an individual element in the static_dynarray
 * @tparam Capacity the size (in T elements) the array can have - and the number of
 * elements its storage will always take up
 */
template<typename T, size_t Capacity>
class static_dynarray {
	static_assert(std::is_copy_assignable<T>::value and
		(std::is_trivially_copy_constructible<T>::value or
		(std::is_default_constructible<T>::value)),
		"Unsupported element type");
public: // enumerations
	enum { capacity = Capacity };

public: // type definitions
	using value_type = T;

	// Most of the following should be the same as for spans
	using size_type = size_t;
	using pointer = value_type*;
	using const_pointer = value_type const*;
	using reference = value_type&;
	using const_reference = value_type const&;
	using iterator = T*; // This is not span_type's iterator, at least iwht libstdc++
	using const_iterator = T const*; // ditto
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

public: // member getters
	constexpr size_t size() const noexcept { return size_;}
	// value_type (&data() noexcept)[capacity] { return data_;}
	// value_type const (&data() const noexcept)[capacity] { return data_;}
	CONSTEXPR_CPP14 pointer data() noexcept { return data_;}
	constexpr const_pointer data() const noexcept { return data_;}

public: // other non-mutator methods
	size_t max_size() const noexcept { return capacity; }
	bool empty() const noexcept { return size_ == 0; }
	bool full() const noexcept { return size_ == capacity; }
	// constexpr span<value_type> as_span() const noexcept { return { data_, size_ }; }

protected: // data members
	size_t size_;
	value_type data_[capacity]; // What about capacity 0?
	// How about using std::array<value_type, capacity> data_; ?

	using storage_type = decltype(data_);

protected:
	// Override our current size and copy the data - with the implicit
	// assumption size < capacity
	CONSTEXPR_CPP14 void initialize_data(size_type size, const_pointer data)
		noexcept(noexcept(std::copy(data, data+size, data_)))
	{
		// TODO: Should we check for null pointer use when size > 0?
		std::copy(data, data+size, data_);
	}

	// Override our current size and move the data - with the implicit
	// assumption size < capacity
	CONSTEXPR_CPP14 void move_initialize_data(size_type size, pointer data)
		noexcept(noexcept(std::move(data, data+size, data_)))
	{
		// TODO: Should we check for null pointer use when size > 0?
		std::move(data, data+size, data_);
	}

public: // constructors and destructor

	// Note: I've decided to the vector constructors pitfall, where ambiguity results from the different possible
	// implementations of values in ctors with one or two values: They may be elements, the may be the size in case
	// of a single value, and they may be a pair of start and end iterators/pointers. This is done by treating
	// anything which could be a sequence of actual elements - as just that. To construct otherwise, you either
	// use the "keyword-like" argument do_not_initialize, or you use named constructor idioms; specifically,
	// generate_dynarray can be used to iterate a range.

#ifndef NDEBUG
#define STATIC_DYNARRAY_MAYBE_INITIALZE_DATA , data_{0}
#else
#define STATIC_DYNARRAY_MAYBE_INITIALZE_DATA
#endif

	constexpr       static_dynarray() noexcept : size_{0} STATIC_DYNARRAY_MAYBE_INITIALZE_DATA {}
	CONSTEXPR_CPP14 static_dynarray(size_type size, const_pointer data)
		noexcept(noexcept(initialize_data(size, data))) : size_{size} { initialize_data(size, data); }

	// TODO: Perhaps drop the enable_if ?
	template <
		typename... Us,
		typename = typename std::enable_if<
			sizeof...(Us) >= 1
			and detail::all_true<std::is_constructible<value_type, Us>::value...>::value
		>::type
	>
	CONSTEXPR_CPP14 static_dynarray(Us... values) noexcept : size_(sizeof...(Us)), data_{ values... } {}

	CONSTEXPR_CPP14 static_dynarray(static_dynarray const& other) noexcept(noexcept(initialize_data(other.size_, other.data_)))
	: size_{other.size_} STATIC_DYNARRAY_MAYBE_INITIALZE_DATA
	{
		assert(other.size_ <= Capacity);
		initialize_data(other.size_, other.data_);
	}
	CONSTEXPR_CPP14 static_dynarray(static_dynarray && other) noexcept(noexcept(move_initialize_data(other.size_, other.data_)))
	: static_dynarray(other) { }
	// what about a plain array ref? and what about CTAD for plain arrays?

	template <size_type OtherCapacity>
	CONSTEXPR_CPP14 static_dynarray(c_array<const value_type, OtherCapacity> const& source) noexcept : static_dynarray(source, OtherCapacity) {}

	struct do_not_initialize {};
	CONSTEXPR_CPP14 static_dynarray(size_type size, do_not_initialize) noexcept
	: size_(size) STATIC_DYNARRAY_MAYBE_INITIALZE_DATA {}

	// Should we default a dtor?
	~static_dynarray() noexcept(noexcept(std::is_nothrow_destructible<value_type>::value)) = default;

public: // mutator methods
	// Note: std::array's fill method returns void. Why? Don't know, returning the array makes
	// a lot more sense to me
	CONSTEXPR_CPP14 static_dynarray& fill(const_reference value) noexcept(noexcept(data_[0] = value))
	{
		std::fill(begin(), end(), value);
		return *this;
	}

	// Does the noexcept matter here?
	CONSTEXPR_CPP14 void swap(static_dynarray& other)
	noexcept(noexcept(move_initialize_data(other.size_, other.data_)))
	{
		move_initialize_data(other.size_, other.data_);
	}

public: // iterators
	CONSTEXPR_CPP14 iterator begin() noexcept { return iterator(data()); }
	constexpr const_iterator begin() const noexcept { return const_iterator(data()); }
	CONSTEXPR_CPP14 iterator end() noexcept { return iterator(data() + size_); }
	constexpr const_iterator end() const noexcept { return const_iterator(data() + size_); }
	CONSTEXPR_CPP14 reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	constexpr const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
	CONSTEXPR_CPP14 reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
	constexpr const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
	constexpr const_iterator cbegin() const noexcept { return const_iterator(data()); }
	constexpr const_iterator cend() const noexcept { return const_iterator(data() + size_); }
	constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
	constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

public: // individual element access methods & operators
	CONSTEXPR_CPP14 reference operator[](size_type n) noexcept { return data_[n]; }
	constexpr const_reference operator[](size_type n) const noexcept { return data_[n]; }

	CONSTEXPR_CPP17 reference at(size_type n) noexcept(false) {
		if (n > size_) {
			throw std::out_of_range("static_dynarray::at() index out of bounds");
		}
		return data_[n];
	}

	CONSTEXPR_CPP17 const_reference at(size_type n) const noexcept(false) {
		if (n > size_) {
			throw std::out_of_range("static_dynarray::at() index out of bounds");
		}
		return data_[n];
	}

	CONSTEXPR_CPP14 reference front() noexcept { return data_[0]; }
	constexpr const_reference front() const noexcept { return data_[0]; }
	CONSTEXPR_CPP14 reference back() noexcept { return data_[size_]; }
	constexpr const_reference back() const noexcept { return data_[size_]; }

public: // operators

	CONSTEXPR_CPP14 static_dynarray& operator=(static_dynarray const& other) noexcept(false)
    {
		size_ = other.size_;
		initialize_data(other.size(), other.data());
        return *this;
    }

	CONSTEXPR_CPP14 static_dynarray& operator=(static_dynarray&& other) noexcept
	{
		size_ = other.size_;
		move_initialize_data(other.size_, other.data_);
		return *this;
		// other will be destructed, and our previous pointer - released if necessary
	}

	// constexpr operator memory::const_region_t() const noexcept { return { data(), size() * sizeof(T) }; }

	// template<typename = typename enable_if_t<! std::is_const<value_type>::value>>
	// constexpr operator memory::region_t() const noexcept { return { data(), size() * sizeof(value_type) }; }

	// constexpr operator span_type() const noexcept { return as_span(); }

}; // class static_dynarray

#if __cpp_deduction_guides >= 201606
template<typename T, typename... Ts>
static_dynarray(T, Ts...) -> static_dynarray<std::enable_if_t<(std::is_same_v<T, Ts> and ...), T>, 1 + sizeof...(Ts)>;
#endif

template<typename T, size_t Capacity>
CONSTEXPR_CPP20 bool operator==(static_dynarray<T, Capacity> const& lhs, static_dynarray<T, Capacity> const& rhs)
noexcept(noexcept(std::declval<T>() == std::declval<T>()))
{
	return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template<typename T, size_t Capacity>
CONSTEXPR_CPP20 bool operator!=(static_dynarray<T, Capacity> const& lhs, static_dynarray<T, Capacity> const& rhs)
noexcept(noexcept(lhs == rhs)) { return not (rhs == lhs); }

namespace detail {

template<typename Generator, size_t...Is>
constexpr static_dynarray<decltype(std::declval<Generator>()(0)), sizeof...(Is)>
generate_static_dynarray(Generator gen, index_sequence<Is...>)
noexcept(noexcept(std::declval<Generator>()(0)))
{
	return { gen(Is)... };
}

// Note: If size is too large, it is effectively clipped to sizeof...(Is)
template<typename Generator, size_t Capacity, size_t...Is>
constexpr static_dynarray<decltype(std::declval<Generator>()(0)), Capacity>
generate_static_dynarray(size_t size, Generator&& gen, index_sequence<Is...>)
noexcept(noexcept(std::declval<Generator>()(0)))
{
	using value_type = decltype(std::declval<Generator>()(0));
	static_assert(std::is_default_constructible<value_type>::value,
		"sub-capacity static_dynarray sizes are only possible with a default-constructible element type");
	return (sizeof...(Is) == size) ?
		static_dynarray<value_type, Capacity>{ gen(Is) ... } :
		generate_static_dynarray<Generator, Capacity>(size, std::forward<Generator>(gen),
			make_index_sequence<(sizeof...(Is) == 0 ? 0 : sizeof...(Is)-1)>());
}

} // namespace detail

template<size_t Size, typename Generator>
constexpr static_dynarray<decltype(std::declval<Generator>()(0)), Size>
generate_static_dynarray(Generator gen)
{
	static_assert(not std::is_reference<decltype(std::declval<Generator>()(0))>::value, "Haven't decided what to do with references yet");
	return detail::generate_static_dynarray(gen, make_index_sequence<Size>());
}

template<size_t Capacity, typename Generator>
static_dynarray<decltype(std::declval<Generator>()(0)), Capacity>
CONSTEXPR_CPP14 generate_static_dynarray(size_t size, Generator&& gen) noexcept(false)
{
	static_assert(not std::is_reference<decltype(std::declval<Generator>()(0))>::value, "Haven't decided what to do with references yet");
	if (size > Capacity) { throw std::invalid_argument("Specified size exceeds specified array capacity"); }
	if (size == 0) return {};
	return detail::generate_static_dynarray<Generator, Capacity>(size, std::forward<Generator>(gen), make_index_sequence<Capacity>());
}

template<size_t Capacity, typename T>
static_dynarray<T, Capacity>
CONSTEXPR_CPP14 generate_uniform_static_dynarray(size_t size, T const& v) noexcept(false)
{
	return static_dynarray<T, Capacity>(size).fill(v);
}

// TODO: Consider making Container a plain template, and always taking T explicitly
// (or inferring it from Container::value_type if it's not specified; but - is that
// even possible?
template <size_t Capacity, typename Container>
static_dynarray<typename Container::value_type, Capacity> to_static_dynarray(Container const& container)
{
	auto iter = container.begin();
	auto gen = [&](size_t) { return *(iter++); };
	return generate_static_dynarray<Capacity>(container.size(), gen);
}

template <size_t Capacity, typename Container, typename F>
auto to_static_dynarray(Container const& container, F && f) ->
static_dynarray<decltype(f(container.front())), Capacity>
{
	auto iter = container.begin();
	auto generator = [&](size_t) { return f(*(iter++)); };
	return generate_static_dynarray<Capacity>(container.size(), generator);
}

// TODO: Support generation of static_dynarray's with a lower size than capacity;

} // namespace opencl

#endif // OPENCL_WRAPPERS_STATIC_STATIC_DYNARRAY_HPP_
