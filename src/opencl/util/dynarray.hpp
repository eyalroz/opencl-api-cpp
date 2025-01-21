/**
 * @file
 *
 * @brief Contains the @ref opencl::dynarray class
 *
 * @note There is no OpenCL-specific code in this file; the class is usable
 * entirely independently of the OpenCL APIs and GPUs in general. The only
 * mention of OpenCL is the namespace.
 */

#ifndef OPENCL_WRAPPERS_DYNARRAY_HPP_
#define OPENCL_WRAPPERS_DYNARRAY_HPP_

#include "span.hpp"
#include "region.hpp"
#include "memory.hpp"

#include "type_traits.hpp"
#include <cstring>
#include <cassert>

namespace opencl {

/**
 * An `std::dynarray`-inspired class, sans std::allocator: Contiguous storage; size always equal to
 * the capacity, both set at construction time; and dynamic storage, allocated _separately_ from
 * this class itself.
 *
 * @note in owning standard-library containers, allocation is tied up with the container class itself,
 * via an Allocator template parameter. This class forgoes that "pleasure" - which is more feasible
 * for a container which never re-allocates - and simply takes the allocated memory on construction.
 *
 * @note dynarray is similar to the `dynarray` container, which was proposed for, but not finally
 * included in, C++14. It can be though of as a variation on std::array, with the the size and capacity
 * set dynamically, at construction time, rather than statically.
 *
 * @note dynarray = span +non-unique-ownership + non_null . Well, sort of, because this
 * class supports complex construction-allocation and deletion patterns, through deleter objects.
 *
 * @tparam T an individual element in the dynarray
 */
template<typename T>
class dynarray : public span<T> {
public: // type definitions
    using value_type = T;
    using span_type = span<value_type>;
    using const_span_type = span<const value_type>;

    // Exposing some span type definitions, strictly for terseness
    // (they're all visible on the outside anyway)
    using size_type = typename span_type::size_type;
    using pointer = typename span_type::pointer;
    using const_pointer = typename const_span_type::pointer;
    using reference = typename span_type::reference;

    /// Notes:
    /// 1. It may be sufficient to pass a memory_region rather than typed span
    /// 2. The deleter _may not_ perform destruction. This is because if it were allowed to do so,
    ///    the dynarray type would need some indication, to avoid destructing itself
    using deleter_type = void (*)(span_type);

    /// Notes:
    /// 1. For n elements, n * sizeof(T) bytes will be allocated;
    /// 2. What about alignment?
    /// 3. The allocator _may not_ perform construction of the elements. This is because if
    ///    it were allowed to do so,  the dynarray type would need some indication, to avoid
    ///    repeat construction.
    using allocator_type = pointer (*)(size_type num_elements);
    using iterator = pointer;
    using const_iterator = const_pointer;

protected:

    void elementwise_copy_construct(span<T const> const& other) noexcept(noexcept(T(other.data()[0])))
    {
        assert((other.size() == 0 or other.data()) && "trying to construct from null data");
#if __cplusplus >= 201712L
        // TODO: We could use std::integral_value-based dispatching to achieve this
        // with C++11 as well, but - is that worth it?
        if constexpr (std::is_trivially_copy_constructible<T>::value) {
            std::memcpy(span_type::data_, other.data_, sizeof(T) * other.size_);
        }
        else
#endif
        for (size_t i = 0; i < span_type::size(); ++i) {
            new(&span_type::data()[i]) T(other.data()[i]);
        }
    }

    void elementwise_default_construct() noexcept(noexcept(T()))
    {
        if (std::is_trivially_default_constructible<T>::value) { return; }
        for (size_t i = 0; i < span_type::size(); ++i) {
            new(&span_type::data()[i]) T;
        }
    }

    void elementwise_destruct() { for (auto& element : *this) { element.~T(); } }

    void elementwise_destruct_and_delete()
    {
        if (data() != nullptr) {
            elementwise_destruct();
            deleter_(*this);
        }
    }

public: // exposing span data members & adding our own
    using span_type::data;
    using span_type::size; // In elements, not bytes
    allocator_type allocator_;
    deleter_type deleter_;

public: // constructors and destructor

    allocator_type allocator() const noexcept { return allocator_; }
    deleter_type deleter() const noexcept { return deleter_; }

    // Note: span_type's default ctor will create a {nullptr, 0} empty span.
    constexpr dynarray() noexcept : span_type(), allocator_{nullptr}, deleter_{nullptr} {}

    dynarray(dynarray const& other) noexcept(noexcept(other.allocator_(other.size()))) :
        span_type { other.allocator_(other.size()), other.size() },
        allocator_(other.allocator_), deleter_(other.deleter_)
    {
        elementwise_copy_construct( { other.data() , other.size() } );
    }

    // Notes:
    // 1. This template provides constructibility of dynarray<const T> from dynarray<const T>&
    // 2. Perhaps we should consider adding a struct-keyword-based copying ctor, taking a span,
    //    allocator and deallocator, in which case we could use that without resorting to a
    //    a reinterpretation.
    template<typename U>
    dynarray(dynarray<U> const& other) noexcept(noexcept(other.allocator_(other.size_))) :
        dynarray(reinterpret_cast<dynarray<T> const&>(other))
    {
        static_assert(std::is_same<U, T const>::value
            and std::is_assignable<allocator_type, typename dynarray<U>::allocator_type>::value
            and std::is_assignable<deleter_type, typename dynarray<U>::deleter_type>::value,
            "Invalid dynarray initializer");
    }

    // Note: This template provides constructibility of dynarray<const T> from dynarray<const T>&&
    template<typename U>
    dynarray(dynarray<U>&& other) : dynarray{ other.release(), other.allocator_, other.deleter_ }
    {
        static_assert(
            std::is_same<U, T const>::value
            and std::is_assignable<allocator_type, typename dynarray<U>::allocator_type>::value
            and std::is_assignable<deleter_type, typename dynarray<U>::deleter_type>::value,
            "Invalid dynarray initializer");
    }

    dynarray(size_type size, allocator_type allocator, deleter_type deleter) noexcept(noexcept(allocator(size)))
        : span_type{size > 0 ? allocator(size) : nullptr, size}, allocator_(allocator), deleter_(deleter)
    {
        elementwise_default_construct();
    }

    /// Take ownership of an existing span
    ///
    /// @note These ctors are all explicit to prevent accidentally assuming ownership
    /// of a non-owned span when passing to a function, then trying to release that
    /// memory returning from it.
    ///@{
    dynarray(span_type span, allocator_type allocator, deleter_type deleter) noexcept
    : span_type{span}, allocator_{allocator}, deleter_{deleter} { }
    dynarray(pointer data, size_type size, allocator_type allocator, deleter_type deleter) noexcept
    : dynarray(span_type{data, size}, allocator, deleter) { }
    dynarray(memory::region_t region, allocator_type allocator, deleter_type deleter) NOEXCEPT_IF_NDEBUG
        : dynarray(span_type{region.start(), region.size() / sizeof(T)}, allocator, deleter)
    {
#ifndef NDEBUG
        if (sizeof(T) * size != region.size()) {
            throw std::invalid_argument("Attempt to create a dynarray with a memory region which"
                "does not comprise an integral number of areas of the element type size");
        }
#endif
    }

    template <
        typename... Us,
        typename = typename std::enable_if<
            sizeof...(Us) >= 1
            and detail::all_true<std::is_constructible<value_type, Us>::value...>::value
        >::type
    >
    CONSTEXPR_CPP14 dynarray(Us... values) :
        span_type{ detail::operator_new<T>(sizeof...(Us)), sizeof...(Us) },
        allocator_{ detail::operator_new<T> },
        deleter_{ detail::operator_delete<T> }
    {
        T* ptr = data();
        (void) std::initializer_list<void*> { new(ptr++) T(values) ...  };
    }

    ///@}

    /// A move constructor.
    ///
    /// @TODO Can we drop this one in favor of the general move ctor?
    dynarray(dynarray&& other) noexcept : dynarray(other.release(), other.allocator_, other.deleter_) { }

    ~dynarray() noexcept(noexcept(deleter_(*this)))
    {
        elementwise_destruct_and_delete();
#ifndef NDEBUG
        release();
#endif
    }

public: // operators

    CONSTEXPR_CPP14 dynarray& operator=(dynarray const& other) noexcept(false)
    {
        assert((other.size() == 0 or other.data()) && "got a non-empty dynarray with null data pointer");

        allocator_ = other.allocator_;
        deleter_ = other.deleter_;
        auto allocation = allocator_(other.size());
        span_type::operator=(span_type{allocation, other.size()});
        elementwise_copy_construct(other);
        return *this;
    }

    CONSTEXPR_CPP14 dynarray& operator=(dynarray&& other) noexcept
    {
        swap(other);
        return *this;
    }

    /// No plain dereferencing - as there is no guarantee that any object has been
    /// initialized at those locations, nor do we know its type

    constexpr operator memory::const_region_t() const noexcept { return { data(), size() * sizeof(T) }; }

    template<typename = typename std::enable_if<! std::is_const<T>::value>::type>
    constexpr operator memory::region_t() const noexcept { return { data(), size() * sizeof(T) }; }

    constexpr operator span_type() const noexcept { return get(); }

public: // non-mutators
    constexpr span_type get() const noexcept { return { data(), size() }; }

protected: // mutators
    /// Exchange the pointer and deleter with another object.
    void swap(dynarray& other) noexcept
    {
        std::swap<span_type>(*this, other);
        std::swap(allocator_, other.allocator_);
        std::swap(deleter_, other.deleter_);
    }

public:
    /**
     * Release ownership of the stored span
     *
     * @note This is not marked nodiscard by the same argument as for std::unique_ptr;
     * see also @url https://stackoverflow.com/q/60535399/1593077 and
     * @url http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0600r1.pdf
     *
     * @note it is the caller's responsibility to ensure it has a copy of the deleter
     * for the released span.
     */
    span_type release() noexcept
    {
        span_type released { data(), size() };
        span_type::operator=(span_type{ static_cast<T*>(nullptr), 0 });
        // Note that we are _not_ replacing the allocator and deleter
        return released;
    }
}; // class dynarray

#if __cpp_deduction_guides >= 201606
template<typename T, typename... Ts>
dynarray(T, Ts...) -> dynarray<std::enable_if_t<(std::is_same_v<T, Ts> and ...), T>>;
#endif


/// Note: We are not comparing the allocator or deleter
template<typename T>
CONSTEXPR_CPP20 bool operator==(dynarray<T> const& lhs, dynarray<T> const& rhs)
noexcept(noexcept(*lhs.begin() == *rhs.begin()))
{
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

/// Note: We are not comparing the allocator or deleter
template<typename T>
CONSTEXPR_CPP20 bool operator!=(dynarray<T> const& lhs, dynarray<T> const& rhs)
noexcept(noexcept(*lhs.begin() == *rhs.begin()))
{
    return not (rhs == lhs);
}

/**
 * A parallel of std::make_unique_for_overwrite but for @ref dynarray<T>'s, i.e. which maintains
 * the number of elements allocated.
 *
 * @param size the number of elements in the dynarray to be created. It may legitimately be 0.
 * @tparam T the type of elements in the allocated @ref dynarray.
 * @param size The number of @tparam T elements to allocate
 */
template <typename T>
dynarray<T> make_dynarray(size_t size)
{
    static_assert(std::is_default_constructible<T>::value, "Unexpected non-default-constructible type");
    // Note: It _is_ acceptable pass 0 here.
    // See https://stackoverflow.com/q/1087042/1593077
    return dynarray<T>(size, detail::operator_new<T>, detail::operator_delete<T>);
}

/**
 * The alternative to `std::generate` and similar functions, for the dynarray, seeing
 * how its elements must be constructed as it is constructed.
 *
 * @param size the number of elements in the dynarray to be created. It may legitimately be 0.
 * @param generator_by_index
 *     a function for generating new values for move-construction into the new dynarray
 *
 * @tparam T the type of elements in the allocated @ref dynarray.
 * @tparam Generator A type invokable with the element index, to produce a T-constructor-
 *     argument; must be a noexcept function! And the constrctor of T from its result must
 *     also be noexcept. Otherwise we leek memory here
 *
 * @param size The number of @tparam T elements to allocate
 */
///@{
template <typename T, typename Generator>
dynarray<T> generate_dynarray(size_t size, Generator generator_by_index)
{
    // Q: Why do we not use the array allocation operation, new T[size_]?
    // A: Two reasons. First, there is no guarantee T is default-constructible; and
    //    even if it were - we have a _generator_ function which does the
    //    construction, so we can't very well have the new operator do it for us.
    //    The function which _does_ use the array construction operator is
    //    @ref make_dynarray(size_t size)
    //    The second reasons is, that the array allocation also constructs
    //    the elements in a way which necessitates the array destruct operation
    //    to eventually run. However - the dynarray class will
    //    elementwise-destruct its elements before calling the deleter, so that
    //    would be an error.
    // Q: Do I need to check the alignment here? Perhaps allocate more to ensure alignment?
    auto data = detail::operator_new<T>(size);
    auto span_ = span<T>{data, size};
    detail::elementwise_construct(span_, generator_by_index);
    return dynarray<T>(span_, detail::operator_new<T>, detail::operator_delete<T>);
}

template <typename Generator>
dynarray<decltype(std::declval<Generator>()(0))>
generate_dynarray(size_t size, Generator generator_by_index) noexcept
{
    using T = decltype(std::declval<Generator>()(0));
    return generate_dynarray<T, Generator>(size, std::forward<Generator>(generator_by_index));
}
///@}

template <typename T, typename Container>
dynarray<T> to_dynarray(Container const& container)
{
    auto iter = container.begin();
    return generate_dynarray<T>(container.size(), [&](size_t) { return *(iter++); });
}

template <typename Container>
dynarray<typename Container::value_type> to_dynarray(Container const& container)
{
    auto iter = container.begin();
    return generate_dynarray<typename Container::value_type>(container.size(), [&](size_t) { return *(iter++); });
}

template <typename T, typename Container, typename F>
dynarray<T> to_dynarray(Container const& container, F && f)
{
    auto iter = container.begin();
    auto generator = [&](size_t) { return f(*(iter++)); };
    return generate_dynarray<T, decltype(generator)>(container.size(), generator);
}

template <typename Container, typename F>
dynarray<decltype(std::declval<F>()(std::declval<typename Container::value_type>()))>
to_dynarray(Container const& container, F && f)
{
    auto iter = container.begin();
    using value_type = decltype(f(*iter));
    return generate_dynarray<value_type>(container.size(), [&](size_t) { return f(*(iter++)); });
}
} // namespace opencl

#endif // OPENCL_WRAPPERS_DYNARRAY_HPP_
