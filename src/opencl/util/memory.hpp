/**
 * @file
 *
 * @brief Non-OpenCL-specific utility code realted to memory allocation
 * and deallocation
 */
#ifndef OPENCL_WRAPPERS_UTIL_MEMORY_HPP_
#define OPENCL_WRAPPERS_UTIL_MEMORY_HPP_

#include "span.hpp"

namespace opencl {
namespace detail {

// Notes:
//
//
/**
 * Allocate a number of contiguous type-T elements using C++'s `operator new[]`
 *
 * @return a raw pointer to the first element
 *
 * @notes:
 *
 *  1. This applies the default T constructor to each element!
 *  2. The result must be eventually deleted using `delete[]`
 *     (e.g. using the @ref operator_delete_array function)
 */
template <typename T>
T* operator_new_array(size_t size)
{
    return new T[size];
}

/**
 * Allocate space for a number of type-T elements using C++'s `operator new`
 *
 * @return a raw `T`-pointer to beginning of the allocated area
 *
 * @notes
 *
 * 1. This does _not_ construct T values; one must construct those using
 *    placement-new at each element location.
 * 2. Memory allocator with this function must be deleted with the non-array
 *    `operator delete`.
 */
template <typename T>
T* operator_new(size_t num_elements)
{
    return static_cast<T*>(::operator new(sizeof(T)*num_elements));
}

/**
 * Allocate space for a number of type-T elements using the @ref malloc() function,
 * from the standard C library.
 *
 * @notes
 *
 * 1. This does _not_ construct T values; one must construct those using
 *    placement-new at each element location.
 * 2. Memory allocator with this function must be deleted with the @ref free()
 *    function from the C standard library
 */
template <typename T>
span<T> c_malloc(size_t num_elements)
{
    auto ptr = std::malloc(num_elements * sizeof(T));
    if (not ptr) {
        throw std::bad_alloc{};
    }
    return { ptr, num_elements };
}

/**
 * Take memory that's been allocated without any construction,
 * e.g. with ::operator new(size_t) , and perform move-constructions of
 * type-T values in the relevant memory arrays - so as to make it
 * actually usable as a span of T's.
 *
 * @param[in] generator_by_index A function which generates T values
 * for the different intended elements of the span.
 */
template <typename T, typename Generator>
void elementwise_construct(span<T> sp, Generator && generator_by_index)
noexcept(noexcept(new(&sp.data()[0]) T(generator_by_index(0))))
{
    for (size_t i = 0; i < sp.size(); i++) {
        new(&sp.data()[i]) T(generator_by_index(i));
    }
}
/**
 * Deleter functions corresponding to allocation functions specified earlier
 *
 * @note
 *
 * 2. When the correspond allocator function did not construjct - the deleter
 *    function does not delete either.
 * 1. if the span passed to one of these happens to have a nullptr as the data
 *    point  that's not a problem: Silently ignoring nullptr's is a feature of both
 *    the C++ delete operators and C's free(); the behavior is guarnateed to be no-op.
 *
 * @todo consider having the deleters take a memory region rather than a span.
 */
///@{
template <typename T> void operator_delete_array(span<T> sp) { delete[] sp.data(); }
template <typename T> void operator_delete      (span<T> sp) { ::operator delete(sp.data()); }
template <typename T> void c_free               (span<T> sp) { std::free(sp.data()); }
///@}

/// Apply a destructor, `T::~T`, to every T element.
template <typename T>
void elementwise_destruct(span<T> sp)
{
    for (auto& element : sp) { element.~T(); }
}

/// Perform both elementwise destruction and an untyped deletion of
/// a span of consecutive, live, type-T elements.
template <typename T>
void elementwise_destruct_then_delete(span<T> sp)
{
    elementwise_destruct(sp);
    operator_delete(sp);
}

} // namespace detail
} // namespace opencl

#endif //OPENCL_WRAPPERS_UTIL_MEMORY_HPP_
