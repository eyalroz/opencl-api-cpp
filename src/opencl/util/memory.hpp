/**
 * @file
 *
 * @brief Miscellaneous utility code realted to memory allocation and deallocation
 */
#ifndef OPENCL_WRAPPERS_UTIL_MEMORY_HPP_
#define OPENCL_WRAPPERS_UTIL_MEMORY_HPP_

#include "span.hpp"

namespace opencl {
namespace detail {

// Notes:
// 1. This applies the default T constructor to each element!
// 2. The result must be deleted using array-delete
template <typename T>
T* operator_new_array(size_t size)
{
    return new T[size];
}

// Note: This does _not_ construct the T elements - which must each
// be constructed using placement-new
template <typename T>
T* operator_new(size_t num_elements)
{
    return static_cast<T*>(::operator new(sizeof(T)*num_elements));
}

template <typename T>
span<T> c_malloc(size_t num_elements)
{
    auto ptr = std::malloc(num_elements * sizeof(T));
    if (not ptr) {
        throw std::bad_alloc{};
    }
    return { ptr, num_elements };
}

// Take memory that's been allocated without any construction,
// e.g. with ::operator new(size_t) , and perform T constructions
// so as to make it actually usable as a span of T's.
template <typename T, typename Generator>
void elementwise_construct(span<T> sp, Generator && generator_by_index)
noexcept(noexcept(new(&sp.data()[0]) T(generator_by_index(0))))
{
    for (size_t i = 0; i < sp.size(); i++) {
        new(&sp.data()[i]) T(generator_by_index(i));
    }
}

/// @note if a nullptr happens to be deleted - that's not a problem;
/// it is supported both by C++ delete operators and C's free(), and
/// the behavior is guarnateed to be no-op.
///
/// @todo consider having the deleters take a memory region rather
/// than a span
///@{
template <typename T> void operator_delete_array(span<T> sp) { delete[] sp.data(); }
template <typename T> void operator_delete      (span<T> sp) { ::operator delete(sp.data()); }
template <typename T> void c_free               (span<T> sp) { std::free(sp.data()); }
///@}

template <typename T>
void elementwise_destruct(span<T> sp)
{
    for (auto& element : sp) { element.~T(); }
}

template <typename T>
void elementwise_destruct_then_delete(span<T> sp)
{
    elementwise_destruct(sp);
    operator_delete(sp);
}

} // namespace detail
} // namespace opencl

#endif //OPENCL_WRAPPERS_UTIL_MEMORY_HPP_
