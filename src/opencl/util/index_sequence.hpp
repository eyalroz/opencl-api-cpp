/**
 * @file
 *
 * @brief A simplistic implementation of std::index_sequence, backported from C++14
 *
 * @note There is no OpenCL-specific code in this file, other than the name of the
 * enclosing namespace
 */
#ifndef OPENCL_WRAPPERS_INDEX_SEQUENCE_HPP
#define OPENCL_WRAPPERS_INDEX_SEQUENCE_HPP

namespace opencl {

#if __cplusplus >= 201402L

using std::index_sequence;
using std::make_index_sequence;

#else

template<std::size_t... Indices>
struct index_sequence {
    using value_type = std::size_t;
    static constexpr std::size_t size() { return sizeof...(Indices); }
};

template<std::size_t I, std::size_t N, std::size_t... Indices>
struct make_index_sequence_helper
{
    using type = typename make_index_sequence_helper<I + 1, N, Indices..., I>::type;
};

template<std::size_t N, std::size_t... Indices>
struct make_index_sequence_helper<N, N, Indices...>
{
    using type = index_sequence<Indices...>;
};
template<std::size_t N>
using make_index_sequence = typename make_index_sequence_helper<0, N>::type;
#endif // __cplusplus >= 201402L

} // namespace opencl

#endif //OPENCL_WRAPPERS_INDEX_SEQUENCE_HPP
