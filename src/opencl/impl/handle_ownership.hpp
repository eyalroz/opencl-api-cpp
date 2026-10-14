/**
* @file
 *
 * @brief A mechanism for defining handle-specific traits used for the 
 * handle ownership token mechanism in @ref ../handle_ownership.hpp . The
 * implementations of specific wrapper classes which use a handle ownership
 * token will use the @ref OCLW_DEFINE_HANDLE_TRAITS macro, made available here,
 * to define explicit specializations of the traits class.
 */
#ifndef OPENCL_WRAPPERS_IMPL_OWERSHIP_TOKEN_HPP_
#define OPENCL_WRAPPERS_IMPL_OWERSHIP_TOKEN_HPP_

#include "../handle_ownership.hpp"
#include "../identify.hpp"
#include "../error.hpp"

#ifndef STRINGIFY
#define STRINGIFY(_q) #_q
#endif

namespace opencl {
namespace detail {

template <typename Handle>
using handle_release = cl_int (*)(Handle);

template <typename Handle>
using handle_retain = cl_int (*)(Handle);

template <typename Handle>
struct handle_traits;
// ... and note that all handles are actually pointers

#define OCLW_DEFINE_HANDLE_TRAITS(_type_name, _handle_type, _release_func, _retain_func) \
template <> \
struct handle_traits<_handle_type> { \
    using type = _handle_type; \
    using release_type = handle_release<type>; \
    using retain_type = handle_release<type>; \
    static constexpr release_type release = _release_func; \
    static constexpr char const* release_name = STRINGIFY(_release_func); \
    static constexpr retain_type retain = _retain_func; \
    static constexpr char const* retain_name = STRINGIFY(_retain_func); \
};

template <typename Wrapper>
void handle_release_helper<Wrapper>::operator()(
    typename Wrapper::handle_type handle) OCLW_DESTRUCTOR_NOEXCEPT
{
    using handle_type = typename Wrapper::handle_type;
    using traits = handle_traits<handle_type>;
#ifndef OCLW_THROW_IN_DESTRUCTORS
    try
#endif
    {
        auto status = traits::release(handle);
        throw_if_error_lazy(status, traits::release_name, "Releasing " + opencl::detail::identify(handle));
    }
#ifndef OCLW_THROW_IN_DESTRUCTORS
    catch (...) {}
#endif
}

} // namespace detail
} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_OWERSHIP_TOKEN_HPP_
