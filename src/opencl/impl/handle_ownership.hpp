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

// For each handle, the static members of this traits struct
// must be initialized, so that the classes holding that handle
// can use the handle ownership mechanism
template <typename Handle>
struct handle_traits {
    using handle_type = Handle;
    using release_type = cl_int (*)(Handle);
    using retain_type = cl_int (*)(Handle);

    // Notes:
    // 1. We're using methods rather than variables, because inline variables - here or
    //    in other header files - require C++17.
    // 2. The traits contain info about clRetainXXX functions - evne though we don't
    //    currently make any use of them.

    static release_type release();
    static char const* release_name();
    static retain_type retain();
    static char const* retain_name();
};
// ... and note that all handles are actually pointers

#define OCLW_DEFINE_HANDLE_TRAITS(_type_name, _handle_type, _release_func, _retain_func) \
template <> inline handle_traits<_handle_type>::release_type \
handle_traits<_handle_type>::release() { return _release_func; }; \
template <> inline char const* handle_traits<_handle_type>::release_name() { return STRINGIFY(_release_func); } \
template <> inline handle_traits<_handle_type>::retain_type \
handle_traits<_handle_type>::retain() { return _retain_func; } \
template <> inline char const* handle_traits<_handle_type>::retain_name() { return STRINGIFY(_retain_func); }


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
        auto status = traits::release()(handle);
        throw_if_error_lazy(status, traits::release_name(), "Releasing " + opencl::detail::identify(handle));
    }
#ifndef OCLW_THROW_IN_DESTRUCTORS
    catch (...) {}
#endif
}

} // namespace detail
} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_OWERSHIP_TOKEN_HPP_
