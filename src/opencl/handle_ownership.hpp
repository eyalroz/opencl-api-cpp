/**
* @file
 *
 * @brief WRITEME!
 */
#ifndef OPENCL_WRAPPERS_HANDLE_OWNERSHIP_HPP_
#define OPENCL_WRAPPERS_HANDLE_OWNERSHIP_HPP_

#include "util/ownership_token.hpp"
#include "util/type_traits.hpp"

namespace opencl {
namespace detail {

template <typename Wrapper>
struct handle_release_helper {
    void operator()(typename Wrapper::handle_type handle) OCLW_DESTRUCTOR_NOEXCEPT;
};

template <typename Wrapper>
class handle_ownership_token : public token_holder<handle_release_helper<Wrapper>, typename Wrapper::handle_type> {
    using handle_type = typename Wrapper::handle_type;
    using parent_type = token_holder<handle_release_helper<Wrapper>, handle_type>;
    using parent_type::parent_type;
};

} // namespace detail
} // namespace opencl

#endif // OPENCL_WRAPPERS_HANDLE_OWNERSHIP_HPP_
