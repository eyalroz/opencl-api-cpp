/**
* @file
 *
* @brief Definition of the @ref ownership_token class.
 */
#ifndef OPENCL_WRAPPERS_UTIL_OWERSHIP_TOKEN_HPP_
#define OPENCL_WRAPPERS_UTIL_OWERSHIP_TOKEN_HPP_

#include <utility>

namespace opencl {
namespace detail {
/**
 * A class intended to represent conceptual ownership tokens for entities which
 * require explicit token-obtainment and token-destruction - encapsulating
 * (most) of that logic for the methods of construction, assignment and destruction
 * - with templates allowing adaptation to each token's specifics.
 *
 * @tparam Release A function which can release a token, to be invoked on destruction
 * @tparam ReleaseState The information obtained on token creation, and possibly
 *     other relevant context, necessary for invoking the @tref Release function.
 */
template <typename Release, typename ReleaseState>
class token_holder {
    using release_type = Release;
    using release_state_type = ReleaseState;

protected:
    bool has_token_;
    release_state_type release_state_;

public: // non-mutators
    operator bool() const noexcept { return has_token_; }
    bool has_token() const noexcept { return has_token_; }
public:
    token_holder(bool owning, release_state_type release_state) noexcept
        : has_token_(owning), release_state_(std::move(release_state)) {}
    // refuse ownership on copy
    token_holder(token_holder const& other) noexcept
        : has_token_(false), release_state_(other.release_state_) {}
    // take ownership on move
    token_holder(token_holder&& other) noexcept
        : has_token_(other.has_token_), release_state_(other.release_state_) { other.has_token_ = false; }
    ~token_holder() noexcept(noexcept(Release{}(release_state_)))
    {
        if (has_token_) { Release{}(release_state_); }
    }
    token_holder& operator=(token_holder const& other) = delete;
    friend void swap(token_holder& a, token_holder& b) noexcept
    {
        std::swap(a.has_token_, b.has_token_);
        std::swap(a.release_state_, b.release_state_);
    }
    token_holder& operator=(token_holder&& other) noexcept
    {
        swap(*this, other);
        return *this;
    }
};

} // namespace detail
} // namespace opencl

#endif // OPENCL_WRAPPERS_UTIL_OWERSHIP_TOKEN_HPP_
