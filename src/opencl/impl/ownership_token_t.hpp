#ifndef OPENCL_WRAPPERS_IMPL_OWERSHIP_TOKEN_HPP_
#define OPENCL_WRAPPERS_IMPL_OWERSHIP_TOKEN_HPP_

#include "../types.hpp"
#include "../identify.hpp"

#ifndef STRINGIFY
#define STRINGIFY(_q) #_q
#endif

namespace opencl {
namespace detail {

template <typename Release, typename ReleaseState>
class ownership_token_t {
    using release_type = Release;
    using release_state_type = ReleaseState;

protected:
    bool owning_;
    release_state_type release_state_;

public: // non-mutators
    operator bool() const noexcept { return owning_; }
    bool owning() const noexcept { return owning_; }
public:
    ownership_token_t(bool owning, release_state_type release_state) noexcept
        : owning_(owning), release_state_(std::move(release_state)) {}
    // refuse ownership on copy
    ownership_token_t(ownership_token_t const& other) noexcept
        : owning_(false), release_state_(other.release_state_) {}
    // take ownership on move
    ownership_token_t(ownership_token_t&& other) noexcept
        : owning_(other.owning_), release_state_(other.release_state_) { other.owning_ = false; }
    ~ownership_token_t() noexcept(noexcept(Release{}(release_state_)))
    {
        if (owning_) { Release{}(release_state_); }
    }
    ownership_token_t& operator=(ownership_token_t const& other) = delete;
    friend void swap(ownership_token_t& a, ownership_token_t& b) noexcept
    {
        std::swap(a.owning_, b.owning_);
        std::swap(a.release_state_, b.release_state_);
    }
    ownership_token_t& operator=(ownership_token_t&& other) noexcept
    {
        swap(*this, other);
        return *this;
    }
};

template <typename Handle>
using handle_release = cl_int (*)(Handle);

template <typename Handle>
using handle_retain = cl_int (*)(Handle);

template <typename Handle>
struct handle_traits {};
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
struct handle_release_helper {
    void operator()(typename Wrapper::handle_type handle) OCLW_DESTRUCTOR_NOEXCEPT
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
};

template <typename Wrapper>
class handle_ownership_token_t : public ownership_token_t<handle_release_helper<Wrapper>, typename Wrapper::handle_type> {
    using handle_type = typename Wrapper::handle_type;
    using parent_type = ownership_token_t<handle_release_helper<Wrapper>, handle_type>;
    using parent_type::parent_type;
};

} // namespace detail
} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_OWERSHIP_TOKEN_HPP_
