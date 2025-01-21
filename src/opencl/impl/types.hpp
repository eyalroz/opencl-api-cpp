
#ifndef OPENCL_WRAPPERS_IMPL_TYPES_HPP_
#define OPENCL_WRAPPERS_IMPL_TYPES_HPP_

#include "../types.hpp"
#include "../util/miscellany.hpp"

namespace opencl {

inline access_kind_t operator&(access_kind_t lhs, access_kind_t rhs)
{
    return static_cast<access_kind_t>(static_cast<unsigned char>(lhs) & static_cast<unsigned char>(rhs));
}

inline access_kind_t operator|(access_kind_t lhs, access_kind_t rhs)
{
    return static_cast<access_kind_t>(static_cast<unsigned char>(lhs) | static_cast<unsigned char>(rhs));
}

namespace memory {

inline host_and_kernel_access_t operator&(host_and_kernel_access_t lhs, host_and_kernel_access_t rhs)
{
    return { lhs.host & rhs.host, lhs.kernel & rhs.kernel };
}

inline host_and_kernel_access_t operator|(host_and_kernel_access_t lhs, host_and_kernel_access_t rhs)
{
    return { lhs.host | rhs.host, lhs.kernel | rhs.kernel };
}

constexpr host_and_kernel_access_t full_access() noexcept
{
    return { access_kind_t::read_and_write, access_kind_t::read_and_write };
}

#ifdef CL_VERSION_2_0
namespace shared_virtual {

inline access_spec_t read_and_write() noexcept
{
    return access_spec_t { access_kind_t::read_and_write, coarse_grain, no_atomics };
}

} // namespace shared_virtual
#endif

} // namespace memory

namespace detail {

template <typename T, size_t Capacity, typename BinaryOp>
CONSTEXPR_CPP14 static_dynarray<T, Capacity> apply_binary_op(
    static_dynarray<T, Capacity> const& lhs,
    static_dynarray<T, Capacity> const& rhs,
    BinaryOp op) noexcept(false)
{
    static_assert(std::is_constructible<T, decltype(op(lhs[0], rhs[0]))>::value,
        "Binary operator result cannot be used to construct the required element type");
    if (lhs.size() != rhs.size()) { throw std::invalid_argument("size mismatch"); }
    return opencl::generate_static_dynarray<Capacity>([&](size_t i) -> T { return op(lhs[i], rhs[i]); });
}

// Clip the maximum-length sequence of zeros at the end
// of a given container, returning the result in a container
// limited to a fixed size.
template <dimensionality_t MaxSize, typename Container>
static_dynarray<typename Container::value_type, MaxSize>
clip_zeros(Container const& container)
{
    using value_type = typename Container::value_type;
    auto pos = std::find_if(container.rbegin(), container.rend(), [](value_type const& v) { return v != 0;});
    auto new_size = std::distance(pos, container.rend());
    // TODO: Make this work:
    // return static_dynarray<value_type, MaxSize>{container.begin(), container.begin() + new_size};
    // ... for now, we'll rely on the container being contiguous:
    return static_dynarray<value_type, MaxSize>(new_size, container.data());
}

} // namespace detail

template <dimensionality_t MaxDimensions>
void dimensions_t<MaxDimensions>::validate_num_dimensions(size_type num_dimensions) NOEXCEPT_IF_NDEBUG
{
#ifndef NDEBUG
    if (num_dimensions > MaxDimensions) {
        throw std::invalid_argument("Dimensions structure for more than "
            + std::to_string(num_dimensions) + " dimensions are not supported");
    }
#else
    (void) num_dimensions;
#endif
}

template <dimensionality_t MaxDimensions>
size_t dimensions_t<MaxDimensions>::volume() const noexcept
{
    return std::accumulate(this->begin(),this->end(), 1,
        std::multiplies<dimension_t>());
}

template <dimensionality_t MaxDimensions>
dimension_index_t dimensions_t<MaxDimensions>::effective_dimensionality() const noexcept
{
    return std::accumulate(this->begin(), this->end(), 0,
        [](dimension_index_t d, size_t next_dim) {
            return d + (next_dim > 1 ? 1 : 0);
        });
}

template <dimensionality_t MaxDimensions>
dimension_index_t dimensions_t<MaxDimensions>::dimensionality() const noexcept
{
    // If you have too many dimensions, then... you have bigger trouble than this narrowing cast
    return static_cast<dimension_index_t>(parent_type::size());
}

template <dimensionality_t MaxDimensions>
CONSTEXPR_CPP20 bool operator==(dimensions_t<MaxDimensions> const& lhs, dimensions_t<MaxDimensions> const& rhs) noexcept
{
    using parent_type = static_dynarray<dimension_t, MaxDimensions>;
    return operator==(static_cast<parent_type const&>(lhs), static_cast<parent_type const&>(rhs));
}

template <dimensionality_t MaxDimensions>
CONSTEXPR_CPP20 bool operator!=(dimensions_t<MaxDimensions> const& lhs, dimensions_t<MaxDimensions> const& rhs) noexcept
{
    return not (rhs == lhs);
}

/// @returns true if the dimensions on the left-hand side divide, elementwise, those
/// on the right-hand side
template <dimensionality_t MaxDimensions>
bool divides(dimensions_t<MaxDimensions> const& lhs, dimensions_t<MaxDimensions> const& rhs)
{
    if (lhs.dimensionality() != rhs.dimensionality()) {
        throw std::invalid_argument("argument dimensionality mismatch");
    }
    auto dimensionality = lhs.dimensionality();
    if (lhs.effective_dimensionality() != dimensionality) {
        throw std::invalid_argument("divisor has effective dimensionality 0 (i.e. is an empty volume)");
    }
    // Poor man's all_of applied after zip
    for (dimension_index_t i = 0; i < dimensionality; ++i) {
        if (not rhs[i] % lhs[i] != 0) { return false; }
    }
    return true;
}

template <dimensionality_t MaxDimensions>
dimensions_t<MaxDimensions> round_up(dimensions_t<MaxDimensions> const& dims, dimensions_t<MaxDimensions> const& moduli)
{
#ifndef NDEBUG
    if (dims.dimensionality() != moduli.dimensionality()) {
        throw std::invalid_argument("argument dimensionality mismatch");
    }
    if (moduli.volume() == 0) {
        throw std::invalid_argument(
            "the moduli have a 0-dimension for some of axes - cannot round up to a multiple of their volumen");
    }
#endif
    auto round_up_single_dim = [&](dimension_index_t i) -> dimension_t { return detail::round_up(dims[i], moduli[i]); };
    return generate_static_dynarray<MaxDimensions>(dims.dimensionality(), round_up_single_dim);
}

template <dimensionality_t MaxDimensions>
dimensions_t<MaxDimensions> operator*(dimensions_t<MaxDimensions> lhs, dimensions_t<MaxDimensions> rhs)
{
    if (lhs.dimensionality() != rhs.dimensionality()) {
        throw std::invalid_argument("argument dimensionality mismatch");
    }
    return generate_static_dynarray<MaxDimensions>([&](size_t i) { return lhs[i] * rhs[i];});
    // dimensions_t result {lhs.size()};
    // for (size_type i = 0; i < result.size(); ++i) {
    //     result[i] = lhs[i] * rhs[i];
    // }
    // return result;
}

namespace nd_range {

inline dimensions_t point(dimension_index_t dimensionality) noexcept
{
    auto result = dimensions_t(dimensionality);
    std::fill(result.begin(), result.end(), 1);
    return result;
}

inline composite_dimensions_t composite_dimensions_t::point(dimension_index_t dimensionality) noexcept
{
    return { nd_range::point(dimensionality), nd_range::point(dimensionality) };
}

inline void composite_dimensions_t::round_up_overall_dimensions()
{
    overall = round_up(overall, workgroup);
}

} // namespace nd_range

namespace program {

inline char const* name_of(build_status_t status)
{
    switch (status) {
    case build_status_t::none: return "none";
    case build_status_t::error: return "error";
    case build_status_t::success: return "success";
    case build_status_t::in_progress:
    default:
        return "in progress";
    }
}

} // namespace program

namespace image {

// inline dimensionality_t dimensionality(dimensions_t const& dims) NOEXCEPT_IF_NDEBUG
// {
// #ifndef NDEBUG
//     if ( (dims[1] > 0 and dims[0] == 0) or (dims[2] > 0 and dims[1] == 0) ) {
//         throw std::invalid_argument(
//             "Invalid image dimensions: Positive-dimensions must be contiguous beginning with the first dimension");
//     }
// #endif
//     return (dims[0] > 0) + (dims[1] > 0) + (dims[2] > 0);
// }

inline box_spec_t rect(dimensions_t origins, dimensions_t extents)
{
    box_spec_t result;
    result.extents = to_static_dynarray<max_image_dimensions>(extents);
    result.origin = to_static_dynarray<max_image_dimensions>(origins);
    return result;
}

} // namespace image

// This type is rather similar to std::expected, but it can cannibalize
// one of the 'error' values as an indicator of success. Also, it's more
// struct'y and less class'y.
template <typename T>
struct value_or_status_t {
    static_assert(not std::is_same<T, bool>::value, "the value type can't be bool (to avoid confusion)");
    // Note: We can't require T not to be the same as status_t, because status_t is, alas,
    // just an alias for some integer.
    // static_assert(std::is_trivially_constructible<T>::value);
    using value_type = T;
    T value;
    status_t status; // When this is status::success, value is valid; otherwise, it isn't.

    bool has_value() const noexcept { return is_success(status); }
    operator bool() const noexcept { return has_value(); }
    T const& operator*() const noexcept { return value; }
    T& operator*() noexcept { return value; }
    T* operator->() noexcept { return &value; }
    T const* operator->() const noexcept { return &value; }
    T get() const
    {
        if (not has_value()) {
            throw std::invalid_argument("Attempt to use a value which had failed to be obtained, with "
                "error " + std::string{description_of(status)});
        }
        return value;
    }
    T value_or(const T& fallback) const noexcept { return has_value() ? value : fallback; }
};

namespace device {

namespace partition {

inline spec_t equally(count_t num_parts) { return spec_t{mode_t::equally, { num_parts } }; }

inline spec_t by_counts(span<compute_unit_count_t> counts)
{
    spec_t result;
    result.mode = mode_t::by_counts;
    result.subspec.compute_unit_counts = counts;
    return result;
}

inline spec_t by_affinity(affinity_domain_t affinity_domain)
{
    spec_t result {};
    result.mode = mode_t::by_affinity_domain;
    result.subspec.affinity_domain = affinity_domain;
    return result;
}

} // namespace partition

} // namespace device

inline char const* name_of(device::partition::mode_t mode)
{
    using pm = device::partition::mode_t;
    switch (mode) {
    case pm::equally:                return "equally";
    case pm::by_compute_unit_counts: return "by compute units count";
    case pm::by_affinity_domain:     return "by affinity domain";
    }
    return nullptr;
}

inline char const* name_of(device::affinity_domain_t domain)
{
    using ad = device::affinity_domain_t;
    switch (domain) {
    case ad::numa:               return "NUMA";
    case ad::l4_cache:           return "L4 cache";
    case ad::l3_cache:           return "L3 cache";
    case ad::l2_cache:           return "L2 cache";
    case ad::l1_cache:           return "L1 cache";
    case ad::next_partitionable: return "next partitionable affinity domain";
    }
    return nullptr;
}

inline box_spec_t rect(dimensions_t<2> origins, dimensions_t<2> extents, dimension_t row_pitch)
{
    box_spec_t result;
    result.extents = to_static_dynarray<box_spec_t::max_dimensions>(extents);
    result.origin = to_static_dynarray<box_spec_t::max_dimensions>(origins);
    result.pitches[0] = row_pitch;
    return result;
}

inline box_spec_t rect(dimensions_t<2> origins, dimensions_t<2> extents)
{
    return rect(origins, extents, extents[0] + origins[0]);
}

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_TYPES_HPP_
