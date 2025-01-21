
#ifndef OPENCL_WRAPPERS_LAUNCH_CONFIG_IMPL_HPP_
#define OPENCL_WRAPPERS_LAUNCH_CONFIG_IMPL_HPP_

#include "../launch_config.hpp"
#include "../device_capabilities.hpp"
#include "../identify.hpp"

namespace opencl {

namespace kernel {
namespace detail {

// void validate_workgroup_dimension_compatibility(device_t const&, nd_range::workgroup_dimensions_t);
// void validate_grid_dimension_compatibility(device_t const&, nd_range::dimension_t);
// void validate_global_dimension_compatibility(device_t const&, nd_range::dimension_t);
// void validate_dimension_compatibility(device_t const&, nd_range::composite_dimensions_t);

inline void validate_workgroup_dimensions(nd_range::workgroup_dimensions_t const& dims)
{
    if (dims.volume() == 0) {
        throw std::invalid_argument("Zero-volume workgroup dimensions provided");
    }
}

inline void validate_grid_dimensions(nd_range::dimensions_t const& grid_dims)
{
    if (grid_dims.volume() == 0) {
        throw std::invalid_argument("Zero-volume grid dimensions provided");
    }
}

inline void validate_overall_dimensions(nd_range::dimensions_t const& nd_range_dims)
{
    if (nd_range_dims.volume() == 0) {
        throw std::invalid_argument("Zero-volume overall dimensions provided");
    }
}

inline void validate_composite_dimensions(nd_range::composite_dimensions_t const& dims)
{
    validate_workgroup_dimensions(dims.workgroup);
    validate_overall_dimensions(dims.overall);
}

// Note: The reason for the verbose name is the identity of the block and nd_range dimension types
inline void validate_workgroup_dimension_compatibility(device_t const& device, nd_range::workgroup_dimensions_t const& dimensions)
{
    auto caps = device.capabilities();
    auto max_dimensionality = caps.maximum_grid_dimensionality();
    if (dimensions.dimensionality() > max_dimensionality) {
        std::ostringstream oss;
        oss << "Workgroup dimensions " << nd_range::detail::identify(dimensions)
            << " are of dimensionality " << dimensions.dimensionality()
            << " , while " + opencl::detail::identify(device)
            << " only supports launches of dimensionality up to " << max_dimensionality;
        throw std::invalid_argument(oss.str());
    }
    auto max_threads_per_workgroup = caps.maximum_workgroup_size();
    auto volume = dimensions.volume();
    if (volume > max_threads_per_workgroup) {
        std::ostringstream oss;
        oss << "Workgroup dimensions " << nd_range::detail::identify(dimensions)
            << "have " << volume << " threads overall, exceeding " << max_threads_per_workgroup
            << " , the maximum number of threads per workgroup supported by "
            << opencl::detail::identify(device);
        throw std::invalid_argument(oss.str());
    }
}

inline void validate_grid_dimension_compatibility(device_t const& device, nd_range::dimensions_t const& grid_dimensions)
{
    (void) device;
    (void) grid_dimensions;
    throw std::runtime_error("Unimplemented");
}

inline void validate_overall_dimension_compatibility(device_t const&, nd_range::dimensions_t const& dims)
{
    (void) dims;
    throw std::runtime_error("Unimplemented");
}

inline void validate_dimension_compatibility(device_t const& device, nd_range::composite_dimensions_t const& dims)
{
    validate_workgroup_dimension_compatibility(device, dims.workgroup);
    validate_overall_dimension_compatibility(device, dims.overall);
}

} // namespace detail

} // namespace kernel

} // namespace opencl

#endif //OPENCL_WRAPPERS_LAUNCH_CONFIG_IMPL_HPP_
