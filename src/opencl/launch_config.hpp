
#ifndef OPENCL_WRAPPERS_LAUNCH_CONFIG_WRAPPER_HPP_
#define OPENCL_WRAPPERS_LAUNCH_CONFIG_WRAPPER_HPP_

#include "types.hpp"
#include "util/optional.hpp"

namespace opencl {

namespace kernel {
namespace detail {

// Note: The reason for the verbose name is the identity of the block and nd_range dimension types
void validate_workgroup_dimensions(nd_range::workgroup_dimensions_t const& dims);
void validate_grid_dimensions(nd_range::dimensions_t const& grid_dims);
void validate_overall_dimensions(nd_range::dimensions_t const& overall_dims);
void validate_composite_dimensions(nd_range::composite_dimensions_t const& dims);

void validate_workgroup_dimension_compatibility(device_t const&, nd_range::workgroup_dimensions_t const&);
void validate_grid_dimension_compatibility(device_t const&, nd_range::dimensions_t const&);
void validate_overall_dimension_compatibility(device_t const&, nd_range::dimensions_t const&);
void validate_dimension_compatibility(device_t const&, nd_range::composite_dimensions_t const&);

} // namespace detail

struct launch_configuration_t {
    nd_range::composite_dimensions_t dimensions;
    // TODO: Introduce support for dynamic shared memory; we would have the kernel
    // indicate which parameter to stick it in, but this structure say how much to use
    optional<nd_range::dimensions_t> offset;
    nd_range::dimension_t const * get_offset() const
    {
        return offset ? offset->data() : nullptr;
    }
};

} // namespace kernel

} // namespace opencl

#endif //OPENCL_WRAPPERS_LAUNCH_CONFIG_WRAPPER_HPP_
