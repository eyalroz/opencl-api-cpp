#ifndef OPENCL_WRAPPERS_IMPL_BUILDERS_LAUNCH_CONFIG_HPP_
#define OPENCL_WRAPPERS_IMPL_BUILDERS_LAUNCH_CONFIG_HPP_

#ifdef _MSC_VER
// Prevent Microsoft Visual C++ from defining min() and max() macros
// See @url https://stackoverflow.com/q/4913922/1593077
#define NOMINMAX
#endif

#include "../../builders/launch_config.hpp"

#include <limits>
#include <string>
#include <sstream>

namespace opencl {
namespace builders {

namespace detail {

inline void validate_grid_dimension_compatibility(
	device_t const&,
	nd_range::workgroup_dimensions_t) noexcept(false)
{
	// TODO: WRITEME
}

} // namespace detail

namespace nd_range {

namespace detail {

inline dimension_t div_rounding_up_(dimension_t dividend, dimension_t divisor)
{
	auto quotient = dividend / divisor;
		// It is up to the caller to ensure we don't overflow the dimension_t type
	return (divisor * quotient == dividend) ? quotient : quotient + 1;
}

inline dimensions_t div_rounding_up(
	dimensions_t const& dividend,
	dimensions_t const& divisor)
{
	if (dividend.size() != divisor.size()) {
		throw std::invalid_argument(
			"Mismatched number of dimensions between dividend (" + std::to_string(dividend.size()) +
			") and divisor (" + std::to_string(divisor.size()) + ")");
 	}
	return opencl::detail::apply_binary_op<dimension_t, opencl::nd_range::max_dimensions>(dividend, divisor, div_rounding_up_);
}

// Note: We're not implementing a grid-to-workgroup rounding up here, since - currently -
// block_dimensions_t is the same as grid_dimensions_t.

} // namespace detail

} // namespace grid

#ifndef NDEBUG

namespace detail {

static void validate_all_dimensions_compatibility(
	optional<nd_range::dimensions_t> const& workgroup,
	optional<nd_range::dimensions_t> const& grid,
	optional<nd_range::dimensions_t> const& overall)
{
	if (workgroup and grid and overall and (*grid) * (*workgroup) != (*overall)) {
		throw std::invalid_argument("specified workgroup, grid and overall dimensions do not agree");
	}
	if (workgroup and overall) {
		auto const& workgroup_ = *workgroup;
		auto const& overall_ = *overall;
		if (workgroup_.dimensionality() != overall_.dimensionality()) {
			throw std::invalid_argument("Different dimensionality of the specified workgroup and overall dimensions");
		}
		for (nd_range::dimension_index_t d = 0; d < workgroup_.dimensionality(); d++) {
			if (workgroup_[d] > overall_[d]) {
				throw std::invalid_argument("For dimension index " + std::to_string(d)
					+ ", the specified workgroup dimension " + std::to_string(workgroup_[d]) +
					+ " exceeds overall dimension " + std::to_string(overall_[d]));
			}
		}
	}
	if (grid and overall) {
		if (grid->dimensionality() != overall->dimensionality()) {
			throw std::invalid_argument("Different dimensionality of the specified grid and overall dimensions");
		}
		for (nd_range::dimension_index_t d = 0; d < grid->dimensionality(); d++) {
			if ((*grid)[d] > (*overall)[d]) {
				throw std::invalid_argument("specified grid dimensions exceed overall dimensions");
			}
		}
	}
}

} // namespace detail

#endif // NDEBUG

inline platform_t launch_configuration_t::platform() const
{
	if (device_) { return device_->platform(); }
	throw std::runtime_error(
		"Cannot provide the platform before a kernel or a device"
		" have been associated with the launch configuration.");
}

nd_range::composite_dimensions_t
inline launch_configuration_t::get_unvalidated_composite_dimensions() const noexcept(false)
{
	// TODO: There's a bunch of redundant code in here I think
	nd_range::composite_dimensions_t result;

	if (saturate_with_active_workgroups_) {
		if (not dimensions_.workgroup) {
			throw std::logic_error(
				"The workgroup dimensions must be known to estimate how many of them one"
				" needs for saturating a device");
		}
		if (dimensions_.grid or dimensions_.overall) {
			throw std::logic_error(
				"Conflicting specifications: Grid or overall dimensions specified, but requested to "
				"saturate kernels with active workgroups");
		}

		result.workgroup = dimensions_.workgroup.value();

		// TODO: IMPLEMENT THIS
		throw std::runtime_error("Not yet supported!");

		// auto num_block_threads = static_cast<nd_range::dimension_t>(dimensions_.workgroup.value().volume());
		// auto workgroups_per_multiprocessor = kernel_->max_active_workgroups_per_multiprocessor(num_workgroup_threads, dshmem_size);
		// auto num_multiprocessors = device().get_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
		// result.grid = workgroups_per_multiprocessor * num_multiprocessors;
		// return result;
	}
	if (dimensions_.workgroup and dimensions_.grid and not dimensions_.overall) {
		result.overall = dimensions_.grid.value() * dimensions_.workgroup.value();
		result.workgroup = dimensions_.workgroup.value();
		return result;
	}
	if (dimensions_.grid and dimensions_.overall and not dimensions_.workgroup) {
		if (not divides(dimensions_.workgroup.value(), dimensions_.overall.value())) {
			throw std::invalid_argument("If nd-grid dimensions in terms of workgroups are defined - they must divide the overall dimensions");
		}
		result.workgroup = nd_range::detail::div_rounding_up(dimensions_.overall.value(), dimensions_.grid.value());
		result.overall = dimensions_.grid.value();
		return result;
	}

	if (dimensions_.grid and dimensions_.workgroup) {
		if (dimensions_.overall and dimensions_.grid.value() * dimensions_.workgroup.value() != dimensions_.overall.value()) {
	        throw std::invalid_argument("specified workgroup, grid and overall dimensions do not agree");
	    }
		result.workgroup = dimensions_.workgroup.value();
		result.overall = dimensions_.overall.value();
		return result;
	}

	if (not dimensions_.workgroup and not dimensions_.overall) {
		throw std::logic_error(
			"Neither workgroup nor overall dimensions have been specified");
	}
	result.workgroup = dimensions_.workgroup.value();
	result.overall = dimensions_.overall.value();
	if (round_up_overall_dims_) { result.round_up_overall_dimensions(); }
	return result;
}

inline nd_range::composite_dimensions_t
launch_configuration_t::get_composite_dimensions() const noexcept(false)
{
	auto result = get_unvalidated_composite_dimensions();
#ifndef NDEBUG
	validate_composite_dimensions(result);
#endif
	return result;
}

inline optional<launch_configuration_t::offset_type>
launch_configuration_t::get_offset() const noexcept(false)
{
#ifndef NDEBUG
	if (offset_) validate_offset(*offset_);
#endif
	return offset_;
}

inline kernel::launch_configuration_t launch_configuration_t::build() const
{
	kernel::launch_configuration_t result;
	result.dimensions = get_composite_dimensions();
	result.offset = get_offset();
	return result;
}

inline launch_configuration_t&
launch_configuration_t::dimensionality(nd_range::dimension_index_t dimensionality)
{
	dimensionality_constraint = dimensionality;
	return *this;
}

inline launch_configuration_t&
launch_configuration_t::clone(kernel::launch_configuration_t const& config)
{
#ifndef NDEBUG
	// TODO: Implement launch config validation
	//detail::validate(config);
	// TODO: Implement device and kernel compatibility for launch configs
	// if (device_) { detail::validate_compatibility(device(), config); }
#endif
	// dynamic_shared_memory_size_ = config.dynamic_shared_memory_size;
	dimensions(config.dimensions);
	offset(config.offset);
	return *this;
}

#ifndef NDEBUG
inline void launch_configuration_t::validate_workgroup_dimensions(nd_range::dimensions_t const& workgroup_dims) const
{
	if (device_) {
		kernel::detail::validate_workgroup_dimension_compatibility(*device_, workgroup_dims);
	}

	detail::validate_all_dimensions_compatibility(
		workgroup_dims,
		dimensions_.grid,
		dimensions_.overall);
	// TODO: Check divisibility
	if (device_) kernel::detail::validate_workgroup_dimension_compatibility(*device_, workgroup_dims);
}

inline void launch_configuration_t::validate_grid_dimensions(dimensions_t const& grid_dims) const
{
	kernel::detail::validate_grid_dimensions(grid_dims);
	if (device_) {
		kernel::detail::validate_grid_dimension_compatibility(*device_, grid_dims);
	}
	if (dimensions_.workgroup and dimensions_.overall) {
		detail::validate_all_dimensions_compatibility(
			dimensions_.workgroup, grid_dims, dimensions_.overall);
	}
	// TODO: Check divisibility
}

inline void launch_configuration_t::validate_overall_dimensions(dimensions_t const& overall_dims) const
{
	if (dimensions_.workgroup and dimensions_.grid) {
		if (dimensions_.grid.value() * dimensions_.workgroup.value() != overall_dims) {
			throw std::invalid_argument(
			"specified overall dimensions conflict with the already-specified "
			"workgroup and grid dimensions");
		}
	}
}

inline void launch_configuration_t::validate_device(device_t const& device) const
{
	if (dimensions_.workgroup or (dimensions_.grid and dimensions_.overall)) {
		auto workgroup_dims = dimensions_.workgroup ?
			dimensions_.workgroup.value() :
			get_composite_dimensions().workgroup;
		kernel::detail::validate_workgroup_dimension_compatibility(device, workgroup_dims);
	}
}

inline void launch_configuration_t::validate_composite_dimensions(
	nd_range::composite_dimensions_t const& composite_dims) const
{
	if (device_) {
		kernel::detail::validate_workgroup_dimension_compatibility(*device_, composite_dims.workgroup);
		kernel::detail::validate_grid_dimension_compatibility(*device_, composite_dims.overall);
	}

	// Is there anything to validate regarding the grid dims?

}

inline optional<nd_range::dimension_index_t> launch_configuration_t::get_unvalidated_dimensionality() const
{
	optional<nd_range::dimension_index_t> result = dimensionality_constraint;
	// Note: I would really like to have a Javascript-like operator?. here...
	if (not result and dimensionality_constraint) { result = dimensionality_constraint; }
	if (not result and dimensions_.workgroup) { result = dimensions_.workgroup->dimensionality(); }
	if (not result and dimensions_.grid) { result = dimensions_.grid->dimensionality(); }
	if (not result and dimensions_.overall) { result = dimensions_.overall->dimensionality(); }
	return result;
}

inline void launch_configuration_t::validate_dimensionality(
	nd_range::dimension_index_t dimensionality,
	const char* name) const
{
	auto check = [&](
		optional<nd_range::dimension_index_t> element_dims,
		const char* e_name)
	{
		if (not element_dims or dimensionality == element_dims.value()) { return; }
		std::ostringstream oss;
		oss << "Dimensionality "
			<< (name ? std::string{" of "} + name : "")
			<< *element_dims << " is incompatible with " << e_name << " dimensionality ("
			<< *element_dims << ")";
		throw std::invalid_argument(oss.str());
	};
	check(dimensionality_constraint, "constrained common");
	if (dimensions_.workgroup) check(dimensions_.workgroup->dimensionality(), "workgroup");
	if (dimensions_.grid)      check(dimensions_.grid->dimensionality(), "grid");
	if (dimensions_.overall)   check(dimensions_.overall->dimensionality(), "overall");
}

inline void launch_configuration_t::validate_dimensionality_of(
	dimensions_t const& dimensions, const char* dimensions_name) const
{
	validate_dimensionality(dimensions.dimensionality(), dimensions_name);
}

inline optional<nd_range::dimension_index_t> launch_configuration_t::get_dimensionality() const
{
	auto result = get_unvalidated_dimensionality();
#ifndef NDEBUG
	if (result) { validate_dimensionality(*result); }
#endif
	return result;
}

inline optional<nd_range::dimensions_t> launch_configuration_t::get_unverified_overall_dimensions() const
{
	if (dimensions_.overall) { return dimensions_.overall; }
	if (dimensions_.workgroup and dimensions_.grid) {
		return *dimensions_.workgroup * *dimensions_.grid;
	}
	return nullopt;
}

inline void launch_configuration_t::validate_offset(offset_type const& offset) const
{
	validate_dimensionality_of(offset);
	auto overall = get_unverified_overall_dimensions();
	if (overall) {
		// At this point we are already certain that offset and overall have the same dimensionality
		for (nd_range::dimension_index_t i = 0; i < overall->dimensionality(); ++i) {
			if (offset[i] >= (*overall)[i]) {
				throw std::invalid_argument(
					"Offset " + std::to_string(offset[i]) + "exceeds overall dimension "
					+ std::to_string((*overall)[i]) + ", in coordinate + "
					+ std::to_string(i));
			}
		}
	}
}

#endif // ifndef NDEBUG

inline launch_configuration_t&
launch_configuration_t::dimensions(nd_range::composite_dimensions_t const& composite_dims)
{
#ifndef NDEBUG
	validate_composite_dimensions(composite_dims);
#endif
	dimensions_.overall = composite_dims.overall;
	dimensions_.grid = nullopt;
	dimensions_.workgroup = composite_dims.workgroup;
	return *this;
}

inline launch_configuration_t&
launch_configuration_t::workgroup_dimensions(nd_range::dimensions_t const& dims)
{
#ifndef NDEBUG
	validate_workgroup_dimensions(dims);
#endif
	dimensions_.workgroup = dims;
	if (dimensions_.overall) {
		// TODO: Make sure we're consistent about not "forcing" the dims
		// to agree with existing grid+overall dims. Actually, it's more complicated
		// than that, since we need to choose which of the two to nullify/overwrite
		// due to the new setting. In short, serious re-thinking is needed. Maybe just
		// always keep workgroup & overall , and have the grid dims be always derivative?
		dimensions_.grid = nullopt;
	}
	return *this;
}

inline launch_configuration_t&
launch_configuration_t::workgroup_dimensions(dimension_t x, dimension_t y, dimension_t z)
{
	return workgroup_dimensions(nd_range::dimensions_t{x, y, z});
}

inline launch_configuration_t& launch_configuration_t::workgroup_size(dimension_t size)
{
	return workgroup_dimensions(size, 1lu, 1lu);
}

inline launch_configuration_t& launch_configuration_t::use_maximum_linear_workgroup()
{
	dimension_t max_size;
	if (device_) {
		max_size = device().capabilities().maximum_workgroup_size();
	}
	else {
		throw std::logic_error("Request to use the maximum-size linear workgroup, with no device or kernel specified");
	}
	auto workgroup_dims = nd_range::dimensions_t { max_size, 1lu, 1lu };

	if (dimensions_.grid and dimensions_.overall) {
		dimensions_.overall = nullopt;
	}
	dimensions_.workgroup = workgroup_dims;
	return *this;
}

inline launch_configuration_t& launch_configuration_t::grid_dimensions(dimensions_t const& dims)
{
#ifndef NDEBUG
	validate_grid_dimensions(dims);
#endif
	if (dimensions_.workgroup) {
		dimensions_.overall = nullopt;
	}
	dimensions_.grid = dims;
	saturate_with_active_workgroups_ = false;
	return *this;
}

inline launch_configuration_t&
launch_configuration_t::grid_dimensions(dimension_t x, dimension_t y, dimension_t z)
{
	return grid_dimensions(dimensions_t{x, y, z});
}

inline launch_configuration_t& launch_configuration_t::grid_size(size_t size) {
#ifndef NDEBUG
	if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
		throw std::invalid_argument("Specified (1-dimensional) grid size " + std::to_string(size)
			+ "in workgroups exceeds " + std::to_string(std::numeric_limits<int>::max())
			+ " , the maximum supported number of workgroups");
	}
#endif
	return grid_dimensions(size, 1, 1);
}
inline launch_configuration_t& launch_configuration_t::num_workgroups(size_t size) { return grid_size(size); }

inline launch_configuration_t& launch_configuration_t::overall_dimensions(dimensions_t const& dims)
{
#ifndef NDEBUG
	validate_overall_dimensions(dims);
#endif
	dimensions_.overall = dims;
	saturate_with_active_workgroups_ = false;
	return *this;
}

inline launch_configuration_t&
launch_configuration_t::overall_dimensions(dimension_t x, dimension_t y, dimension_t z)
{
	return overall_dimensions(dimensions_t{x, y, z});
}

inline launch_configuration_t& launch_configuration_t::overall_size(size_t size)
{
	static_assert(std::is_same<dimension_t, size_t>::value, "Unexpected type discrepancy");
	return overall_dimensions({size, 1lu, 1lu});
}

inline launch_configuration_t& launch_configuration_t::device(
	const platform::handle_t platform_handle,
	const device::handle_t device_handle)
{
	device_ = device::wrap(platform_handle, device_handle);
	return *this;
}

inline launch_configuration_t& launch_configuration_t::device(const device_t& device)
{
	return this->device(device.platform_handle(), device.handle());
}

inline launch_configuration_t& launch_configuration_t::saturate_with_active_workgroups()
{
	if (not dimensions_.workgroup) {
		throw std::logic_error("The workgroup dimensions must be known to determine how many of them one needs for saturating a device");
	}
	dimensions_.grid = nullopt;
	dimensions_.overall = nullopt;
	saturate_with_active_workgroups_ = true;
	return *this;
}

inline launch_configuration_t&
launch_configuration_t::offset(optional<offset_type> const& offset)
{
	offset_ = offset;
#ifndef NDEBUG
	if (offset) validate_offset(*offset);
#endif
	return *this;
}

inline launch_configuration_t&
launch_configuration_t::round_up_overall_dims() noexcept
{
	round_up_overall_dims_ = true;
	return *this;
}

inline launch_configuration_t&
launch_configuration_t::dont_round_up_overall_dims() noexcept
{
	round_up_overall_dims_ = false;
	return *this;
}

} // namespace builders
} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_BUILDERS_LAUNCH_CONFIG_HPP_
