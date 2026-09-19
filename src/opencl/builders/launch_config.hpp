#ifndef OPENCL_WRAPPERS_BUILDERS_LAUNCH_CONFIG_HPP_
#define OPENCL_WRAPPERS_BUILDERS_LAUNCH_CONFIG_HPP_

#include "../types.hpp"

namespace opencl {

namespace builders {

namespace nd_range {

using opencl::nd_range::dimension_index_t;
using opencl::nd_range::dimensions_t;
using opencl::nd_range::offset_t;
using opencl::nd_range::composite_dimensions_t;
using opencl::nd_range::workgroup_dimensions_t;

} // namespace nd_range

/**
 * A convenience class for gradually constructing a @ref launch_configuration_t instance,
 * as per the "builder pattern".
 *
 * @note with a constructed class, repeatedly invoke a member function to add settings -
 *     the result of the application is the same builder, so you can combine all settings
 *     into a single expression, then finally invoke @ref launch_configuration_t::build to
 *     finalize the build and obtain the @ref launch_configuration_t object.
 *
 * @note This class will perform some validation of the settings you make - but do not
 *     assume it guarantees validity. Also, the validations may be either eager (when making
 *     a setting) or lazy (when finally building the launch configuration).
 */
class launch_configuration_t {
public:
	using self_type = launch_configuration_t;
	using dimensions_t = nd_range::dimensions_t;
	using offset_type = nd_range::offset_t;

protected:

	optional<nd_range::dimension_index_t> dimensionality_constraint;
	struct builder_dims_t {
		optional<nd_range::dimensions_t> workgroup;
		optional<dimensions_t> grid;
		optional<nd_range::dimensions_t> overall;
	} dimensions_;
	optional<offset_type> offset_;


	// TODO: Is kernel a base class for more kinds of kernels?
	optional<device_t> device_;

	bool saturate_with_active_workgroups_ { false };
	bool round_up_overall_dims_ { false };

public:

	platform_t platform() const;
	device_t device() const { return *device_; }
	device_t kernel() const { return *device_; }
	// TODO: Set kernel and device

protected:
	// Note: No support for dynamic shared memory in launch_configuration_t yet, so
	// the builder can't be concerned with it

	nd_range::composite_dimensions_t get_unvalidated_composite_dimensions() const noexcept(false);
	nd_range::composite_dimensions_t get_composite_dimensions() const noexcept(false);
	optional<offset_type> get_offset() const noexcept(false);

public:
	/// Use the information specified to the builder (and defaults for the unspecified
	/// information) to finalize the construction of a kernel launch configuration,
	/// which can then be passed along with the kernel to a kernel-launching function,
	/// e.g. the standalone @ref kernel::launch or the stream command @ref stream_t::enqueue_t::kernel_launch
	kernel::launch_configuration_t build() const;
	launch_configuration_t& dimensionality(nd_range::dimension_index_t dimensionality);
	launch_configuration_t& clone(kernel::launch_configuration_t const& config);
#ifndef NDEBUG
	void validate_workgroup_dimensions(nd_range::dimensions_t const& workgroup_dims) const;
	void validate_grid_dimensions(dimensions_t const& grid_dims) const;
	void validate_overall_dimensions(dimensions_t const& overall_dims) const;
	void validate_device(device_t const& device) const;
	void validate_composite_dimensions(nd_range::composite_dimensions_t const& composite_dims) const;
	optional<nd_range::dimension_index_t> get_unvalidated_dimensionality() const;
	void validate_dimensionality(nd_range::dimension_index_t dimensionality, const char* name = nullptr) const;
	void validate_dimensionality_of(dimensions_t const& dimensions, const char* dimensions_name = nullptr) const;
	optional<nd_range::dimension_index_t> get_dimensionality() const;
	optional<nd_range::dimensions_t> get_unverified_overall_dimensions() const;
	void validate_offset(offset_type const& offset) const;
#endif // ifndef NDEBUG

	launch_configuration_t& dimensions(nd_range::composite_dimensions_t const& composite_dims);
	launch_configuration_t& workgroup_dimensions(nd_range::dimensions_t const& dims);
	/// Set the dimensions for each workgroup in the intended kernel launch grid
	launch_configuration_t& workgroup_dimensions(
		dimension_t x,
		dimension_t y = 1lu,
		dimension_t z = 1lu);

	/// Set the workgroup in the intended kernel launch grid to be uni-dimensional
	/// with a specified size
	launch_configuration_t& workgroup_size(dimension_t size);

	/**
	 * Set the intended kernel launch grid to have 1D workgroups, of the maximum
	 * length possible given the information specified to the builder.
	 *
	 * @note This will fail if neither a kernel nor a device have been chosen
	 * for the launch.
	 */
	launch_configuration_t& use_maximum_linear_workgroup();

	/// Set the dimension of the grid for the intended kernel launch, in terms
	/// of workgroups
	///@{
	launch_configuration_t& grid_dimensions(dimensions_t const& dims);

	///@}
	launch_configuration_t& grid_dimensions(
		dimension_t x,
		dimension_t y = 1,
		dimension_t z = 1);

	/// Set the grid for the intended launch to be one-dimensional, with a specified number
	/// of workgroups
	///@{
	launch_configuration_t& grid_size(size_t size);
	launch_configuration_t& num_workgroups(size_t size);


	/// Set the overall number of _threads_, in each dimension, of all workgroups
	/// in the grid of the intended kernel launch
	///@{
	launch_configuration_t& overall_dimensions(dimensions_t const& dims);
	launch_configuration_t& overall_dimensions(dimension_t x, dimension_t y = 1, dimension_t z = 1);
	///@}

	/// Set the intended launch grid to be linear, with a specified overall number of _threads_
	/// over all (1D) workgroups in the grid
	launch_configuration_t& overall_size(size_t size);

	/**
	 * Indicate that the intended kernel launch would occur on (some stream in
	 * some context on) the specified device. Such an indication provides this
	 * object with some information regarding ranges of possible values for
	 * certain parameters (e.g. shared memory size, dimensions).
	 *
	 * @note Do not call both this and the @ref kernel() method; prefer just that one.
	 */
	///@{
	launch_configuration_t& device(
		const platform::handle_t platform_handle,
		const device::handle_t device_handle);

	launch_configuration_t& device(const device_t& device);
	///@}

	/**
	 * @brief This will use information about the kernel, the already-set workgroup size,
	 * and the device to create a unidimensional grid of workgroups to exactly saturate
	 * the CUDA device's capacity for simultaneous active workgroups.
	 *
	 * @note This will _not_ set the workgroup size - unlike {@ref min_params_for_max_occupancy()}.
	 */
	launch_configuration_t& saturate_with_active_workgroups();
	launch_configuration_t& offset(optional<offset_type> const& offset);
	launch_configuration_t& clear_offset() noexcept { return offset(nullopt); }
	launch_configuration_t& no_offset() noexcept { return clear_offset(); }
	launch_configuration_t& round_up_overall_dims() noexcept;
	launch_configuration_t& dont_round_up_overall_dims() noexcept;
}; // launch_configuration_t

/// A slightly shorter-named construction idiom for @ref launch_configuration_t
inline launch_configuration_t launch_config() { return {}; }

} // namespace builders

} // namespace opencl

#endif // OPENCL_WRAPPERS_BUILDERS_LAUNCH_CONFIG_HPP_
