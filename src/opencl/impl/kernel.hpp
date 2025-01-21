
#ifndef OPENCL_WRAPPERS_IMPL_KERNEL_HPP_
#define OPENCL_WRAPPERS_IMPL_KERNEL_HPP_

#include "../device.hpp"
#include "../kernel.hpp"
#include "../program.hpp"
#include "build_info.hpp"
#include "../identify.hpp"
#include "info.hpp"
#include "ownership_token_t.hpp"

#include <array>

namespace opencl {

inline char const* info::traits_t<info::kernel>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case CL_KERNEL_FUNCTION_NAME:   return "name";
    case CL_KERNEL_NUM_ARGS:        return "number of parameters";
    case CL_KERNEL_REFERENCE_COUNT: return "reference count";
    case CL_KERNEL_CONTEXT:         return "context";
    case CL_KERNEL_PROGRAM:         return "program";
#ifdef CL_VERSION_1_2
    case CL_KERNEL_ATTRIBUTES:      return "other various attributes"; // Now that's vague...
#endif
    default: return nullptr;
    }
}

inline char const* info::traits_t<info::kernel_workgroup>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case CL_KERNEL_WORK_GROUP_SIZE:         return "maximum supported overall workgroup size";
    case CL_KERNEL_COMPILE_WORK_GROUP_SIZE: return "supported workgroup dimensions specified in the kernel source";
    case CL_KERNEL_LOCAL_MEM_SIZE:          return "local memory size per workgroup";
    case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
                                            return "preferred multiple for the workgroup size";
    case CL_KERNEL_PRIVATE_MEM_SIZE:        return "minimum amount of private memory size per workgroup";
#ifdef CL_VERSION_2_1
    case CL_KERNEL_GLOBAL_WORK_SIZE:        return "global dimensions (work size)";
#endif
    default: return nullptr;
    }
}

#ifdef CL_VERSION_1_2
inline char const* info::traits_t<info::kernel_parameter>::attribute_name(attribute_id_type attribute) noexcept
{
    switch (attribute) {
    case CL_KERNEL_ARG_ADDRESS_QUALIFIER:   return "address qualifier";
    case CL_KERNEL_ARG_ACCESS_QUALIFIER:    return "access qualifier";
    case CL_KERNEL_ARG_TYPE_NAME:           return "type name";
    case CL_KERNEL_ARG_TYPE_QUALIFIER:      return "type qualifier";
    case CL_KERNEL_ARG_NAME:                return "name";
    default: return nullptr;
    }
}
#endif

namespace detail {

OCLW_DEFINE_HANDLE_TRAITS("kernel", kernel::handle_t, clReleaseKernel, clRetainKernel);

} // namespace detail

namespace kernel {

inline kernel_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    program::handle_t   program_handle,
    handle_t            kernel_handle,
    bool                is_owning) noexcept
{
    return { platform_handle, context_handle, program_handle, kernel_handle, is_owning };
}

inline nd_range::dimensions_t nd_range_info_t::global_dimensions() const
{
    nd_range::dimensions_t result;
    info::get_array<CL_KERNEL_GLOBAL_WORK_SIZE>({kernel_handle, device_handle}, {result.data(), result.size()});
    return result;
}

inline nd_range::dimensions_t nd_range_info_t::source_workgroup_dimensions_constraint() const
{
    std::array<dimension_t, 3> result; // Yes, we always get 3 dimensions
    info::get_array<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>({kernel_handle, device_handle}, {result.data(), result.size()});
    return opencl::detail::clip_zeros<nd_range::max_dimensions>(result);
}

inline nd_range::dimensions_t nd_range_info_t::workgroup_dimensions() const
{
    std::array<dimension_t, 3> result; // Yes, we always get 3 dimensions
    info::get_array<CL_KERNEL_WORK_GROUP_SIZE>({kernel_handle, device_handle},  {result.data(), result.size()});
    return opencl::detail::clip_zeros<nd_range::max_dimensions>(result);
}

inline size_t nd_range_info_t::private_memory_size() const
{
    return info::get_scalar<CL_KERNEL_WORK_GROUP_SIZE>({kernel_handle, device_handle});
}

inline size_t nd_range_info_t::local_memory_size() const
{
    return info::get_scalar<CL_KERNEL_LOCAL_MEM_SIZE>({kernel_handle, device_handle});
}

inline size_t nd_range_info_t::preferred_workgroup_size_multiple() const
{
    return info::get_scalar<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>({kernel_handle, device_handle});
}

#if CL_VERSION_1_2
namespace detail {

inline parameter::type_qualifiers_t make_type_qualifiers(cl_kernel_arg_type_qualifier qualifiers_bitfield)
{
    parameter::type_qualifiers_t result;
    result.const_    = qualifiers_bitfield & CL_KERNEL_ARG_TYPE_CONST;
    result.restrict_ = qualifiers_bitfield & CL_KERNEL_ARG_TYPE_RESTRICT;
    result.volatile_ = qualifiers_bitfield & CL_KERNEL_ARG_TYPE_VOLATILE;
#if CL_VERSION_2_0
    result.pipe      = qualifiers_bitfield & CL_KERNEL_ARG_TYPE_PIPE;
#endif
    return result;
}

} // namespace detail
#endif // CL_VERSION_1_2

} // namespace kernel

inline std::string kernel_t::name() const
{
    return info::get_string<CL_KERNEL_FUNCTION_NAME>(handle_);
}

inline kernel::arity_t kernel_t::arity() const
{
    return info::get_scalar<CL_KERNEL_NUM_ARGS>(handle_);
}

#if CL_VERSION_1_2
inline std::string kernel_t::declared_attributes() const
{
    return info::get_string<CL_KERNEL_ATTRIBUTES>(handle_);
}

inline dynarray<kernel::parameter::info_t> kernel_t::parameter_info() const
{
    auto single_arg_info_getter = [&](size_t index) { return this->parameter_info(index); };
    return generate_dynarray<kernel::parameter::info_t>(num_parameters(), single_arg_info_getter);
}

inline kernel::parameter::info_t kernel_t::parameter_info(kernel::parameter::index_t parameter_index) const
{
    kernel::parameter::info_t result;
    auto handle_pair = info::detail::kernel_param_index_pair_t{handle_, parameter_index};
    result.address_space = static_cast<kernel::parameter::address_space_t>(info::get_scalar<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(handle_pair));
    result.access_qualifier = static_cast<access_kind_t>(info::get_scalar<CL_KERNEL_ARG_ACCESS_QUALIFIER>(handle_pair));
    cl_kernel_arg_type_qualifier qualifiers_bitfield =  info::get_scalar<CL_KERNEL_ARG_TYPE_QUALIFIER>(handle_pair);
    result.type_qualifiers = kernel::detail::make_type_qualifiers(qualifiers_bitfield);
    result.type_name = info::get_string<CL_KERNEL_ARG_TYPE_NAME>(handle_pair);
    result.name = info::get_string<CL_KERNEL_ARG_NAME>(handle_pair);
    return result;
}
#endif

inline kernel::nd_range_info_t kernel_t::nd_range_info_for(device_t const& device) const
{
    return kernel::nd_range_info_t { handle_, device.handle() };
}

template<typename... Ts>
event_t kernel_t::enqueue_launch(queue_t const &queue, kernel::launch_configuration_t const& launch_config, Ts &&... args)
{
    return launch(queue, *this, launch_config, std::forward<Ts>(args)...);
}

inline platform_t kernel_t::platform() const noexcept
{
    return platform::wrap(platform_handle_);
}

inline context_t kernel_t::context() const noexcept
{
    return context::wrap(platform_handle_, {}, context_handle_, is_not_owning);
}

inline program::with_kernels_t kernel_t::program() const noexcept
{
    return program::detail::wrap_with_kernels(platform_handle_, context_handle_, program_handle_, nullopt, is_not_owning);
}

namespace kernel {

inline kernel_t instantiate(program::with_kernels_t const& program, char const* kernel_name)
{
    status_t status;
    auto kernel_handle = clCreateKernel(program.handle(), kernel_name, &status);
    if (status == CL_INVALID_PROGRAM_EXECUTABLE) {
        auto failed_target = program::detail::first_build_failure_target(program);
        if (failed_target) {
            throw std::invalid_argument("Attempt to create a kernel from an OpenCL program which has failed to "
                "build for " + opencl::detail::identify(*failed_target) + " (and possibly others)");
        }
    }
    throw_if_error_lazy(status, "clCreateKernel", "Creating " + std::string(kernel_name) + " in "
        + opencl::detail::identify(program));
    return wrap(program.platform_handle(), program.context_handle(), program.handle(), kernel_handle, is_owning);
}

namespace detail {

inline dynarray<kernel::handle_t> create_handles(program::with_kernels_t const& program, size_t num_kernels)
{
    cl_uint num_created { 0 };
    auto new_kernel_handles = make_dynarray<kernel::handle_t>(num_kernels);
    auto status = clCreateKernelsInProgram(program.handle(), num_kernels, new_kernel_handles.data(), &num_created);
    throw_if_error_lazy(status, "clCreateKernelsInProgram", "Creating " + std::to_string(num_kernels)
        + "kernels in " + opencl::detail::identify(program) + "; only " + std::to_string(num_created)
        + " of " + std::to_string(num_kernels) + " created");
    if (num_created != num_kernels) {
        throw std::runtime_error("Creating " + std::to_string(num_kernels)
            + "kernels in " + opencl::detail::identify(program) + ": only " + std::to_string(num_created)
            + " of " + std::to_string(num_kernels) + " created");
    }
    return new_kernel_handles;
}

}

inline dynarray<kernel_t> instantiate_all(program::with_kernels_t const& program)
{
    size_t num_kernels = program.num_kernels();
    auto new_kernel_handles = detail::create_handles(program, num_kernels);
    auto generator = [&](size_t i) -> kernel_t {
        return wrap(
            program.platform_handle(), program.context_handle(),
            program.handle(), new_kernel_handles[i], is_owning);
    };
    return generate_dynarray<kernel_t>(num_kernels, generator);
}

inline kernel_t clone(kernel_t const& kernel)
{
    cl_int status;
    auto result_handle = clCloneKernel(kernel.handle(), &status);
    throw_if_error_lazy(status, "clCloneKernel", "Cloning " + opencl::detail::identify(kernel));
    return wrap(kernel.platform_handle(), kernel.context_handle(), kernel.program_handle(), result_handle, is_owning);
}

} // namespace kernel

#if CL_VERSION_2_0
inline void kernel_t::set_acessible_svm_regions(span<void const*> region_ptrs) const
{
    auto status = clSetKernelExecInfo(handle_, CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT,
        sizeof(memory::device_address_t) * region_ptrs.size(), region_ptrs.data());
    throw_if_error_lazy(status, "clSetKernelExecInfo", "Trying to set the device-side pointers accessible by "
        + opencl::detail::identify(*this) + " (in addition to any device-side pointers set as arguments");

}

inline void kernel_t::set_acessible_svm_regions(span<memory::shared_virtual::region_t> regions) const
{
    static auto get_ptr = [](memory::shared_virtual::region_t const& region) { return region.data(); };
    auto ptrs = to_dynarray<void const*>(regions, get_ptr);
    return set_acessible_svm_regions(ptrs);
}

inline void kernel_t::set_fine_grain_svm_access(bool allowed) const
{
    cl_bool allowed_ = allowed;
    auto status = clSetKernelExecInfo(handle_, CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM,
        sizeof(cl_bool), &allowed_);
    throw_if_error_lazy(status, "clSetKernelExecInfo", "Trying to set the device-side pointers accessible by "
        + opencl::detail::identify(*this) + " (in addition to any device-side pointers set as arguments");
}

#ifdef cl_ext_buffer_device_address
// TODO: Should I require these to be regions as well?
template <typename T>
void kernel_t::set_acessible_device_ptrs(span<T*> const& device_ptrs) const
{
    auto status = clSetKernelExecInfo(handle_, CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT,
        sizeof(memory::device_address_t) * device_ptrs.size(), device_ptrs.data());
    throw_if_error_lazy(status, "clSetKernelExecInfo", "Trying to set the device-side pointers accessible by "
        + opencl::detail::identify(*this) + " (in addition to any device-side pointers set as arguments");
}
#endif
#endif


} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_KERNEL_HPP_
