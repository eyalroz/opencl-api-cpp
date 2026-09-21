#ifndef OPENCL_WRAPPERS_KERNEL_HPP_
#define OPENCL_WRAPPERS_KERNEL_HPP_

#include "types.hpp"
#include "context.hpp"
#include "program.hpp"

namespace opencl {

namespace kernel {

kernel_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    program::handle_t   program_handle,
    handle_t            kernel_handle,
    bool                is_owning) noexcept;

kernel_t instantiate(program::with_kernels_t const& program, char const* kernel_name);
dynarray<kernel_t> instantiate_all(program::with_kernels_t const& program);

struct nd_range_info_t {
    kernel::handle_t kernel_handle;
    device::handle_t device_handle;

    nd_range::dimensions_t global_dimensions() const;
    nd_range::dimensions_t workgroup_dimensions() const;
    nd_range::dimensions_t source_workgroup_dimensions_constraint() const;
    size_t private_memory_size() const;
    size_t local_memory_size() const;
    size_t preferred_workgroup_size_multiple() const;
};

kernel_t clone(kernel_t const& kernel);

#if CL_VERSION_1_2

namespace parameter {

enum address_space_t {
    global   = CL_KERNEL_ARG_ADDRESS_GLOBAL,
    local    = CL_KERNEL_ARG_ADDRESS_LOCAL,
    constant = CL_KERNEL_ARG_ADDRESS_CONSTANT,
    private_ = CL_KERNEL_ARG_ADDRESS_PRIVATE,
};

struct type_qualifiers_t {
    bool const_;
    bool restrict_;
    bool volatile_;
#if CL_VERSION_2_0
    bool pipe;
#endif
};

/// The aggregate of all information we can obtain via clGetKernelArgInfo
struct info_t {
    address_space_t address_space;
    access_kind_t access_qualifier;
    type_qualifiers_t type_qualifiers;
    std::string name;
    std::string type_name;
};

} // namespace argument

#endif // CL_VERSION_1_2

} // namespace kernel

// TODO: double-check reference counting, move impls away
class kernel_t  {
public:
    using handle_type = kernel::handle_t;

protected:
    platform::handle_t platform_handle_;
    context::handle_t context_handle_;
    program::handle_t program_handle_;
    handle_type handle_;
    detail::handle_ownership_token_t<kernel_t> ownership_token_;
    // and note that there is no association with a specific device!

public:
    platform::handle_t platform_handle() const noexcept { return platform_handle_; }
    context::handle_t context_handle() const noexcept { return context_handle_; }
    program::handle_t program_handle() const noexcept { return program_handle_; }
    kernel::handle_t handle() const noexcept { return handle_; }
    bool owning() const noexcept { return ownership_token_.owning(); }

    platform_t platform() const noexcept;
    context_t context() const noexcept;
    program::with_kernels_t program() const noexcept;

protected:
    kernel_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        program::handle_t program_handle,
        kernel::handle_t handle,
        bool owning) noexcept
    :
        platform_handle_(platform_handle), context_handle_(context_handle), program_handle_(program_handle),
        handle_(handle), ownership_token_(owning, handle) {}


public:
    friend kernel_t kernel::wrap(platform::handle_t, context::handle_t, program::handle_t, kernel::handle_t, bool) noexcept;

    // TODO: Should I delete instead of move the rvalue-reference methods?

    // get-info-based methods
    std::string name() const;
    kernel::arity_t arity() const;
    kernel::arity_t num_parameters() const { return arity(); }
#if CL_VERSION_1_2
    std::string declared_attributes() const; // the __attribute__((...)) stuff in the source
    dynarray<kernel::parameter::info_t> parameter_info() const;
    kernel::parameter::info_t parameter_info(kernel::parameter::index_t parameter_index) const;
#endif

    kernel::nd_range_info_t nd_range_info_for(device_t const& device) const;

    // setExecInfo-based methods

#if CL_VERSION_2_0
    void set_acessible_svm_regions(span<memory::shared_virtual::region_t> regions) const;
    void set_acessible_svm_regions(span<void const*> region_ptrs) const;
    void set_fine_grain_svm_access(bool allowed) const;
#ifdef cl_ext_buffer_device_address
    // TODO: Should I require these to be regions as well?
    template <typename T>
    void set_acessible_device_ptrs(span<T*> const& device_ptrs) const;
#endif
#endif

    template <typename... Ts>
    event_t enqueue_launch(queue_t const& queue, kernel::launch_configuration_t const& launch_config, Ts&&... args);

    kernel_t clone() const { return kernel::clone(*this); }
}; // class kernel_t

bool operator==(kernel_t const& lhs, kernel_t const& rhs) noexcept;
inline bool operator!=(kernel_t const& lhs, kernel_t const& rhs) noexcept { return not (lhs == rhs); }

} // namespace opencl

#endif // OPENCL_WRAPPERS_KERNEL_HPP_
