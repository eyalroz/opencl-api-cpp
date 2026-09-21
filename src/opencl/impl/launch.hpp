#ifndef OPENCL_WRAPPERS_IMPL_LAUNCH_HPP_
#define OPENCL_WRAPPERS_IMPL_LAUNCH_HPP_

#include "../identify.hpp"
#include "../queue.hpp"

namespace opencl {

namespace kernel {
namespace detail {

using parameter_index_t = cl_uint;

enum class parameter_kind_t {
    raw_mem_object,
    wrapped_mem_object, // e.g. buffer or image
    svm_region,
    local_memory_allowance,
#ifdef cl_ext_buffer_device_address
    device_ptr,
#endif
    other
};

template<parameter_kind_t Kind>
using lifted_kind_t = ::std::integral_constant<parameter_kind_t, Kind>;

/**
 * @brief adapt a type to be usable as a kernel parameter.
 *
 * OpenCL kernels can't really accept just any parameter type a C++ function may
 * accept (even if they're OpenCL C++ kernels, and certainly not if they're OpenCL
 * C kernels). Specifically: No references, arrays decay (IIANM) and functions pass
 * by address. However - not all "decaying" of `::std::decay` is necessary. The
 * relevant transformation can be effected by this type-trait struct.
 */
template<typename P>
struct kernel_argument_decay {
private:
    using U = typename ::std::remove_reference<P>::type;
public:
    using type = typename ::std::conditional<
        ::std::is_array<U>::value,
        typename ::std::remove_extent<U>::type*,
        typename ::std::conditional<
            ::std::is_function<U>::value,
            typename ::std::add_pointer<U>::type,
            U
        >::type
    >::type;
};

template<typename P>
using kernel_argument_decay_t = typename kernel_argument_decay<P>::type;

/**
 * @brief A trait characterizing those types which can be used as kernel parameters
 *
 * When passing arguments to a kernel function, their representation is copied from the host
 * to the GPU device; so their type must allow for a construction by means of such copying;
 * this is essentially the @ref std::is_trivially_copy_constructible trait.
 * which serves to identify types that can be safely passed to kernels (well, not really;
 * this needs to be tightened further for OpenCL C).
 *
 * @tparam T The prospective kernel parameter type
 * @note The trait is based on the CUDA C++ Programming Guide's requirements from
 *       kernel parameter types, mostly (see
 *       @url https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-function-argument-processing );
 *       There is also a requirement for supporting destructor calls before the kernel
 *       execution is actually concluded - which is all but impossible to check via a trait;
 *       so it is up to users of the library to either exercise care, or otherwise specialize
 *       the trait for known-to-be-problematic types, precluding their use. Finally, this
 *       whole trait may need to be redone for the more common denominator of OpenCL C
 *       parameters.
 */
template <typename T>
struct is_valid_kernel_argument {
    // TODO: Should a pointer a pointer be a valid kernel argument?
    enum { value =
        std::is_trivially_copyable<T>::value or
        std::is_base_of<memory::object_t, T>::value  or
        std::is_base_of<memory::shared_virtual::region_t, T>::value or
        std::is_base_of<memory::shared_virtual::const_region_t, T>::value};
};

/**
 * @brief A convenience type using the @ref is_valid_kernel_argument trait struct
 */
template <typename T>
using is_valid_kernel_argument_t = typename is_valid_kernel_argument<T>::type;

inline void set_argument(
    lifted_kind_t<parameter_kind_t::raw_mem_object>,
    kernel_t const & kernel,
    parameter_index_t index,
    cl_mem argument)
{
    // Note: No support for shared memory and for SVM ('managed' memory) for now
    auto status = clSetKernelArg(kernel.handle(), index, sizeof(cl_mem), &argument);
    throw_if_error_lazy(status, "clSetKernelArg", "Setting argument " + std::to_string(index+1)
        + " (1-based) of " + opencl::detail::identify(kernel) + " to " + opencl::detail::identify(argument));
}

template <typename T>
void set_argument(
    lifted_kind_t<parameter_kind_t::wrapped_mem_object>,
    kernel_t const & kernel,
    parameter_index_t index,
    T&& argument)
{
    set_argument(lifted_kind_t<parameter_kind_t::raw_mem_object>{}, kernel, index, argument.handle());
}

template <typename T>
void set_argument(
    lifted_kind_t<parameter_kind_t::svm_region>,
    kernel_t const & kernel,
    parameter_index_t index,
    T&& argument)
{
    auto status = clSetKernelArgSVMPointer(kernel.handle(), index,  &argument.data());
    throw_if_error_lazy(status, "clSetKernelArg", "Setting argument " + std::to_string(index+1)
        + " (1-based) of " + opencl::detail::identify(kernel) + " to " + opencl::detail::identify(argument));
}

inline void set_argument(
    lifted_kind_t<parameter_kind_t::local_memory_allowance>,
    kernel_t const & kernel,
    parameter_index_t index,
    memory::local::allowance_t argument)
{
    auto status = clSetKernelArgSVMPointer(kernel.handle(), index,  &argument.size);
    throw_if_error_lazy(status, "clSetKernelArg", "Setting a local memory allowance via parameter "
        + std::to_string(index + 1) + " (1-based) of " + opencl::detail::identify(kernel) + " to "
        + opencl::detail::identify(argument));
}

#ifdef cl_ext_buffer_device_address

template <typename T>
void set_argument(
    lifted_kind_t<parameter_kind_t::device_ptr>,
    kernel_t const & kernel,
    parameter_index_t index,
    T* argument)
{
    auto status = clSetKernelArgDevicePointerEXT(kernel.handle(), index,  static_cast<cl_mem_device_address_ext>(argument));
    throw_if_error_lazy(status, "clSetKernelArgDevicePointerEXT", "Setting a device address argument to parameter "
        + std::to_string(index + 1) + " (1-based) of " + opencl::detail::identify(kernel) + " to "
        + opencl::detail::identify(argument));
}

#endif // cl_ext_buffer_device_address

template <typename T>
void set_argument(
    lifted_kind_t<parameter_kind_t::other>,
    kernel_t const & kernel,
    parameter_index_t index,
    T&& argument)
{
    // Note: No support for dynamic local memory for now
    static_assert(is_valid_kernel_argument<kernel_argument_decay_t<T>>::value,
        "Attempt to use a value of an invalid type as a kernel argument");
    // TODO: Actually, we should probably be a lot more strict. for example, we should probably bar pointers.
    auto status = clSetKernelArg(kernel.handle(), index, sizeof(T), &argument);
    throw_if_error_lazy(status, "clSetKernelArg", "Failed setting kernel argument at index " + std::to_string(index));
}

// TODO: With C++14, get rid of this in favor of a simple switch statement within the kind_of() function
template <typename T>
using kind_of_helper_ = std::integral_constant<parameter_kind_t,
    std::is_same<T, cl_mem>::value ? parameter_kind_t::raw_mem_object :
    std::is_base_of<memory::object_t, T>::value ? parameter_kind_t::wrapped_mem_object :
    (std::is_same<T, memory::shared_virtual::region_t>::value or
    std::is_same<T, memory::shared_virtual::const_region_t>::value) ? parameter_kind_t::svm_region :
    std::is_same<T, memory::local::allowance_t>::value ? parameter_kind_t::local_memory_allowance :
    parameter_kind_t::other>;

template <typename T>
using kind_of_helper = kind_of_helper_<opencl::detail::remove_cvref_t<T>>;

template <typename T>
constexpr parameter_kind_t kind_of() { return kind_of_helper<T>::value; }

template <typename T>
constexpr parameter_kind_t kind_of(T&&) { return kind_of<T>(); }

template <typename T>
void set_argument(kernel_t const & kernel, parameter_index_t index, T&& argument)
{
    constexpr auto kind = kind_of<T>();
    set_argument<T>(lifted_kind_t<kind>{}, kernel, index, std::forward<T>(argument));
}

} // namespace detail

} // namespace kernel

namespace detail {

// To be called on kernels which have had all of their arguments set
inline queue::event_t enqueue_primed_kernel_launch(
    kernel_t const& kernel,
    queue_t const& queue,
    kernel::launch_configuration_t const& launch_config)
{
    span<event::handle_t> no_events {};
    event::handle_t result_event_handle;
    auto launch_config_offset = launch_config.offset ? launch_config.offset->data() : nullptr;
    auto status = clEnqueueNDRangeKernel(
        queue.handle(),
        kernel.handle(),
        launch_config.dimensions.dimensionality(),
        launch_config_offset,
        launch_config.dimensions.overall.data(),
        launch_config.dimensions.workgroup.data(),
        no_events.size(), no_events.data(),
        &result_event_handle);
    throw_if_error_lazy(status, "clEnqueueNDRangeKernel", "Enqueueing " + opencl::detail::identify(kernel)
        + " on " + opencl::detail::identify(queue));
    return event::detail::wrap_by_queue(result_event_handle, queue, is_owning);
}

template<class...Args>
void set_each_kernel_arg(kernel_t const& kernel, kernel::detail::parameter_index_t& index, Args&&...args) {
    (void) std::initializer_list<int>{(kernel::detail::set_argument(kernel, index++, std::forward<Args>(args)), 0)...};
}

} // namespace detail

template <typename... Ts>
queue::event_t enqueue_launch(
    kernel_t const& kernel,
    queue_t const& queue,
    kernel::launch_configuration_t const& launch_config,
    Ts&&... arguments)
{
#ifndef NDEBUG
    auto num_params = kernel.num_parameters();
    auto num_args = sizeof...(Ts);
    if (num_args != num_params) {
        std::stringstream oss;
        oss << "Attempt to launch " << detail::identify(kernel) << " with " << num_args
            << "argument, but the kernel function has " << num_params << " parameters";
        throw std::invalid_argument(oss.str());
    }
#endif
    kernel::detail::parameter_index_t index = 0;
    detail::set_each_kernel_arg(kernel, index, std::forward<Ts>(arguments)...);
    return detail::enqueue_primed_kernel_launch(kernel, queue, launch_config);
}

} // namespace opencl

#endif // OPENCL_WRAPPERS_IMPL_LAUNCH_HPP_
