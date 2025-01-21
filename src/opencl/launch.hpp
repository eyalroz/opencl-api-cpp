#ifndef OPENCL_WRAPPERS_LAUNCH_HPP_
#define OPENCL_WRAPPERS_LAUNCH_HPP_

#include "types.hpp"

namespace opencl {

template <typename... Ts>
queue::event_t launch(
    kernel_t const& kernel,
    queue_t const& queue,
    kernel::launch_configuration_t const& launch_config,
    Ts&&... arguments);

} // namespace opencl

#endif // OPENCL_WRAPPERS_LAUNCH_HPP_
