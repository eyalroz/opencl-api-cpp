#ifndef OPENCL_WRAPPERS_BUILDERS_BUFFER_HPP_
#define OPENCL_WRAPPERS_BUILDERS_BUFFER_HPP_

#include "../buffer.hpp"
#include "memory_object.hpp"

// TODO: Add missing support for
// CL_MEM_IMMUTABLE_EXT               if cl_ext_immutable_memory_objects
// CL_MEM_DEVICE_HANDLE_LIST_KHR      if cl_khr_external_memory,
// CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT  if cl_ext_buffer_device_address

namespace opencl {

namespace builders {

// TODO: Add support for properties and clCreateBufferWithProperties
class buffer_t : public detail::memory_object_t<buffer_t> {
public:
    using parent_type = memory_object_t<buffer_t>;
    using parent_type::parent_type;
protected:
    using flags_type = memory::flags_t;

public:
    buffer_t& size(size_t size);
    template <typename ContiguousContainer>
    buffer_t& size_by(ContiguousContainer && container);

    opencl::buffer_t create();
    opencl::buffer_t build();

    // Add support for sub-buffers... maybe

    void validate(bool ensure_sufficiency) const override;
}; // buffer_t

/// A slightly shorter-named construction idiom for the @ref buffer_t builder
inline buffer_t buffer() { return {}; }

} // namespace builders

} // namespace opencl

#endif // OPENCL_WRAPPERS_BUILDERS_BUFFER_HPP_
