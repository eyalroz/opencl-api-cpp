/**
* @file
*
* @brief Definitions of a hierarchy of classes, with{@ref program_t} as
* its base class, as well as declarationsof named constructor idioms, comparisons
* and other related functions.
*/
#ifndef OPENCL_WRAPPERS_PROGRAM_HPP_
#define OPENCL_WRAPPERS_PROGRAM_HPP_

#include "types.hpp"
#include "context.hpp"

#include <future>

#include "options/compilation.hpp"

namespace opencl {

namespace program {

namespace detail {

template <typename Program>
Program upcast(program_t const& program) = delete; // but there are specializations...

// Avoid using this, as much as possible. It is more convenient for the user to have
// an instance of one of the child classes of program_t, indication what kind of
// program components the "program" actually holds, and exposing relevant methods. This
// exists for internal use (and since friend functions must be declared
program_t wrap(
    platform::handle_t  platform_handle,
    context::handle_t   context_handle,
    program::handle_t   program_handle,
    bool                owning) noexcept;

with_kernels_t wrap_with_kernels(
    platform::handle_t platform_handle,
    context::handle_t context_handle,
    program::handle_t program_handle,
    optional<dynarray<std::string>> kernel_names,
    bool owning) noexcept;

with_intermediate_language_t wrap_with_intermediate_language(
    platform::handle_t platform_handle,
    context::handle_t context_handle,
    program::handle_t program_handle,
    bool owning) noexcept;

} // namespace detail

template <
    template <typename> class BinariesContainer,
    template <typename> class DevicesContainer>
with_binaries_t create(
    context_t const &                                 context,
    BinariesContainer<input::binary_t const> const &  binaries,
    DevicesContainer<device_t const> const &          devices);

#ifdef CL_VERSION_1_2
/**
 * Creates kernel objects exposing built-in OpenCL kernels, which can than be
 * scheduled like users' own compiled and linked kernels.
 *
 * @param context
 *     The OpenCL context in which the kernels are to be instantiated and used
 * @param kernel_names
 *     Names of the OpenCL-builtin kernels to be made accessible through the created object.
 */
///@{

/// @param devices The devices on which the kernels are to be made available
template <template <typename> class DevicesContainer>
with_kernels_t create_with_builtin_kernels(
    context_t const &                         context,
    span<string_view const>                   kernel_names,
    DevicesContainer<device_t const> const &  devices);

/// @param device The single device on which the kernels are to be made available
with_kernels_t create_with_builtin_kernels(
    context_t const &                         context,
    span<string_view const>                   kernel_names,
    device_t const &                          device);

///@}
#endif

// Need to support compiling by passing initialization list of cl_programs (which themselves were perhaps created from source code)

} // namespace program

// TODO: Reference counting
/**
 * @brief A class representing [OpenCL 'programs'](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html#_program_objects),
 * wrapping the handle type of the OpenCL C API, and serving as a base class for more specialized
 * program classes exhibiting various features/aspects.
 *
 * The OpenCL API uses a single handle for collections of entities at different points
 * in the process of building and preparing invocable, executable code: source files in either
 * a higher-level or intermediate language; source file compiled in binary code usable on some
 * devices, but not yet linked; and fully-linked code from which @ref kernel_t objects can be
 * obtained and invoked. The underlying C-ish API has all relevant operations be applicable to
 * "programs"; but we shall distinguish among them based on what they contain and what they
 * may already have undergone, exposing operations as relevant. This is the lowest common
 * denominator, used - through inheritance - in all "programs", and with functionality they
 * all support.
 *
 * @note No method is marked constexpr, as OpenCL programs are not intended to come into
 * existence at compile-time
 */
class program_t {
public:
    using handle_type = program::handle_t;

protected:
    platform::handle_t platform_handle_;
    context::handle_t context_handle_;
    handle_type handle_;
    detail::handle_ownership_token_t<program_t> ownership_token_;

public:
    platform::handle_t platform_handle() const noexcept { return platform_handle_; }
    context::handle_t context_handle() const noexcept { return context_handle_; }
    handle_type handle() const noexcept { return handle_; }
    bool owning() const noexcept { return ownership_token_.owning(); }

    platform_t platform() const noexcept;
    context_t context() const noexcept;

    // bool input_type_is_known() const noexcept { return input_type_.has_value(); }
    // shall we make this a virtual method?
    //    program::input_type_t input_type() const { return *input_type_; }

protected:
    program_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        program::handle_t program_handle,
        bool owning) noexcept
    :
        platform_handle_(platform_handle), context_handle_(context_handle), handle_(program_handle),
        ownership_token_(owning, program_handle) {}

public:
    friend program_t program::detail::wrap(platform::handle_t, context::handle_t, program::handle_t, bool) noexcept;
    friend program::with_kernels_t program::detail::wrap_with_kernels(
        platform::handle_t, context::handle_t, program::handle_t, optional<dynarray<std::string>>, bool) noexcept;
    friend program::with_intermediate_language_t program::detail::wrap_with_intermediate_language(
        platform::handle_t, context::handle_t, program::handle_t, bool owning) noexcept;

    bool has_sources() const;
    bool has_intermediate_language() const;
    bool has_binaries() const;
    bool has_kernels() const;
}; // class program_t

inline bool operator==(const program_t& lhs, const program_t& rhs) noexcept;
inline bool operator!=(const program_t& lhs, const program_t& rhs) noexcept { return not (lhs == rhs); }

bool has_sources(program_t const& program);
bool has_intermediate_language(program_t const& program);
bool has_binaries(program_t const& program);
bool has_kernels(program_t const& program);

} // namespace opencl

// We need this specialization to use device_t's as unordered_map keys
template<> struct std::hash<opencl::device_t> {
    std::size_t operator()(opencl::device_t const& device) const noexcept {
        return std::hash<opencl::device::handle_t>{}(device.handle());
    }
};

namespace opencl {

namespace program {

class with_intermediate_language_t : public virtual program_t {
    using parent_type = program_t;
    using parent_type::parent_type;

    friend with_intermediate_language_t create_with_intermediate_language(context_t const &, memory::region_t);
    //template<> friend intermediate_language_t ::opencl::detail::upcast<intermediate_language_t>(program_t const& program);
    // friend template<typename Program> Program ::opencl::detail::upcast(program_t const& program);
    friend with_intermediate_language_t wrap_with_intermediate_language(
        platform::handle_t, context::handle_t, handle_t, bool) noexcept;

public:
    string_view intermediate_language() const;
};

struct binary_t {
    optional<device_t> target;
#if CL_VERSION_1_2
    binary_type_t type;
#endif
    raw_binary_t raw;
};

namespace detail {

with_binaries_t wrap_binaries(
    platform::handle_t platform_handle,
    context::handle_t  context_handle,
    handle_t           program_handle,
    bool               owning);

} // namespace detail

// TODO: Considering making this expose a binaries field,
// or method, which retrieves a sequence of wrapped binaries,
// each with a device, a span of bytes, and maybe more info
class with_binaries_t : public virtual program_t {
public:
    using parent_type = program_t;
    using map_type = std::unordered_map<device_t, binary_t>;

    friend with_binaries_t wrap_binaries(
        platform::handle_t platform_handle,
        context::handle_t  context_handle,
        handle_t           program_handle,
        bool               owning);

protected:
    with_binaries_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        handle_t program_handle,
        bool owning) noexcept
    :
        parent_type(platform_handle, context_handle, program_handle, owning)
    {}

public:
    map_type binaries() const;
    dynarray<device_t> targets() const;
    optional<binary_t> binary_for(device_t const& target) const;

#ifdef CL_VERSION_1_2
    binary_type_t binary_type_for(device_t const& target) const;
#endif
}; // class with_binaries_t

template <
    template <typename> class BinariesContainer,
    template <typename> class DevicesContainer>
with_binaries_t create(
    context_t const &                                 context,
    BinariesContainer<input::binary_t const> const &  binaries,
    DevicesContainer<device_t const> const &          devices);

template <template <typename> class DevicesContainer>
with_binaries_t create(
    context_t const &                                 context,
    input::binary_t const &                           binary,
    DevicesContainer<device_t const> const &          devices);

template <template <typename> class BinariesContainer>
with_binaries_t create(
    context_t const &                                 context,
    BinariesContainer<input::binary_t const> const &  binaries,
    device_t const &                                  device);

with_binaries_t create(
    context_t const &                                 context,
    input::binary_t const &                           binary,
    device_t const &                                  device);

// Note that built-in kernels have no binaries!
class with_kernels_t : public virtual program_t {
    using parent_type = program_t;
public:
    // Note: Need to make the getter thread-safe
    mutable optional<dynarray<std::string>>  kernel_names_;

public:
    with_kernels_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        program::handle_t program_handle,
        optional<dynarray<std::string>> kernel_names,
        bool owning) noexcept
    :
        parent_type(platform_handle, context_handle, program_handle, owning), kernel_names_(std::move(kernel_names)) {}
    // getting a kernel by name ; for now, no kernels facade member... or - should I allow that after all?

    // Note: This takes a const char* rather than a string view, since the OpenCL API call needs a
    // null-terminated string. Should I also expose something more convenient, which would require
    // making a copy?
    kernel_t instantiate_kernel(char const* kernel_name) const;
    dynarray<kernel_t> instantiate_kernels(span<char const*> kernel_names = {}) const;

    // TODO: Treat the program like a collection of kernels, or at least -
    // offer a member which is a collection of kernel names

    // get-info-based methods
    // ...
    size_t num_kernels() const;
    dynarray<std::string> const& kernel_names() const;
}; // with_kernels_t

class build_step_result_t : public with_binaries_t {
public:
    using parent_type = with_binaries_t;

protected:
    build_step_result_t(
        platform::handle_t platform_handle,
        context::handle_t context_handle,
        handle_t program_handle,
        bool succeeded,
        bool owning
        // dynarray<device::handle_t> device_handles,
        // dynarray<program::binary_t> binaries
        ) noexcept
    :
        program_t(platform_handle, context_handle, program_handle, owning),
        parent_type(platform_handle, context_handle, program_handle, owning),
        succeeded_(succeeded)
    {}

public:
    bool succeeded() const { return succeeded_; }
    dynarray<build_info_t> info() const;
    build_info_t info_for(device_t const&) const;

protected:
    bool succeeded_;
    // dynarray<device::handle_t> device_handles_;
    // // TODO: Should we really cache the binaries spans? It's not a lot of memory, but still
    // dynarray<program::binary_t> binaries_;
}; // class build_step_result_t

void unload_compiler();

with_intermediate_language_t create_with_intermediate_language(
    context_t const &  context,
    memory::region_t   il_source);


} // namespace program

} // namespace opencl

#endif // OPENCL_WRAPPERS_PROGRAM_HPP_
