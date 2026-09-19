
#ifndef OPENCL_WRAPPERS_TYPES_HPP_
#define OPENCL_WRAPPERS_TYPES_HPP_

#include "util/string_view.hpp"
#include "util/span.hpp"
#include "util/optional.hpp"
#include "util/dynarray.hpp"
#include "util/static_dynarray.hpp"

// We'll be more friendly to "newbie" users by assuming they want OpenCL 3.0
// without a warning
#if !defined(CL_TARGET_OPENCL_VERSION)
#define CL_TARGET_OPENCL_VERSION 300
#endif

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <CL/cl_ext.h>
#endif // __APPLE__

#ifndef OCLW_MAX_NDRANGE_DIMENSIONS
#define OCLW_MAX_NDRANGE_DIMENSIONS 3
#endif

#include <cstddef>
#include <numeric>

namespace opencl {
enum : bool {
    is_nullable = true,
    is_not_nullable = false,
    isnt_nullable = false,
};

enum : bool {
    is_not_owning = false,
    is_owning = true,
    isnt_owning = false,
};

enum : bool {
    is_read_only = true,
    is_not_read_only = false,
    isnt_read_only = is_not_read_only,
};

enum : bool {
    is_blocking = false,
    is_non_blocking = true,
    isnt_blocking = false,
};

// TODO: Consider removing string_view from here, and changing the string_view
// source to be in the opencl namespace
using string_view = nonstd::string_view;

/// @brief A inheritance-based-tag for string_views, guaranteeing that their data
/// is null-terminated at the a position  corresponding to their length.
///
/// TODO: Consider putting this in detail
struct null_terminated_string_view : string_view {};

// Aliases for standard types

using size_t = std::size_t;
#if __cplusplus >= 201703L
using byte_t = std::byte;
#else
using byte_t = unsigned char;
#endif

// Aliases for C-API types

using status_t            = cl_int;
using dimensionality_t    = cl_uint;
// TODO: Perhaps make this into a multi-component structure; I have a hunch this uses X * 10 + Y for "X.Y"
using raw_version_t       = cl_version;
using vendor_id_t         = cl_uint;
using clock_frequency_t   = cl_uint;
using address_bit_count_t = cl_uint;
using timestamp_t         = cl_ulong;
namespace platform { using handle_t       = cl_platform_id; }
namespace platform { using index_t        = cl_uint; }
namespace platform { using attribute_id_t = cl_platform_info; }
namespace device   { using handle_t       = cl_device_id; }
namespace device   { using index_t        = cl_uint; }
namespace device   { using count_t        = cl_uint; }
namespace device   { using vendor_id_t    = cl_uint; }
namespace device   { using compute_unit_count_t = cl_uint; }
namespace context  { using handle_t       = cl_context; }
namespace context  { namespace device { using opencl::device::index_t; } }
namespace queue    { using handle_t       = cl_command_queue; }
namespace program  { using handle_t       = cl_program; }
namespace program  { struct binary_t;  }
namespace event    { using handle_t       = cl_event; }
namespace version  { using component_t    = int; }
namespace memory   { using offset_t       = std::size_t; }
namespace memory   { using alignment_t    = cl_uint; }
namespace memory   { using handle_t       = cl_mem; } // Note: This is used for different memory objects, e.g. @ref buffer_t @ref pipe_t, @ref image_t
namespace memory   { using flags_t        = cl_mem_flags; }
namespace memory   { using handle_t       = cl_mem; }
namespace image    { using descriptor_t   = cl_image_desc; }
namespace kernel   { using handle_t       = cl_kernel; }
namespace kernel   { using arity_t        = cl_uint; }
namespace kernel   { using attribute_id_t = cl_kernel_info; }
namespace kernel   { namespace parameter { using index_t = cl_uint; } }
namespace pipe     { using capacity_t     = cl_uint; }
namespace pipe     { using packet_size_t  = cl_uint; }
namespace pipe     { using handle_t       = cl_mem; }
namespace buffer   { using handle_t       = cl_mem; }
namespace image    { using handle_t       = cl_mem; }
namespace image    { using attribute_id_t = cl_image_info; }
namespace image    { using dimension_t    = size_t; }
namespace image    { using mipmap_level_t = cl_uint; }
namespace image    { namespace channel { using count_t = cl_uint; } }
namespace image    { namespace sampler { using handle_t = cl_sampler; } }
// TODO: I am of two minds about the names of the following two; I don't much like Khronos' choice here
namespace image    { namespace sampler { enum class addressing_mode_t : cl_addressing_mode; } }
namespace image    { namespace sampler { enum class filtering_mode_t : cl_filter_mode; } }
#if CL_VERSION_1_2
namespace memory   { namespace migration { using flags_t = cl_mem_migration_flags; } }
#endif // CL_VERSION_1_2
#if cl_ext_buffer_device_address
namespace memory  { using device_address_t = cl_mem_device_address_ext; }
#endif

// Forward declarations

struct version_t;
class platform_t;
class device_t;
class buffer_t;
class image_t;
class kernel_t;
class program_t;
namespace device { class capabilities_t; }
namespace program {
class with_intermediate_language_t;
class source_t;
class with_binaries_t;
class with_kernels_t;
using header_t = source_t;
class compilation_t;
class link_t;
class build_info_t;
}
class context_t;
class queue_t;
using command_queue_t = queue_t;
class event_t;
class user_event_t;
namespace queue { class event_t; }
namespace queue { namespace event { using handle_t = opencl::event::handle_t; } }
class pipe_t;
namespace context { class device_t; }
namespace memory {
class object_t;
class copy_parameters_t;
class properties_t;
namespace local { struct allowance_t; }
}
using memory_object_t = memory::object_t;
namespace kernel {
struct launch_configuration_t;
class arguments_t; // Will we really be using this one?
namespace parameter {
struct info_t;
}
struct nd_range_info_t;
} // namespace kernel
namespace program {
class build_result_t;
namespace compilation { struct options_t; }
namespace link { struct options_t; }
using build_options_t = compilation::options_t;
}
namespace image {
class sampler_t;
class array_t;
struct format_t;
struct spec_t;
}

namespace device {

enum class type_t : cl_device_type {
    default_    = CL_DEVICE_TYPE_DEFAULT, // Not a proper device type, but... that's how OpenCL defined it
    cpu         = CL_DEVICE_TYPE_CPU,
    gpu         = CL_DEVICE_TYPE_GPU,
    accelerator = CL_DEVICE_TYPE_ACCELERATOR,
#ifdef CL_VERSION_1_2
    custom      = CL_DEVICE_TYPE_CUSTOM
#endif
    // Note: Not defining a value corresponding to "default" and to "all"; that's ugly punning which
    // breaks the semantics of this type and it won't do
};

struct synchronized_timestamps_t { timestamp_t host, device; };

enum affinity_domain_t {
    numa = CL_DEVICE_AFFINITY_DOMAIN_NUMA,
    NUMA = numa,
    l4_cache = CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE,
    l3_cache = CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE,
    l2_cache = CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE,
    l1_cache = CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE,
    /// The following is useful when a device is already a sub-device, partitioned
    /// according to some domain, and one wants to partition it further
    next_partitionable = CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE,
    next_partitionable_domain = next_partitionable,
};

namespace partition {

enum class mode_t {
    equally = CL_DEVICE_PARTITION_EQUALLY,
    equipartition = equally,
    by_counts = CL_DEVICE_PARTITION_BY_COUNTS,
    by_compute_unit_counts = by_counts,
    by_affinity = CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
    by_affinity_domain = by_affinity,
};

struct spec_t {
    mode_t mode;
    union subspec_t {
        index_t num_parts;
        span<compute_unit_count_t> compute_unit_counts;
        affinity_domain_t affinity_domain;
        subspec_t() noexcept {}
        subspec_t(index_t num_parts) noexcept : num_parts(num_parts) {}
        subspec_t(affinity_domain_t affinity_domain) noexcept : affinity_domain(affinity_domain) {}
        subspec_t(span<compute_unit_count_t> compute_unit_counts) noexcept : compute_unit_counts(compute_unit_counts) {}
        ~subspec_t() = default;
    } subspec;
};

spec_t equally(index_t num_parts);
spec_t by_counts(span<compute_unit_count_t> counts);
spec_t by_affinity(affinity_domain_t affinity_domain);

} // namespace partition

} // namespace device

inline char const* name_of(device::affinity_domain_t domain);

namespace context {
namespace device { using handle_t = opencl::device::handle_t; }
namespace device { using type_t = opencl::device::type_t; }
} // namespace context

namespace queue {
using callback_t = void (CL_CALLBACK *)(void *);
} // namespace queue

// TODO: Maybe we could create literals with _access ?
enum class access_kind_t : unsigned char {
    none           = 0,
    read           = 1 << 0,
    write          = 1 << 1,
    readwrite      = read | write,
    read_and_write = readwrite,
};

access_kind_t operator&(access_kind_t lhs, access_kind_t rhs);
access_kind_t operator|(access_kind_t lhs, access_kind_t rhs);

namespace memory {

enum class destination_t { host, device };

struct host_and_kernel_access_t {
    access_kind_t host;
    access_kind_t kernel;

    operator flags_t() const;
};

host_and_kernel_access_t operator&(host_and_kernel_access_t lhs, host_and_kernel_access_t rhs);
host_and_kernel_access_t operator|(host_and_kernel_access_t lhs, host_and_kernel_access_t rhs);

constexpr host_and_kernel_access_t full_access() noexcept;

using subregion_spec_t = cl_buffer_region;

namespace mapping {

enum class kind_t : unsigned char {
    read,
    write,
    read_and_write,
    readwrite = read_and_write,
    invalidate_and_write,
    full_overwrite = invalidate_and_write,
};

/// A specification for mapping a memory object (buffer or image) to a host-side memory region
struct spec_t {
    kind_t kind { kind_t::read_and_write };
    /// What part of the buffer should be mapped? If this is disenaged - the entire buffer is mapped
    optional<subregion_spec_t> subregion { };

    spec_t() = default;
    spec_t(kind_t kind_, optional<subregion_spec_t> subregion_spec = nullopt) : kind(kind_), subregion(std::move(subregion_spec)) {}
    spec_t(subregion_spec_t subregion_spec) : subregion(subregion_spec) {}
};

} // namespace mapping

#ifdef CL_VERSION_2_0
namespace shared_virtual {

class region_t;
class const_region_t;

using flags_t = cl_svm_mem_flags;

struct access_spec_t {
    opencl::access_kind_t access_kind;
    bool fine_grain;
    bool atomics;
};

enum { default_alignment = 0 };
enum : flags_t {
    coarse_grain = 0,
    fine_grain = CL_MEM_SVM_FINE_GRAIN_BUFFER,
    with_atomics = CL_MEM_SVM_ATOMICS,
    no_atomics = 0
};

flags_t make_flags(access_spec_t access_spec);
access_spec_t read_and_write() noexcept;

} // namespace shared_virtual
#endif

} // namespace memory

using dimension_t = size_t;
using dimension_index_t = dimensionality_t;

template <dimensionality_t MaxDimensions>
struct dimensions_t : static_dynarray<dimension_t, MaxDimensions> {
public:
    using parent_type = static_dynarray<dimension_t, MaxDimensions>;
    using typename static_dynarray<dimension_t, MaxDimensions>::size_type;
protected:
    static void validate_num_dimensions(size_type num_dimensions) NOEXCEPT_IF_NDEBUG;
public:
    using parent_type::parent_type;
    using parent_type::operator[];

    dimensions_t(dimension_index_t dimensionality = 0) noexcept
    : parent_type(generate_uniform_static_dynarray<MaxDimensions>(dimensionality, 1lu)) { }
    // Note: The move is actually redundant, since these are integers, but - some IDEs are complaining
    dimensions_t(parent_type values) noexcept : parent_type{std::move(values)} {}
    dimensions_t(dimensions_t const& other) noexcept = default;
    dimensions_t(dimensions_t && other) noexcept = default;
    // TODO: Perhaps expand this to any container? But then - what about static_dynarray itself?
    dimensions_t(dynarray<dimension_t> const& dims) : parent_type{to_static_dynarray<MaxDimensions>(dims)} {}

    size_t volume() const noexcept;
    dimension_index_t effective_dimensionality() const noexcept;
    dimension_index_t dimensionality() const noexcept;

    dimensions_t& operator=(const dimensions_t& other) = default;
    dimensions_t& operator=(dimensions_t&& other) = default;
};

template <dimensionality_t MaxDimensions>
CONSTEXPR_CPP20 bool operator==(dimensions_t<MaxDimensions> const& lhs, dimensions_t<MaxDimensions> const& rhs) noexcept;
template <dimensionality_t MaxDimensions>
CONSTEXPR_CPP20 bool operator!=(dimensions_t<MaxDimensions> const& lhs, dimensions_t<MaxDimensions> const& rhs) noexcept;

template <dimensionality_t MaxDimensions>
struct offset_t : static_dynarray<dimension_t, MaxDimensions> {
public:
    using parent_type = static_dynarray<dimension_t, MaxDimensions>;
    using typename static_dynarray<dimension_t, MaxDimensions>::size_type;
public:
    using parent_type::parent_type;
    using parent_type::operator[];

    offset_t(dimension_index_t dimensionality = 0)
    : parent_type(generate_static_dynarray<MaxDimensions>(dimensionality, [](size_t) { return 1lu; })) { }
    // Note: The move is actually redundant, since these are integers, but - some IDEs are complaining
    offset_t(parent_type values) : parent_type{std::move(values)} {}
    offset_t(offset_t const& other) noexcept = default;
    offset_t(offset_t && other) noexcept = default;

    // constexpr dimension_index_t dimensionality() const;

    offset_t& operator=(const offset_t& other) noexcept = default;
    offset_t& operator=(offset_t&& other) noexcept = default;
};

namespace nd_range {

using opencl::size_t;
using opencl::dimension_index_t;
using opencl::dimension_t;

enum : dimension_index_t { max_dimensions = OCLW_MAX_NDRANGE_DIMENSIONS };

using dimensions_t = opencl::dimensions_t<max_dimensions>;
using offset_t = opencl::offset_t<max_dimensions>;

// TODO: Alter dimensions_t so as to allow for easy extension-with-ones for construction
#if OCLW_MAX_NDRANGE_DIMENSIONS == 3
/// Dimensions of an equi-lateral 3D cube
inline dimensions_t cube(dimension_t x) noexcept   { return dimensions_t{ x, x, x }; }

/// Dimensions of an equi-lateral 2D square, with a trivial third dimension
inline dimensions_t square(dimension_t x) noexcept { return dimensions_t{ x, x, 1lu }; }

/// Dimensions of a 1D line, with the last two dimensions trivial
inline dimensions_t line(dimension_t x) noexcept   { return dimensions_t{ x, 1lu, 1lu }; }
#endif // OCLW_MAX_NDRANGE_DIMENSIONS == 3

/// Dimensions of a single point - trivial in all axes
///
/// @TODO Templatize and constexpr'ify this
inline dimensions_t point(dimension_index_t dimensionality) noexcept;

using workgroup_dimensions_t = dimensions_t;
using overall_dimensions_t = dimensions_t;

/**
 * Composite dimensions for an nd_range - in terms of workgroups, then also down
 * into the workgroup dimensions completing the information to the thread level.
 *
 * @todo Should we verify that workgroup divides overall, on construction?
 */
struct composite_dimensions_t {
    nd_range::overall_dimensions_t   overall;
    nd_range::workgroup_dimensions_t workgroup;

    /// @returns The overall dimensions of the entire grid as a single 3D entity
    overall_dimensions_t flatten() const noexcept { return overall; }

    /// @returns The total number of threads over all workgroups of the grid
    size_t volume() const noexcept { return overall.volume(); }

    /// @returns the number of axes in which the grid overall has a defined dimension.
    ///
    /// @note This is not the _effective_ dimensionality. For example, the overall
    /// dimensions may be [1, 1, 1] - a single thread in the entire grid; but this will
    /// return 3.
    size_t dimensionality() const noexcept { return overall.dimensionality(); }

    size_t effective_dimensionality() const noexcept { return flatten().effective_dimensionality(); }

    // TODO: Implement the following
    // bool validate() const noexcept(false);

    /// A named constructor idiom for the composite dimensions of a single-workgroup grid
    /// with a single-thread workgroup
    static composite_dimensions_t point(dimension_index_t dimensionality) noexcept;

    /// Increase the overall dimensions to consist of integer multiples of the local dimensions
    void round_up_overall_dimensions();
};

} // namespace nd_range

/// A 1, 2 or 3-dimensional box of elements within some span or sequence of elements.
struct box_spec_t {
    enum { max_dimensions = 3 };
    offset_t<max_dimensions>          origin;
    dimensions_t<max_dimensions>      extents;
    dimensions_t<max_dimensions - 1>  pitches;
};

box_spec_t rect(dimensions_t<2> origins, dimensions_t<2> extents, dimension_t row_pitch);
box_spec_t rect(dimensions_t<2> origins, dimensions_t<2> extents);

struct box_target_spec_t {
    enum { max_dimensions = 3 };
    offset_t<max_dimensions>          origin;
    dimensions_t<max_dimensions - 1>  pitches;
};

box_spec_t rect_target(dimensions_t<2> origins, dimension_t row_pitch);
box_spec_t rect(dimensions_t<2> origins);

namespace program {

enum class input_type_t {
    text_source           = 0,
    intermediate_language = 1,
    binary                = 2,
    builtin_kernel_name   = 3,
    il = intermediate_language,
};

/// Non-owning source code text, with an associated identifying name
struct named_source_text_t {
    /// A name identifying the source text
    ///
    /// @note the name need not be that of a kernel function, nor of a filename from which the
    /// source code was read - it is arbitrary.
    string_view name;

    /// The source code
    ///
    /// @note must be non-null.
    string_view text;
};

using raw_binary_t = span<byte_t const>;

enum class binary_type_t : cl_program_binary_type {
    none            = CL_PROGRAM_BINARY_TYPE_NONE,
    compiled_object = CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT,
    library         = CL_PROGRAM_BINARY_TYPE_LIBRARY,
    executable      = CL_PROGRAM_BINARY_TYPE_EXECUTABLE,
    intermediate    = CL_PROGRAM_BINARY_TYPE_INTERMEDIATE
};

enum class build_status_t : cl_build_status {
    none        = CL_BUILD_NONE,
    error       = CL_BUILD_ERROR,
    success     = CL_BUILD_SUCCESS,
    in_progress = CL_BUILD_IN_PROGRESS,

    // not_started = none,
    // failure = error,
    // failed = error,
    // succeeded = success,
};

constexpr int operator==(build_status_t lhs, int rhs) { return static_cast<int>(lhs) == rhs; }
constexpr int operator!=(build_status_t lhs, int rhs) { return not (lhs == rhs); }
constexpr int operator==(int lhs, build_status_t rhs) { return lhs == static_cast<int>(rhs); }
constexpr int operator!=(int lhs, build_status_t rhs) { return not (lhs == rhs); }

} // namespace program

constexpr bool is_success(program::build_status_t status) noexcept  { return static_cast<int>(status) == CL_BUILD_SUCCESS; }

inline char const* name_of(program::build_status_t status);

namespace program {
namespace input {

// TODO:
// 1. Consider using a struct templated over the input type; although... that _would_ be more verbose
// 2. Consider using const types (e.g. const_region_t etc.
struct text_source_t { null_terminated_string_view input; };
struct intermediate_language_t { memory::region_t input; };
using il_t = intermediate_language_t;
using binary_t = memory::region_t;  // Is this a good idea considering program::binary_t?
using builtin_kernel_name_t = string_view;

} // namespace input

namespace detail {

template <input_type_t> struct input_t_helper;
template <> struct input_t_helper<input_type_t::text_source>           { using type = input::text_source_t; };
template <> struct input_t_helper<input_type_t::intermediate_language> { using type = input::intermediate_language_t; };
template <> struct input_t_helper<input_type_t::binary>                { using type = input::binary_t; };
template <> struct input_t_helper<input_type_t::builtin_kernel_name>   { using type = input::builtin_kernel_name_t; };

}

template <input_type_t input_type>
using input_t = detail::input_t_helper<input_type>;

} // namespace program

namespace event {

enum class execution_status_t : cl_int {
    queued                 = CL_QUEUED,
    in_queue = queued, enqueued = queued,
    submitted              = CL_SUBMITTED,
    submitted_to_device    = submitted,
    running                = CL_RUNNING,
    in_progress = running, in_flight = running,
    complete               = CL_COMPLETE,
};

}

// TODO: Make this into a multi-level case class
enum class command_type_t : cl_command_type {
    nd_range_kernel               = CL_COMMAND_NDRANGE_KERNEL, kernel = nd_range_kernel,
    task                          = CL_COMMAND_TASK,
    native_kernel                 = CL_COMMAND_NATIVE_KERNEL,
    read_buffer                   = CL_COMMAND_READ_BUFFER,
    write_buffer                  = CL_COMMAND_WRITE_BUFFER,
    copy_buffer                   = CL_COMMAND_COPY_BUFFER,
    read_image                    = CL_COMMAND_READ_IMAGE,
    write_image                   = CL_COMMAND_WRITE_IMAGE,
    copy_image                    = CL_COMMAND_COPY_IMAGE,
    copy_image_to_buffer          = CL_COMMAND_COPY_IMAGE_TO_BUFFER,
    copy_buffer_to_image          = CL_COMMAND_COPY_BUFFER_TO_IMAGE,
    map_buffer                    = CL_COMMAND_MAP_BUFFER,
    map_image                     = CL_COMMAND_MAP_IMAGE,
    unmap_memory_object           = CL_COMMAND_UNMAP_MEM_OBJECT,
    marker                        = CL_COMMAND_MARKER,
    acquire_gl_objects            = CL_COMMAND_ACQUIRE_GL_OBJECTS,
    release_gl_objects            = CL_COMMAND_RELEASE_GL_OBJECTS,
#ifdef CL_VERSION_1_1
    read_buffer_rectangle         = CL_COMMAND_READ_BUFFER_RECT, read_buffer_rect = read_buffer_rectangle,
    write_buffer_rectangle        = CL_COMMAND_WRITE_BUFFER_RECT, write_buffer_rect = write_buffer_rectangle,
    copy_buffer_rectangle         = CL_COMMAND_COPY_BUFFER_RECT, copy_buffer_rect = copy_buffer_rectangle,
    user                          = CL_COMMAND_USER,
#endif
#ifdef CL_VERSION_1_2
    barrier                       = CL_COMMAND_BARRIER,
    migrate_memory_objects        = CL_COMMAND_MIGRATE_MEM_OBJECTS, migrate_mem_objects = migrate_memory_objects,
    fill_buffer                   = CL_COMMAND_FILL_BUFFER,
    fill_image                    = CL_COMMAND_FILL_IMAGE,
#endif
#ifdef CL_VERSION_2_0
    shared_virtual_memory_free    = CL_COMMAND_SVM_FREE, svm_free = shared_virtual_memory_free,
    shared_virtual_memory_copy    = CL_COMMAND_SVM_MEMCPY, svm_copy = shared_virtual_memory_copy, svm_memcpy = shared_virtual_memory_copy,
    shared_virtual_memory_fill    = CL_COMMAND_SVM_MEMFILL, svm_fill = shared_virtual_memory_fill,
    shared_virtual_memory_map     = CL_COMMAND_SVM_MAP, svm_map = shared_virtual_memory_map,
    shared_virtual_memory_unmap   = CL_COMMAND_SVM_UNMAP, svm_unmap = shared_virtual_memory_unmap,
#endif
#ifdef CL_VERSION_3_0
    shared_virtual_memory_migrate = CL_COMMAND_SVM_MIGRATE_MEM, svm_migrate = shared_virtual_memory_migrate,
#endif
};

namespace image {

enum {
    max_image_dimensions = 3,
    max_image_pitches = max_image_dimensions - 1
};

using offset_t = opencl::offset_t<max_image_dimensions>;

struct dimensions_t : opencl::dimensions_t<max_image_dimensions> {
    using parent_type = opencl::dimensions_t<max_image_dimensions>;
    using parent_type::parent_type;
    constexpr dimension_t width() const NOEXCEPT_IF_NDEBUG { return (*this)[0]; }
    constexpr dimension_t height() const NOEXCEPT_IF_NDEBUG { return (*this)[1]; }
    constexpr dimension_t depth() const NOEXCEPT_IF_NDEBUG { return (*this)[2]; }
    constexpr dimension_t safe_width() const NOEXCEPT_IF_NDEBUG { return size() > 0 ? (*this)[0] : 0; }
    constexpr dimension_t safe_height() const NOEXCEPT_IF_NDEBUG { return size() > 1 ? (*this)[1] : 0; }
    constexpr dimension_t safe_depth() const NOEXCEPT_IF_NDEBUG { return size() > 2 ? (*this)[2] : 0; }
};

using pitches_t = static_dynarray<dimension_t, max_image_pitches>;
using mipmap_level_count_t = cl_uint;
enum { default_mipmap_level_count = 1 };

struct layout_t {
    dimensions_t          extents;
    pitches_t             pitches;
    mipmap_level_count_t  num_mipmap_levels;

    /// @note The volume is the _effective_ volume of image elements, not the
    /// space taken up considering also pitches
    size_t volume() const noexcept { return extents.volume(); }
    dimensionality_t dimensionality() const noexcept { return extents.dimensionality(); }
    // TODO: Perhaps add effective_dimensionality?
    optional<dimension_t> row_pitch() const noexcept;
    optional<dimension_t> slice_pitch() const noexcept;
    size_t storage_size_in_elements() const noexcept;
};

enum class addressing_mode_t : cl_addressing_mode {
    none            = CL_ADDRESS_NONE,
    clamp_to_edge   = CL_ADDRESS_CLAMP_TO_EDGE,
    clamp           = CL_ADDRESS_CLAMP,
    clamp_to_value = clamp,
    repeat          = CL_ADDRESS_REPEAT,
#ifdef CL_VERSION_1_1
    mirrored_repeat = CL_ADDRESS_MIRRORED_REPEAT,
#endif
};

enum class filtering_mode_t : cl_filter_mode {
    nearest         = CL_FILTER_NEAREST,
    nearest_element = nearest,
    linear          = CL_FILTER_LINEAR,
    linear_interpolation = linear,
};

/// A 1, 2 or 3-dimensional box of pixels/voxels within an image
struct box_spec_t {
    offset_t       origin;
    dimensions_t   extents;
    // Note: No pitches
};

box_spec_t rect(opencl::dimensions_t<2> origins, opencl::dimensions_t<2> extents);

namespace mapping {
struct spec_t {
    memory::mapping::kind_t kind { memory::mapping::kind_t::read_and_write };
    optional<box_spec_t> box;
    optional<pitches_t> pitches;
};

} // namespace mapping

} // namespace image

namespace detail {

template <typename Handle> class handle_with_ownership;

} // namespace detail

namespace info {

namespace detail {
template <int InfoParameterAPINumber>
struct raw_info_parameter_type_mapper;

} // namespace detail

template <int InfoParameterAPINumber>
using parameter_value_t = typename detail::raw_info_parameter_type_mapper<InfoParameterAPINumber>::type;

namespace detail {
struct kernel_device_handle_pair_t {
    kernel::handle_t  kernel_;
    device::handle_t  device_;
};
struct program_device_handle_pair_t {
    program::handle_t program_;
    device::handle_t  device_;
};
struct kernel_param_index_pair_t {
    kernel::handle_t           kernel_;
    kernel::parameter::index_t index_;
};

} // namespace detail

} // namespace info

} // namespace opencl

#define OCLW_CONCATENATE(s1, s2) s1##s2
#define OCLW_EXPAND_THEN_CONCATENATE(s1, s2) OCLW_CONCATENATE(s1, s2)

#endif // OPENCL_WRAPPERS_TYPES_HPP_
