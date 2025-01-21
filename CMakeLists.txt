# Does your system only have an older version of CMake? Not to worry!
# CMake offers download-and-use binary packages, with no installation 
# necessary...  Visit https://cmake.org/download/ and grab one for 
# your platform. They are not finicky with library dependencies, so
# compatability is very likely. Also, the package's CMake binary will
# not mistake any other local CMake-related files for its own.
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

PROJECT(opencl-api-cpp
	VERSION 0.1
	DESCRIPTION "C++ bindings for the OpenCL API"
	HOMEPAGE_URL https://github.com/eyalroz/opencl-api-cpp
)

include(GNUInstallDirs)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")

option(OCLW_USE_CUDA_OPENCL "Find OpenCL via the CUDA Toolkit" OFF)
if (OCLW_USE_CUDA_OPENCL)
	find_package(CUDAToolkit REQUIRED)
	add_library(opencl_tgt ALIAS CUDA::OpenCL)
else()
	find_package(OpenCL REQUIRED)
	add_library(opencl_tgt ALIAS OpenCL::OpenCL)
endif()

include(CheckLibraryExists)

set(oclw_namespace "OpenCL")

# First ugly hack to facilitate FetchContent use:
# Prefix target names with something project-specific
add_library(ocl-cpp-api INTERFACE)
target_compile_features(ocl-cpp-api INTERFACE cxx_std_11) # This means _at least_ C++11
target_include_directories(
	ocl-cpp-api
	INTERFACE
	"$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(ocl-cpp-api INTERFACE opencl_tgt)

# Additional hacks for facilitating FetchContent use,
# which in particular will let you use the same target names as if you had
# imported them with find_package
add_library("${oclw_namespace}::CppApi" ALIAS ocl-cpp-api)
set_target_properties(ocl-cpp-api
	PROPERTIES
	EXPORT_NAME CppApi
	OUTPUT_NAME CppApi
	WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# We should really not need this! CLion is stupid
target_sources(ocl-cpp-api PRIVATE src/opencl.hpp)

# --------
# Examples
# --------

option(OCLW_BUILD_EXAMPLES "Build example programs" OFF)

if (OCLW_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

# ------------------------
# Installing the libraries
# ------------------------

install(FILES "${PROJECT_SOURCE_DIR}/cmake/opencl-api-cpp-config.cmake"
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/opencl-api-cpp")

install(
	TARGETS ocl-cpp-api
	EXPORT opencl-api-cpp_export
	RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

export(
	EXPORT opencl-api-cpp_export
	NAMESPACE "${oclw_namespace}::"
	FILE "${PROJECT_BINARY_DIR}/opencl-api-cpp-targets.cmake"
)

install(
	DIRECTORY src/opencl
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
	FILES_MATCHING REGEX "\\.(h|hpp)$"
)

install(
	EXPORT opencl-api-cpp_export
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/opencl-api-cpp"
	NAMESPACE "${oclw_namespace}::"
	FILE "opencl-api-cpp-targets.cmake"
)

include(CMakePackageConfigHelpers)

set(COMPAT_SETTING SameMinorVersion)
write_basic_package_version_file(
	"opencl-api-cpp-config-version.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY ${COMPAT_SETTING}
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/opencl-api-cpp-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/opencl-api-cpp"
)
