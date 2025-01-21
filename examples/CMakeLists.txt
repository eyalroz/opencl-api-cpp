list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

enable_language(C)
enable_language(CXX)
include(CheckIncludeFiles)
include(CompileWithWarnings)

if (MSVC)
    # MSVC is very problematic! ... lets go easy on it
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /NODEFAULTLIB:nvptxcompiler_static.lib")
    set(CMAKE_CXX_STANDARD 17)
    add_compile_definitions("$<$<COMPILE_LANGUAGE:C,CXX>:_CRT_SECURE_NO_DEPRECATE>")
    add_compile_definitions("$<$<COMPILE_LANGUAGE:C,CXX>:_CRT_SECURE_NO_WARNINGS>")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

link_libraries(OpenCL::CppApi)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")

# NVIDIA OpenCL SDK

# Intel OpenCL examples

# using the C API (not the C++ bindings)

add_executable(event_refcount src/using_c_api/event_refcount.c printing.hpp)
add_executable(vector_addition_c_api src/using_c_api/vector_addition.c)
add_executable(c_api_vectorAdd src/using_c_api/vectorAdd_.c)

# Examples from the clhpp repository

add_executable(clhpp_header_example src/clhpp/headerexample.cpp)
set_target_properties(clhpp_header_example PROPERTIES CXX_STANDARD 20)
add_executable(clhpp_trivialSizeTCompat src/clhpp/trivialSizeTCompat.cpp)
add_executable(clhpp_trivialSizeTCompat_with_pipe src/clhpp/trivialSizeTCompat_with_pipe.cpp)

# Examples from the great-opencl-examples repository

add_executable(great_print_info src/great-opencl-examples/print_info/print_info.cpp)
add_executable(great_array_addition src/great-opencl-examples/array_addition/array_addition.cpp)
set_target_properties(great_array_addition PROPERTIES CXX_STANDARD 14)
add_executable(great_hello_world src/great-opencl-examples/hello_world/hello_world.cpp)
find_package(X11)
if (X11_FOUND)
    add_executable(great_image_filtering src/great-opencl-examples/image_filtering/image_filtering.cpp)
    target_link_libraries(great_image_filtering X11::X11)
    # TODO: At installation (and build?) time, copy the input image and the kernel to a path accessible by the executable
endif ()
# add_executable(great_matrix_multiplication src/great-opencl-examples/matrix_multiplication/matrix_multiplication.cpp)
# add_executable(great_cached_matrix_multiplication src/great-opencl-examples/cached_matrix_multiplication/cached_matrix_multiplication.cpp)

# Other example programs

add_executable(list_platforms src/other/list_platforms.cpp)
add_executable(vectorAdd src/vectorAdd/vectorAdd.cpp)
add_executable(vectorAdd_methods src/vectorAdd_methods/vectorAdd.cpp)
add_executable(inclusion_in_two_translation_units src/inclusion_in_two_translation_units/main.cpp src/inclusion_in_two_translation_units/second_tu.cpp)


