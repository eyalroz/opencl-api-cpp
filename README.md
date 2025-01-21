# Better OpenCL API C++ bindings

**--==Warning: WIP - for testing and evaluation only! ==--**

[![Repository License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Repository documentation](https://img.shields.io/badge/Documentation-CodeDocs-blue.svg)](https://codedocs.xyz/eyalroz/opencl-api-cpp/) 
<!-- [![Linux build passing](https://github.com/eyalroz/opencl-api-cpp/actions/workflows/cmake-build-linux.yml/badge.svg)](https://github.com/eyalroz/opencl-api-cpp/actions/workflows/cmake-build-linux.yml) 
[![Windows build passing](https://github.com/eyalroz/opencl-api-cpp/actions/workflows/cmake-build-windows.yml/badge.svg)](https://github.com/eyalroz/opencl-api-cpp/actions/workflows/cmake-build-windows.yml) -->

<!-- ![Conan Center](https://img.shields.io/conan/v/opencl-api-cpp) 
![Vcpkg Version](https://img.shields.io/vcpkg/v/opencl-api-cpp) -->



| Table of contents |
|:------------------|
|<sub>[Brief general description](#general-description)<br> [Key features](#key-features)<br>[Motivation](#motivation)<br>[Requirements](#requirements)<br>[Using the library in your project](#using-the-library-in-your-project)<br>[Coverage of the APIs](#coverage-of-the-apis)<br>[A taste of some features in play](#a-taste-of-some-features-in-play)<br>[Example programs](#example-programs)<br>[Want to help? Report a bug? Give feedback?](#want-to-help-report-a-bug-give-feedback)</sub>|

## General description

This is a header-only library of C++ bindings/wrappers for the [C-language API](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html) of the [OpenCL execution ecosystem](https://registry.khronos.org/OpenCL/).

It is an *unofficial* alternative to the [Khronos Consortium](https://www.khronos.org)'s own [CLHPP](https://github.com/KhronosGroup/OpenCL-CLHPP) C++ bindings. It is intended to make working with OpenCL be less error-prone, more intuitive and consistent (both to write and read), and requiring less memorizing and less familiarity with idiosyncracies. This is achieved by using modern C++ language capabilities, programming idioms and recommended practices; see [Motivation](#motivation) below for details.

Note: It is not the library's intention for users to write "higher-level" or more "abstract" code; these bindings are merely a modern-C++-arrangement of the OpenCL API itself.

### Key features

This library is characterized by:

- Judicious **namespacing** (and some internal namespace-like classes) for clarity and for semantic grouping of related functionality.
- **proxy/wrapper objects** for devices, queues, events, kernels, contexts, programs of various kinds etc. - all using the [RAII/CADRe](http://en.cppreference.com/w/cpp/language/raii) convention: Resources are obtained on construction,m and released on destruction. No need to check whether your object is in a "disengaged" or "uninitialized" state, nor to remember to initialize and release/free yourself.
- Methods and **functions return what they produce**, since they don't need to return a status code. No more having to pre-allocate result variables and pass pointers to them as out-parameters. Better compositionality!
- All functions and methods throw **informative exceptions** on failure, which carry not just the status code, but also the name of the failing C-API function name and some contextual information.
- You can **forget about numeric IDs and handles**; the proxy classes will fit everywhere **... or access them if you need to**, either using detail-implementation functions of the library, or for compatibility of other OpenCL-related code.
- With C++11 type deduction, you'll likely have **little need to worry/think about types**: What you obtain can be passed on to just like you might imagine; and if you've got it wrong - you'll get an informative error message. So, you may not even need to memorize the wrapper object type names for platforms, contexts, devices, kernels etc.
- Multiple **convenience methods and operators** so that you don't have to apply the Voodoo of OpenCL `getInfo()`mechanisms to get the attributes of your objects.
- Speaking of convenience, the library foregoes a multitude of complex constructors in favor of **named constructor idioms**, expressing intent more clearly and limiting the sets of necessary parameters.
- And as an alternative, you can employ step-by-step **intuitive builder classes for complex entities**; you can use those to set various aspects separately and gradually, and have the builder let you know if it has enough information or not.
- Aims for **clarity and straightforwardness** in naming and semantics, so that you don't need to refer to the official documentation to understand what each class and function do.
- Aims to conform to the [C++ core guidelines](https://github.com/isocpp/CppCoreGuidelines).
- **Header-only**: No need to compile anything special to use the library.
- If you do find you need to read the source, the library is separated into **readability-oriented headers** with no implementations, and a separate `impl/` subdirectory - with all those definitions and details.
- Permissive free software license: [Apache v2.0](LICENSE.txt).

## Motivation

Why even bother writing new C++ bindings for OpenCL? After all, these already exist - and are official to boot: Khronos' [CLHPP](https://github.com/KhronosGroup/OpenCL-CLHPP).

Well, I have used them on occasion. But - oh man! There are just too many damn problems with them, that I just give up and usually just fall back to using the C APIs. Let me point out some of what I've found to be so grating:

* **Painful error handling I**: The wrappers may throw exceptions, but - the function interfaces are simply _littered_ with `cl_int* error` parameters. Sometimes they don't even have an implicit default. Wanna launch a kernel? Not so fast. First spend a command defining a `cl_int` (why do I even have to know about this type?), and then hand over a reference to it.

* **Painful error handling II**: CLHPP exception's `what()` string are typically just one or two words long. Usually it's just the C API call which failed. No contextual description, or "story", is provided, to help with your diagnostics.

* **Painful error handling III**: If a CLHPP exception is thrown and treated like a general C++ exception, so what you get from it is the `what()` string - which typically gives you the API action which failed, but does not include the actuall error, i.e. the string interpretation of the error code - nor even the error code itself!

* **getInfo<>() instead of proper methods**: One of the tedious aspects of the C API is the 'bureaucratic' mechanism for getting information about objects: `clGetFooInfo()`, which takes many arguments; and you have to run it twice - once for size, once for the data; and you have to memorize the exact enum value. Ugh. Now, CLHPP does make this somewhat easier, but it's still `my_foo.getInfo<CL_FOO_CONTEXT>()` - and of course that gives you the raw value, not something nicely wrapped. Why do I still have to bear this punishment?

* **One huge file**: The source code of CLHPP is one file. It does not separate the declarations from the implementations; and the same file also contains a huge amount of `namespace detail` code which the user should be able to ignore. A single header file is more convenient to deploy - but that could be generated by amalgamting the separate header and offering the result as a release artifact.

* **Questionable default entities**: Many of the entity wrapper classes in CLHPP include a mechanism defining default values. So, there's a default `Context`, default `CommandQueue`, default lots-of-stuff. That annoys me both usability-wise and class-design-wise: There *aren't* such defaults in OpenCL. Why make them up? Just because we're using C++? No. This warps the user's perception of the OpenCL API. Moreover - I don't like this mechanism being complected by force into the basic class for the OpenCL entity. Want to have defaults? Fine, put them somewhere else.

* **Can't trust your objects** : With CLHPP, you can default-construct an OpenCL entity wrapper object. That's just wrong. Why should `cl::Device()` work? Am I running the device? Invoking the device? It must obviously have some kind of disengaged semantic... but - that means I have to be suspicious of every `cl::Device` instance I'm getting, anywhere in the code! A `cl::Device` might not be a real device, but rather a "null device", "disengaged value". And the same goes for essentially all classes. That undue burden is pretty much the same as having to make sure my `cl_device_id` is not null, whenever I receive one. I don't want to live in that world :-(

* **Why is everything invokable?** In CLHPP, most classes have an `operator()`. What is that? Am I running my platforms? Invoking my contexts? Nope. Well, those just yield ther raw handle (`cl_platform_id`, `cl_context_id` etc.). That's not how I want it to be accessible.

* **Consecutive arguments easy to confuse** One of the difficulties with the OpenCL C API is that many functions take numtiple parameters, some which are mutually convertible. Example: `flags` and `size` in `clCreateBuffer()`. I would expect C++ bindings to safeguard me from that: Either not having to specify both of them, or using not-implicitly-convertible types. But - CLHPP keeps that problem: Have a look at `cl::Buffer::Buffer()` - it takes exactly those two param

* **The vector obsession** Have you noticed how so many of the CLHPP methods take, or return, `std::vector`'s? `std::vector`, especially with the default allocator, is a notoriously unwieldy class: You can wrap storage in a vector, nor can you release the storage and use it for something else. So, if you have contiguous information somewhere, and you need to pass it to a CLHPP class - though luck, you're going to have to create a new vector for it. No templating nor even the use of `span<T>`'s.

* **Constructor glut** A common 'temptation' in C++ class design is offering a large number of constructors. There are a few cases where this might be merited or unavoidable, but it is typically more confusing than useful. `cl::Buffer` has 22 separate combinations of parameter types usable to construct it! (8 constructor definitions, 7 of which have 2 default parameter values each)... and some of them involve enqueuing copy operations on some command queue.

* **Different dimensions classes for 2D and 3D**, which means that you are either saddled with non-uniform handling of 1D, 2D and 3D cases, or alternatively, you have to rely on 'dummy' values in your 3D dimensions class to express 1D or 2D dims.

Having spent quite a while polishing my [CUDA Modern-C++ API wrappers](https://github.com/eyalroz/cuda-api-wrappers/) library, I was sure it could be done differently; and that if I put my nose to the grindstone, I could make it happen.

You may have noticed this list reads like the opposite of the [key features](#key-features), listed above: The idea is to make this library overcome and rectify these deficiencies as much as possible.

## Requirements

- A build environment supporting OpenCL: Allowing the compilation, linking and execution of programs which use OpenCL via the C API. Particularly,
  - A C or C++ build toolchain
  - An OpenCL driver (typically by one of the device manufactor: AMD, NVIDIA, Intel etc.)
- A C++11-capable C++ compiler.
- Recommended: CMake v3.25 or later - for using the library as a dependency. It is easy to [download and install](https://cmake.org/download/) a recent version of CMake - no need to build it yourself.

## Using the library in your project

### Projects using CMake

For CMake, you have several alternatives for obtaining the library to use in your project:

1. (apriori) If your development environment or OS distribution bundles the library and makes it available on [CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html) - then just skip this list.
2. (apriori) Download a release tarball from the [Releases](https://github.com/eyalroz/opencl-api-cpp/releases) page. Then, configure it with CMake, build it, and install it - to a place visible to cmake when it searches for packages (see [CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)).
2. (at config time) use CMake's `FetchContent` module to have CMake itself obtain the project source code and make it part of your own project's build, e.g.:

   ```cmake
   include(FetchContent)
   FetchContent_Declare(opencl-api-cpp_library
       GIT_REPOSITORY https://github.com/eyalroz/opencl-api-cpp.git
       GIT_TAG v12.34.56 # Replace this with a real available version
       OVERRIDE_FIND_PACKAGE
   )
   ```
Now that you have the package, in your project's `CMakeLists.txt`, you write:
```cmake
find_package(opencl-api-cpp CONFIG REQUIRED)
```
This will let you use the target: `opencl-api-cpp::api` as a dependency for your own targets. Example:
```cmake
target_link_library(my_app opencl-api-cpp::api)
```
**Use without CMake:**

Since this is a header-only library, you can simply add the `src/` subdirectory as one of your project's include directories. However, if you do this, it will be up to you to make sure and have the OpenCL headers include directory in your include path as well, and to link against the relevant OpenCL libraries.

## Coverage of the APIs

This library is intended to cover the OpenCL API, sans graphics-related interoperability functions (for OpenGL, DirectX, Direct3D etc.).

This goal is close to being achieved, but we're not all the way there. You can find remaining omissions as [issues tagged with "core-api-coverage"](https://github.com/eyalroz/opencl-api-cpp/issues?q=is%3Aissue+is%3Aopen+label%3Acore-api-coverage). Perhaps the most prominent omission at this time is execution graphs, i.e. 'queues' which may execute out-of-order, subject to defined dependencies.

Efforts will be made to support Khronos-defined extensions, but - this is less of a priority. Such extensions can be found via [issues tagged with "extension-coverage"](https://github.com/eyalroz/opencl-api-cpp/issues?q=is%3Aissue+is%3Aopen+label%3Aextension-coverage), further development work may take longer. The most prominent unsupported extension is probably [Command Buffers](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html#_command_buffers).

Vendor-specific extensions may or may not be supported, arbitrarily, with no promises made that point.

## A taste of some features in play

Let's start with the very first lines of [the example program](https://github.khronos.org/OpenCL-CLHPP/index.html#example) in the CLHPP documentation's root page:

<table>
<tr>
<th>
Khronos CLHPP
</th>
<th>
opencl-api-cpp
</th>
</tr>

<tr>

<td>
<pre lang="cpp">
std::vector<&zwnj;cl::Platform> platforms;
cl::Platform::get(&platforms);
// ..snip...
for (auto &p : platforms) {
</pre>
</td>

<td>
<pre lang="cpp">
&nbsp;
&nbsp;
&nbsp;
for (auto &p : opencl::platforms()) {
</pre>
</td>

</tr>

</table>
When we let go of the C-way of writing code and get rid of the out-parameter, we...

* don't have to declare a distinct variable.
* don't have to remember to use `std::vector`
* don't even have to care about the type `platforms()` uses.
* don't have to remember both the class name and the method name.
* save two lines of code and remain with a satisfying one-liner (that's not even verbose).

Let's continue: The program must now try and obtain a platform supporting OpenCL 2.0 or later.

<table>
<tr>
<th>
Khronos CLHPP
</th>
<th>
opencl-api-cpp
</th>
</tr>

<tr>

<td>
<pre lang="cpp">
cl::Platform plat;
for (auto &p : platforms) {
  std::string platver = p.getInfo<&zwnj;CL_PLATFORM_VERSION>();
  if (platver.find("OpenCL 2.") != std::string::npos ||
      platver.find("OpenCL 3.") != std::string::npos) {
      plat = p;
  }
}
if (plat() == 0) {
  std::cout << "No OpenCL 2.0 or newer platform found.\n";
  return -1;
}
</pre>
</td>

<td>
<pre lang="cpp">
&nbsp;
&nbsp;
auto platform = [] {
  auto is_acceptable = [](auto const & p) { return p.version().major >= 2; };
  auto platforms = opencl::platforms();
  auto iter = std::ranges::find_if(platforms, is_acceptable);
  (iter < platforms.end()) or die("No OpenCL 2.0 or newer platform found.");
  return *iter; 
}();
</pre>
</td>

</tr>

</table>

The `die()` function is just a simple hack (that's not part of this library):

```cpp
[[noreturn]] int die(std::string_view message) {
    std::cerr << message << '\n';
    exit(EXIT_FAILURE);
}
```
you'll also notice that:

* at no time does the `platform` variable ever hold an invalid value.
* we did not have to memorize any type names from our library. And when typing the code, the method and namespace names are rather intuitive; and we would get help with them if autocomplete is available.
* if we were to restrict ourselves to C++11 (which the library allows), the `find_if()` line would need the `begin()` and `end()` iterators; a little more verbose, but would work fine.
* (Not related to the choice of libraries) the code can be written without any loops and control statements: No `for`, `if`, or `while`. Remember [Sean Parent's 2013 C++ Seasonings talk](https://www.youtube.com/watch?v=W2tWOdzgXHA)?

Our last comparison of code segments regards building a program from a couple of kernel source code strings:

<table>
<tr>
<th>
Khronos CLHPP
</th>
<th>
opencl-api-cpp
</th>
</tr>

<tr>

<td>
<pre lang="cpp">
std::vector<&zwnj;std::string> programStrings;
programStrings.push_back(kernel1);
programStrings.push_back(kernel2);

cl::Program vectorAddProgram(programStrings);
try {
  vectorAddProgram.build("-cl-std=CL2.0");
}
catch (...) {
  // Print build info for all devices
  cl_int buildErr = CL_SUCCESS;
  auto buildInfo = vectorAddProgram
    .getBuildInfo<&zwnj;CL_PROGRAM_BUILD_LOG>(&buildErr);
  for (auto &pair : buildInfo) {
    std::cerr << pair.second << std::endl << std::endl;
  }
  return 1;
}
</pre>
</td>

<td>
<pre lang="cpp">
auto sources = { kernel1, kernel2 };
auto sources_ = opencl::program::source::create(context, sources);
auto options = opencl::program::compilation::options::create();
options.language_version = opencl::make_version("2.0");
auto build_result = opencl::program::build_(sources_, options);
if (not build_result.succeeded()) {
  std::cerr << "Build failed.\n";
  for (auto const& target_build_info : build_result.info()) {
    auto target = target_build_info.device();
    std::cerr << "Build log for device " << target.name() << ":\n\n";
    std::cerr << target_build_info.log()<< std::endl;
  }
  exit(EXIT_FAILURE);
}
</pre>
</td>

</tr>

</table>

Some points to note here:

* A build failure due to a compilation/linking error is not an exceptional situation, and does not merit throwing an exception; this is why the `build_result` object can describe either a successful build with resulting artifacts, or a failure, in which case one can loog into the reason through the object.
* All actually-exceptional situations are checked for and reported; this is unlike the example program, with which things get messy in case of `CL_OUT_OF_HOST_MEMORY` or `CL_COMPILER_NOT_AVAILABLE` or other such errors.
* The use of compiler options is more verbose than the CLHPP example program. However - you do not need to know the JIT compiler's command-line syntax and switch names. The compiler options are well-structured and you can query them and change them quite conveniently without resorting to string parsing.
* The example program iterates over the per-target-device build info, and we mirror this. Instead, one can loop over the targets - which are `context.devices()` when we don't specify targets explicitly; doing so, we could then obtain individual target build info with `build_result.info_for(target_device)`;
* CLHPP's buildInfo is a cryptic pair which is difficult to figure out; without our library - you just don't have to.
* No need to choose the "correct" container type and construct it explicitly (our code shows the use of `std::initializer_lists`'s).


## Example programs

More details about these will be added under the [examples](examples/) directory.

## Want to help? Report a bug? Give feedback?

* Noticed a bug, compatibility issue, missing functionality or other problem? Please [file the issue](https://github.com/eyalroz/opencl-api-cpp/issues) here on GitHub.
* Started using the library in a publicly-available project? Please let [me](https://github.com/eyalroz) know. <!-- * Want to help test new, improved versions? Please email [@eyalroz](https://github.com/eyalroz).-->
* Interested in collaborating on a FOSS project related to the library? Drop [me](https://github.com/eyalroz) a line.
