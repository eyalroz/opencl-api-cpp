#define CL_TARGET_OPENCL_VERSION 200

// TODO: Unify this with vector_addition.c

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <string.h>

const char* kernel_source_code =
  "__kernel void vectorAdd(\n"
  "    __global float       * __restrict C,\n"
  "    __global float const * __restrict A,\n"
  "    __global float const * __restrict B,\n"
  "    unsigned long length)\n"
  "{\n"
  "  int i = get_global_id(0);\n"
  "  if (i < length)\n"
  "    C[i] = A[i] + B[i];\n"
  "}\n";

void ensure_success(cl_int status, char const* action)
{
  if (status != CL_SUCCESS) {
    fprintf(stderr,"%s%sFailed with OpenCL error number %u\n",
      action ? action : "",
      action ? ": " : "",
      status);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[])
{
  (void) argc; (void) argv;

  cl_int status;

  cl_platform_id platform_id;
  status = clGetPlatformIDs(1, &platform_id, NULL);
  ensure_success(status, "clGetPlatformIDs");

  cl_device_id device_id;
  status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
  ensure_success(status, "clGetDeviceIDs");

  cl_context context_id = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
  ensure_success(status, "clCreateContext");

  unsigned long length = strlen(kernel_source_code);
  cl_program program_id = clCreateProgramWithSource(context_id, 1, &kernel_source_code, &length, &status);
  ensure_success(status, "clCreateProgramWithSource");

  status = clCompileProgram(
    program_id,
    1, &device_id, // single device
    "", // no special options
    0, NULL, NULL, // num headers, header sources, header names
    NULL, NULL // no callback
    );
  ensure_success(status, "clCompileProgram");

  cl_program linked = clLinkProgram(context_id,
    1, &device_id, // devices,
    "", // No link options
    1, &program_id,
    NULL, NULL, // no callback
    &status
    );
  ensure_success(status, "clLinkProgram");

  cl_kernel kernel = clCreateKernel(linked, "vectorAdd", &status);
  ensure_success(status, "clCreateKernel");
  (void) kernel;
  printf("SUCCESS\n");
  exit(EXIT_SUCCESS);
}
