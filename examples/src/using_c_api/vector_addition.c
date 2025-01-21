// TODO: Harmonize the kernel types and source code with vectorAdd.c
#define CL_TARGET_OPENCL_VERSION 200

// openCL headers
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SOURCE_SIZE (0x100000)

void ensure_success(cl_int status, char const* action)
{
	if (status != CL_SUCCESS) {
		fprintf(stderr,"%s%sFailed with OpenCL error number %d\n",
		  action ? action : "",
		  action ? ": " : "",
		  (int) status);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char * argv[])
{
	(void) argc; (void) argv;
	int size = 1024;

	// Allocate memories for input arrays and output array.
	float *A = malloc(sizeof(float)*size);
	float *B = malloc(sizeof(float)*size);

	// Output
	float *C = malloc(sizeof(float)*size);

	// Initialize values for array members.
	int i = 0;
	for (i=0; i<size; ++i) {
		A[i] = i+1;
		B[i] = (i+1)*2;
		C[i] = 1234.5678; // a dummy value to overwrite
	}

	char const *kernelSource =
		"__kernel void vectorAdd(                   \n"
		"   __global float* __restrict c,           \n"
		"   __global const  float* __restrict a,    \n"
		"   __global const  float* __restrict b,    \n"
		"   const int length)                       \n"
		"{                                          \n"
		"   int i = get_global_id(0);               \n"
		"   if(i < length) {                        \n"
		"       c[i] = a[i] + b[i];                 \n"
		"   }                                       \n"
		"}                                          \n"
	;

	// Getting platform and device information
	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);

	// Creating context.
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &ret);


	// Creating command queue
	cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceID, 0, &ret);

	// Memory buffers for each array
	cl_mem aMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float), NULL, &ret);
	cl_mem bMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float), NULL, &ret);
	cl_mem cMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), NULL, &ret);


	// Copy lists to memory buffers
	ret = clEnqueueWriteBuffer(commandQueue, aMemObj, CL_TRUE, 0, size * sizeof(float), A, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, bMemObj, CL_TRUE, 0, size * sizeof(float), B, 0, NULL, NULL);

	// Create program from kernel source
	size_t kernelSize = strlen(kernelSource);
	cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, &kernelSize, &ret);
	ensure_success(ret, "clCreateProgramWithSource");

	// Build program
	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

	if (ret == CL_BUILD_PROGRAM_FAILURE) {
		size_t build_log_size = 0;
		ret = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
		ensure_success(ret, "clGetProgramBuildInfo");
		char* buf = malloc(build_log_size + 1);
		ret = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, build_log_size, buf, NULL);
		ensure_success(ret, "clGetProgramBuildInfo");
		buf[build_log_size] = '\0';
		fprintf(stderr, "Build log:\n\n%s\n", buf);
		exit(EXIT_FAILURE);
	}
	ensure_success(ret, "clBuildProgram");

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "vectorAdd", &ret);
	ensure_success(ret, "clCreateKernel");


	// Set arguments for kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cMemObj);
	ensure_success(ret, "clSetKernelArg");
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &aMemObj);
	ensure_success(ret, "clSetKernelArg");
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bMemObj);
	ensure_success(ret, "clSetKernelArg");
	ret = clSetKernelArg(kernel, 3, sizeof(int), &size);
	ensure_success(ret, "clSetKernelArg");


	// Execute the kernel
	size_t globalItemSize = size;
	size_t localItemSize = 64; // globalItemSize has to be a multiple of localItemSize. 1024/64 = 16
	ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
	ensure_success(ret, "clEnqueueNDRangeKernel");

	// Read from device back to host.
	ret = clEnqueueReadBuffer(commandQueue, cMemObj, CL_TRUE, 0, size * sizeof(float), C, 0, NULL, NULL);
	ensure_success(ret, "clEnqueueReadBuffer");

	ret = clFinish(commandQueue);
	ensure_success(ret, "clFinish");

	fputs("All OpenCL actions complete!\n", stdout);
	fputs("Checking result correctness...\n", stdout);
	// Test if correct answer
	for (i=0; i<size; ++i) {
		if (C[i] != (A[i] + B[i])) {
			printf("Incorrect result at index %d: %f != %f + %f\n", i, C[i], A[i], B[i]);
			break;
		}
	}
	printf( (i == size) ? "SUCCESS\n" : "FAILURE\n");

	// Clean up, release memory.
	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(aMemObj);
	ret = clReleaseMemObject(bMemObj);
	ret = clReleaseMemObject(cMemObj);
	ret = clReleaseContext(context);
	free(A);
	free(B);
	free(C);

	return EXIT_SUCCESS;
}
