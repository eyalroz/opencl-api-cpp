#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

// You can have your kernel in a separate file, and just read it
// into a string buffer. That saves the annoying syntax here; but
// we want to see something similar to what we did with CUDA.

#define N 5000

int main(int, char**)
{
    cl_device_id device_id;         // compute device id
    // compute context
    // compute command queue
    // device memory used for the input array
    // device memory used for the input array
    // device memory used for the output array

    size_t length = 500000;

    printf("[Vector addition of %lu elements]\n", length);

   // -------------------------------------------

    // Allocate buffers on the host
    size_t size = length * sizeof(int);
    int *h_A = malloc(size);
    int *h_B = malloc(size);
    int *h_C = malloc(size);

    // Initialize the host input vectors
    for (size_t i = 1; i < length; ++i) {
        h_A[i] = rand();
        h_B[i] = rand();
    }

    cl_platform_id platforms[100]; // This should be enough...
    cl_uint num_platforms;
    clGetPlatformIDs(5,platforms,&num_platforms);

    cl_uint chosen_platform = num_platforms - 1; // naive guess that a GPU platform would follow a CPU one
    // printf("Using the last of %u platforms.\n", num_platforms);

    // Where to run
    clGetDeviceIDs(platforms[chosen_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue commands = clCreateCommandQueueWithProperties(context, device_id, NULL, NULL);

    // Create space for data and copy a and b to device (note that we could also use clEnqueueWriteBuffer to upload)
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, h_A, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, h_B, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);

    clFinish(commands);

    // Read result
    cl_event ev = NULL;
    clEnqueueReadBuffer( commands, d_C, CL_TRUE, 0, size, h_C, 0, NULL, &ev);
    printf("ev is %p\n", ev);
    cl_uint refcount = 123;
    clGetEventInfo(ev, CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &refcount, NULL);
    printf("refcount is %u\n", refcount);
    clWaitForEvents(1, &ev);
    clGetEventInfo(ev, CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &refcount, NULL);
    printf("refcount is now %u\n", refcount);


    // -------------------------------------------

   // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Clean up
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    printf("Done.\n");
    return EXIT_SUCCESS;
}
