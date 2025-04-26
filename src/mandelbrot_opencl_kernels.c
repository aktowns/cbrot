#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_program program = NULL;
static cl_device_id device = NULL;

void opencl_kernel_manager_init() {
  cl_platform_id platform;
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL);
  err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

  // Load source
  FILE *f = fopen("mandelbrot_kernel.cl", "rb");
  if (!f) {
    perror("Failed to open mandelbrot_opencl.cl");
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  size_t src_size = ftell(f);
  rewind(f);

  char *src = (char *)malloc(src_size + 1);
  fread(src, 1, src_size, f);
  src[src_size] = '\0';
  fclose(f);

  program =
      clCreateProgramWithSource(context, 1, (const char **)&src, NULL, &err);
  free(src);

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                          NULL);
    printf("CL Build Error:\n%s\n", log);
    free(log);
    exit(1);
  }
}

void opencl_kernel_manager_shutdown() {
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

void launch_opencl_kernel(const char *kernel_name, double zoom, double offsetX,
                          double offsetY, int width, int height,
                          int maxIterations, unsigned char *restrict image) {
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
  if (err != CL_SUCCESS) {
    printf("Failed to create kernel: %s\n", kernel_name);
    exit(1);
  }

  size_t img_size = width * height * 3;
  cl_mem output_buffer =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, img_size, NULL, &err);

  err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_buffer);
  err |= clSetKernelArg(kernel, 1, sizeof(double), &zoom);
  err |= clSetKernelArg(kernel, 2, sizeof(double), &offsetX);
  err |= clSetKernelArg(kernel, 3, sizeof(double), &offsetY);
  err |= clSetKernelArg(kernel, 4, sizeof(int), &width);
  err |= clSetKernelArg(kernel, 5, sizeof(int), &height);
  err |= clSetKernelArg(kernel, 6, sizeof(int), &maxIterations);

  const size_t global_size[2] = {(size_t)width, (size_t)height};
  err |= clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0,
                                NULL, NULL);

  err |= clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, img_size, image,
                             0, NULL, NULL);

  clReleaseMemObject(output_buffer);
  clReleaseKernel(kernel);

  clFinish(queue);
}
