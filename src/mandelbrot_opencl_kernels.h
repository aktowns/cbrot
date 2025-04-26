#ifndef MANDELBROT_OPENCL_KERNELS
#define MANDELBROT_OPENCL_KERNELS

void opencl_kernel_manager_init();
void opencl_kernel_manager_shutdown();

void launch_opencl_kernel(const char* kernel_name,
                          double zoom, double offsetX, double offsetY,
                          int width, int height, int maxIterations,
                          unsigned char* restrict image);

#endif
