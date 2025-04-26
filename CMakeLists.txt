cmake_minimum_required(VERSION 3.31)

project(MandelbrotRenderer)

set(CMAKE_C_STANDARD 11)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)  # Enables LTO

# Set the required libraries
find_package(PkgConfig)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)
if (OpenMP_C_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
    message(STATUS "OpenMP not found")
endif ()

include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})

# Path to your OpenCL kernel
set(OPENCL_KERNEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/mandelbrot_kernel.cl")

# Copy it into the binary directory automatically
configure_file(${OPENCL_KERNEL_FILE} ${CMAKE_CURRENT_BINARY_DIR}/mandelbrot_kernel.cl COPYONLY)

add_executable(MandelbrotRenderer src/main.c src/mandelbrot.c src/render.c src/input.c src/mandelbrot_opencl_kernels.c)
target_compile_options(MandelbrotRenderer PRIVATE
        # -g                            # Debugging information
        -O3                             # Optimize for speed
        -march=native                   # Use the best available instruction set for the host CPU
        -ffast-math                     # Looser IEEE compliance, allows math reordering and vectorization
        -ftree-vectorize                # Enable vectorization (SIMD)
        -funroll-loops                  # Unroll loops for better performance
        -flto                           # Link Time Optimization
        -fopenmp                        # Enable OpenMP for parallel processing
        -fomit-frame-pointer            # Omit frame pointers for better performance
        -mavx2                          # Enable AVX2 instructions\
        -mavx512f                       # Enable AVX512 instructions
        -mfma                           # Enable FMA (Fused Multiply-Add) instructions
        -mfpmath=sse                    # Use SSE for floating-point math
        -fvectorize                     # Enable vectorization
)
target_link_libraries(MandelbrotRenderer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} glfw OpenMP::OpenMP_C m OpenCL::OpenCL)
