#ifndef RENDER_H
#define RENDER_H

#include <GLFW/glfw3.h>

// Function to render the Mandelbrot set
void renderMandelbrot(GLFWwindow* window, const double zoom, const double offsetX, const double offsetY,
                      unsigned char* image, GLuint highResTexture, int scaleFactor, GLuint fbo);

#endif
