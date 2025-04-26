#include "input.h"
#include "mandelbrot.h"
#include <GLFW/glfw3.h>

void processInput(GLFWwindow *window, double *zoom, double *offsetX,
                  double *offsetY) {
  if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS) {
    set_palette(VIRIDIS_LUT_COLOR);
  }
  if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
    set_palette(CUBEHELIX_LUT_COLOR);
  }
  if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
    set_palette(HSV_LUT_COLOR);
  }
  if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
    set_palette(INFERNO_LUT_COLOR);
  }
  if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
    set_palette(MAGMA_LUT_COLOR);
  }
  if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
    set_palette(PARULA_LUT_COLOR);
  }
  if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
    set_palette(PASTEL_RAINBOW_LUT_COLOR);
  }
  if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS) {
    set_palette(PLASMA_LUT_COLOR);
  }
  if (glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS) {
    set_palette(TURBO_LUT_COLOR);
  }
  if (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS) {
    set_palette(CIVIDIS_LUT_COLOR);
  }

  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
  if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
    *offsetY += 0.11 / *zoom;
  }
  if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
    *offsetY -= 0.11 / *zoom;
  }
  if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
    *offsetX -= 0.11 / *zoom;
  }
  if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
    *offsetX += 0.11 / *zoom;
  }
  if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
    *zoom = 1.0;
    *offsetX = -0.75;
    *offsetY = 0.0;
  }

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    *zoom *= 1.07;
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    *zoom /= 1.08;
  }
}
