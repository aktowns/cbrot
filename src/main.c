#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <time.h>

#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"

#include "input.h"
#include "mandelbrot.h"
#include "mandelbrot_opencl_kernels.h"
#include "render.h"

#define WINDOW_WIDTH 960
#define WINDOW_HEIGHT 540

// Global variables for zoom and offsets
double zoom = 1.0;
double offsetX = -0.75;
double offsetY = 0.0;

// Mouse state for panning
bool isMouseDragging = false;
double lastMouseX = 0, lastMouseY = 0;

double lastTime = 0.0;
int frameCount = 0;
double currentFPS = 0.0;
double targetFrameTime = 1.0 / 10.0; // Target 30 FPS

// Mouse scroll callback for zooming
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  // Zoom in or out depending on the scroll direction
  if (yoffset > 0) {
    zoom *= 1.1; // Zoom in (increase resolution)
  } else {
    zoom /= 1.1; // Zoom out (decrease resolution)
  }

  // Clamp zoom to a reasonable range to avoid excessive zooming
  if (zoom < 0.1)
    zoom = 0.1; // Minimum zoom limit
  if (zoom > 10000)
    zoom = 10000; // Maximum zoom limit
}

// Mouse button callback for panning
void mouse_button_callback(GLFWwindow *window, int button, int action,
                           int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    if (action == GLFW_PRESS) {
      isMouseDragging = true;
      glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
    } else if (action == GLFW_RELEASE) {
      isMouseDragging = false;
    }
  }
}

// Mouse motion callback for panning
void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos) {
  if (isMouseDragging) {
    double dx = xpos - lastMouseX;
    double dy = ypos - lastMouseY;

    // Base pan factor for adjusting sensitivity
    double basePanFactor = 0.01; // Base pan factor for small movements

    // Apply logarithmic scaling to pan factor based on zoom level
    // We reduce the pan factor more gradually at high zoom levels
    double zoomAdjustedFactor =
        basePanFactor / log(zoom * 100.0); // Logarithmic scaling with zoom

    // Clamp pan factor to avoid excessive movement or tiny shifts
    zoomAdjustedFactor = fmin(fmax(zoomAdjustedFactor, 0.0001),
                              0.0001); // Limit sensitivity range

    // Update offsets based on mouse movement with the adjusted sensitivity
    offsetX -= dx * zoomAdjustedFactor; // Horizontal pan
    offsetY += dy * zoomAdjustedFactor; // Vertical pan

    lastMouseX = xpos;
    lastMouseY = ypos;
  }
}

void drawText(const char *text, const float x, const float y) {
  char buffer[99999]; // ~500 chars

  glColor3f(1.0f, 1.0f, 1.0f); // White text

  glEnableClientState(GL_VERTEX_ARRAY);
  const int num_quads =
      stb_easy_font_print(x, y, (char *)text, NULL, buffer, sizeof(buffer));
  glVertexPointer(2, GL_FLOAT, 16, buffer);
  glDrawArrays(GL_QUADS, 0, num_quads * 4);
  glDisableClientState(GL_VERTEX_ARRAY);
}

int main() {
  opencl_kernel_manager_init();
  double frameDuration = 0.0;
  if (!glfwInit()) {
    return -1;
  }

  GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                                        "Mandelbrot Renderer", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);

  // Set up input callbacks
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetCursorPosCallback(window, cursor_pos_callback);

  glfwSwapInterval(1); // Enable vsync

  // Initialize OpenGL context (GLEW)
  if (glewInit() != GLEW_OK) {
    printf("GLEW initialization failed.\n");
    return -1;
  }

  // SuperSample setup
  GLuint fbo, highResTexture;
  const int scaleFactor = 2; // Render at 2x
  const int highW = WINDOW_WIDTH * scaleFactor;
  const int highH = WINDOW_HEIGHT * scaleFactor;

  glGenTextures(1, &highResTexture);
  glBindTexture(GL_TEXTURE_2D, highResTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, highW, highH, 0, GL_RGB,
               GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         highResTexture, 0);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    printf("Framebuffer not complete.\n");
    return -1;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // Main rendering loop
  unsigned char *image = malloc(highW * highH * 3);
  while (!glfwWindowShouldClose(window)) {
    const double currentTime =
        glfwGetTime(); // glfwGetTime() returns seconds as double
    frameCount++;

    // If 1 second has passed, update FPS
    if (currentTime - lastTime >= 1.0) {
      currentFPS = frameCount / (currentTime - lastTime);
      lastTime = currentTime;
      frameCount = 0;
    }

    // Poll for events and handle input
    processInput(window, &zoom, &offsetX, &offsetY);

    // Clear screen
    glClear(GL_COLOR_BUFFER_BIT);

    // Render Mandelbrot set, passing the window as an argument
    renderMandelbrot(window, zoom, offsetX, offsetY, image, highResTexture,
                     scaleFactor, fbo);

    char info[256];
    snprintf(info, sizeof(info),
             "Zoom: %.6e\nOffset: (%.6e, %.6e)\nFPS: %.1f\nFrame Time: %.2f "
             "ms\nPalette: %s",
             zoom, offsetX, offsetY, currentFPS, frameDuration * 1000.0,
             get_palette_name());

    // Set orthographic projection for 2D text
    int winW, winH;
    glfwGetFramebufferSize(window, &winW, &winH);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, winW, winH, 0, -1, 1); // Top-left = (0,0)
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Draw the text
    drawText(info, 10, 10);

    // Restore previous matrices
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();

    const double frameEndTime = glfwGetTime();
    frameDuration = frameEndTime - currentTime;

    if (frameDuration < targetFrameTime) {
      const double sleepTime = targetFrameTime - frameDuration;

      struct timespec ts;
      ts.tv_sec = (time_t)sleepTime;
      ts.tv_nsec = (long)((sleepTime - ts.tv_sec) * 1e9);
      nanosleep(&ts, NULL);
    }
  }
  free(image);

  glfwDestroyWindow(window);
  glfwTerminate();

  opencl_kernel_manager_shutdown();
  return 0;
}
