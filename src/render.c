#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mandelbrot.h"
#include "render.h"

void renderMandelbrot(GLFWwindow *window, const double zoom,
                      const double offsetX, const double offsetY,
                      unsigned char *restrict image,
                      const GLuint highResTexture, const int scaleFactor,
                      const GLuint fbo) {
  int winW, winH;
  glfwGetFramebufferSize(window, &winW, &winH);
  const int highW = winW * scaleFactor;
  const int highH = winH * scaleFactor;
  memset(image, 0, highW * highH * 3); // Clear the image buffer

  glBindTexture(GL_TEXTURE_2D, highResTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, highW, highH, 0, GL_RGB,
               GL_UNSIGNED_BYTE, NULL);

  // Generate the Mandelbrot set into the image buffer
  mandelbrot(zoom, offsetX, offsetY, highW, highH, image);

  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glViewport(0, 0, highW, highH);

  glClear(GL_COLOR_BUFFER_BIT);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, highW, highH, GL_RGB,
                  GL_UNSIGNED_BYTE, image);

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, highResTexture);

  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(-1.0f, -1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(1.0f, -1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(1.0f, 1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(-1.0f, 1.0f);
  glEnd();

  glDisable(GL_TEXTURE_2D);

  glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glBlitFramebuffer(0, 0, highW, highH, 0, 0, winW, winH, GL_COLOR_BUFFER_BIT,
                    GL_LINEAR);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, winW, winH);
}
