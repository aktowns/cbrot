#ifndef MANDELBROT_H
#define MANDELBROT_H

#include "colourmaps/colourmaps.h"

void set_palette(colourmap_type choice);
const char* get_palette_name();

void mandelbrot(long double zoom, long double offsetX, long double offsetY, int width, int height,
                unsigned char* restrict image);

#endif
