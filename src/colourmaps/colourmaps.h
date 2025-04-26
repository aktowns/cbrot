#ifndef COLOURMAPS_H
#define COLOURMAPS_H

#include "cividis_lut.h"
#include "cubehelix_lut.h"
#include "hsv_lut.h"
#include "inferno_lut.h"
#include "magma_lut.h"
#include "parula_lut.h"
#include "pastel_rainbow_lut.h"
#include "plasma_lut.h"
#include "turbo_lut.h"
#include "viridis_lut.h"

typedef enum
{
    CIVIDIS_LUT_COLOR,
    CUBEHELIX_LUT_COLOR,
    HSV_LUT_COLOR,
    INFERNO_LUT_COLOR,
    MAGMA_LUT_COLOR,
    PARULA_LUT_COLOR,
    PASTEL_RAINBOW_LUT_COLOR,
    PLASMA_LUT_COLOR,
    TURBO_LUT_COLOR,
    VIRIDIS_LUT_COLOR
} colourmap_type;

#endif // COLOURMAPS_H
