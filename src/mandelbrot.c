#include <immintrin.h>
#include <math.h>
#include <stdio.h>

#include "mandelbrot.h"

#include "colourmaps/colourmaps.h"
#include "mandelbrot_opencl_kernels.h"

#define MAX_ITERATIONS 10000 // Increase iterations for more detail
#define VEC_SIZE 8           // AVX2 vector size

static double colorShift = 0.0;
static const unsigned char (*activePalette)[256][3] = &viridis;
static colourmap_type currentPalette = VIRIDIS_LUT_COLOR;

const char *get_palette_name() {
  switch (currentPalette) {
  case VIRIDIS_LUT_COLOR:
    return (const char *)viridis_name;
  case CUBEHELIX_LUT_COLOR:
    return (const char *)cubehelix_name;
  case CIVIDIS_LUT_COLOR:
    return (const char *)cividis_name;
  case HSV_LUT_COLOR:
    return (const char *)hsv_name;
  case INFERNO_LUT_COLOR:
    return (const char *)inferno_name;
  case MAGMA_LUT_COLOR:
    return (const char *)magma_name;
  case PARULA_LUT_COLOR:
    return (const char *)parula_name;
  case PASTEL_RAINBOW_LUT_COLOR:
    return (const char *)pastel_rainbow_name;
  case PLASMA_LUT_COLOR:
    return (const char *)plasma_name;
  case TURBO_LUT_COLOR:
    return (const char *)turbo_name;
  }
}

void set_palette(const colourmap_type choice) {
  currentPalette = choice;
  switch (choice) {
  case CIVIDIS_LUT_COLOR:
    activePalette = &cividis;
    break;
  case CUBEHELIX_LUT_COLOR:
    activePalette = &cubehelix;
    break;
  case HSV_LUT_COLOR:
    activePalette = &hsv;
    break;
  case INFERNO_LUT_COLOR:
    activePalette = &inferno;
    break;
  case MAGMA_LUT_COLOR:
    activePalette = &magma;
    break;
  case PARULA_LUT_COLOR:
    activePalette = &parula;
    break;
  case PASTEL_RAINBOW_LUT_COLOR:
    activePalette = &pastel_rainbow;
    break;
  case PLASMA_LUT_COLOR:
    activePalette = &plasma;
    break;
  case TURBO_LUT_COLOR:
    activePalette = &turbo;
    break;
  case VIRIDIS_LUT_COLOR:
    activePalette = &viridis;
    break;
  default:
    activePalette = &viridis;
    break;
  }
}

void mandelbrot_long_double(const long double zoom, const long double offsetX,
                            const long double offsetY, const int width,
                            const int height, const int maxIterations,
                            unsigned char *restrict image) {
  const long double scale = 3.0L / zoom;
  const long double invW = 1.0L / width;
  const long double invH = 1.0L / height;
  const long double baseReal = -0.5L * scale + offsetX;
  const long double baseImag = -0.5L * scale + offsetY;

#pragma omp parallel for schedule(dynamic)
  for (int py = 0; py < height; py++) {
    long double imagPart = py * invH * scale + baseImag;

#pragma omp simd
    for (int px = 0; px < width; px++) {
      long double realPart = px * invW * scale + baseReal;

      long double real = realPart;
      long double imag = imagPart;

      int iteration = 0;
      while (iteration < maxIterations) {
        long double realSquared = real * real;
        long double imagSquared = imag * imag;
        if (realSquared + imagSquared > 4.0L)
          break;

        long double tempReal = realSquared - imagSquared + realPart;
        imag = 2.0L * real * imag + imagPart;
        real = tempReal;

        iteration++;
      }

      int i = (py * width + px) * 3;
      if (iteration == maxIterations) {
        image[i + 0] = 0;
        image[i + 1] = 0;
        image[i + 2] = 0;
      } else {
        image[i + 0] = (*activePalette)[iteration][0];
        image[i + 1] = (*activePalette)[iteration][1];
        image[i + 2] = (*activePalette)[iteration][2];
      }
    }
  }
}

static inline __m256d fast_log2_avx2(__m256d x) {
  __m256i xi = _mm256_castpd_si256(x);
  xi = _mm256_srli_epi64(xi, 52);
  __m256d exp = _mm256_cvtepi64_pd(xi);
  exp = _mm256_sub_pd(exp, _mm256_set1_pd(1023.0));
  return exp;
}

void mandelbrot_avx2(const double zoom, const double offsetX,
                     const double offsetY, const int width, const int height,
                     const int maxIterations, unsigned char *restrict image) {
  const double scale = 3.0 / zoom;
  const double invW = 1.0 / width;
  const double invH = 1.0 / height;
  const double scaledInvW = invW * scale;
  const double offsetReal = offsetX - 0.5 * scale;

#pragma omp parallel for
  for (int py = 0; py < height; py++) {
    const double y0 = ((double)py * invH - 0.5) * scale + offsetY;

    for (int px = 0; px < width; px += VEC_SIZE) {
      __m256d x_pixel = _mm256_set_pd(px + 3, px + 2, px + 1, px);
      __m256d x0 = _mm256_fmadd_pd(x_pixel, _mm256_set1_pd(scaledInvW),
                                   _mm256_set1_pd(offsetReal));
      __m256d y0v = _mm256_set1_pd(y0);

      __m256d x = x0;
      __m256d y = y0v;
      __m256d iter = _mm256_setzero_pd();

      const __m256d two = _mm256_set1_pd(2.0);
      const __m256d four = _mm256_set1_pd(4.0);

      for (int i = 0; i < maxIterations; i++) {
        __m256d x2 = _mm256_mul_pd(x, x);
        __m256d y2 = _mm256_mul_pd(y, y);
        __m256d xy = _mm256_mul_pd(x, y);

        __m256d mag2 = _mm256_add_pd(x2, y2);
        __m256d mask = _mm256_cmp_pd(mag2, four, _CMP_LT_OQ);

        if (_mm256_movemask_pd(mask) == 0)
          break;

        x = _mm256_add_pd(_mm256_sub_pd(x2, y2), x0);
        y = _mm256_fmadd_pd(two, xy, y0v);

        iter = _mm256_add_pd(iter, _mm256_and_pd(mask, _mm256_set1_pd(1.0)));
      }

      __m256d final_iter = iter;
      __m256d x2 = _mm256_mul_pd(x, x);
      __m256d y2 = _mm256_mul_pd(y, y);
      __m256d zn = _mm256_add_pd(x2, y2);

      __m256d log2_zn = fast_log2_avx2(zn);
      __m256d log_zn = _mm256_mul_pd(log2_zn, _mm256_set1_pd(log(2.0) / 2.0));
      __m256d log2_logzn = fast_log2_avx2(log_zn);
      __m256d nu = _mm256_div_pd(log2_logzn, _mm256_set1_pd(log(2.0)));

      __m256d smooth =
          _mm256_sub_pd(_mm256_add_pd(final_iter, _mm256_set1_pd(1.0)), nu);
      __m256d t = _mm256_div_pd(smooth, _mm256_set1_pd((double)maxIterations));

      double smooth_vals[VEC_SIZE];
      double iter_vals[VEC_SIZE];
      _mm256_storeu_pd(smooth_vals, t);
      _mm256_storeu_pd(iter_vals, final_iter);

      for (int i = 0; i < VEC_SIZE; i++) {
        int offset = ((py * width) + (px + i)) * 3;

        if ((int)iter_vals[i] >= maxIterations) {
          image[offset + 0] = 0;
          image[offset + 1] = 0;
          image[offset + 2] = 0;
        } else {
          double t_shifted = fmod(smooth_vals[i] + colorShift, 1.0);
          int idx = (int)(t_shifted * 255.0);
          if (idx < 0)
            idx = 0;
          if (idx > 255)
            idx = 255;

          image[offset + 0] = viridis[idx][0];
          image[offset + 1] = viridis[idx][1];
          image[offset + 2] = viridis[idx][2];
        }
      }
    }
  }
}

__m512d fast_log2_avx512(__m512d x) {
  __m512i xi = _mm512_castpd_si512(x);
  xi = _mm512_srli_epi64(xi, 52);
  __m512d exp = _mm512_cvtepi64_pd(xi);
  exp = _mm512_sub_pd(exp, _mm512_set1_pd(1023.0));
  return exp;
}

void mandelbrot_avx512(const double zoom, const double offsetX,
                       const double offsetY, const int width, const int height,
                       const int maxIterations, unsigned char *restrict image) {
  const double scale = 3.0 / zoom;
  const double invW = 1.0 / width;
  const double invH = 1.0 / height;
  const double scaledInvW = invW * scale;
  const double offsetReal = offsetX - 0.5 * scale;

#pragma omp parallel for
  for (int py = 0; py < height; py++) {
    const double y0 = ((double)py * invH - 0.5) * scale + offsetY;

    for (int px = 0; px < width; px += VEC_SIZE) {
      const __m512d x_pixel = _mm512_set_pd(px + 7, px + 6, px + 5, px + 4,
                                            px + 3, px + 2, px + 1, px);
      const __m512d x0 = _mm512_fmadd_pd(x_pixel, _mm512_set1_pd(scaledInvW),
                                         _mm512_set1_pd(offsetReal));
      const __m512d y0v = _mm512_set1_pd(y0);

      __m512d x = x0;
      __m512d y = y0v;
      __m512d iter = _mm512_setzero_pd();

      const __m512d two = _mm512_set1_pd(2.0);
      const __m512d four = _mm512_set1_pd(4.0);

      for (int i = 0; i < maxIterations; i++) {
        const __m512d x2 = _mm512_mul_pd(x, x);
        const __m512d y2 = _mm512_mul_pd(y, y);
        const __m512d xy = _mm512_mul_pd(x, y);

        const __m512d mag2 = _mm512_add_pd(x2, y2);
        const __mmask8 mask = _mm512_cmp_pd_mask(mag2, four, _CMP_LT_OQ);

        if (mask == 0)
          break;

        x = _mm512_add_pd(_mm512_sub_pd(x2, y2), x0);
        y = _mm512_fmadd_pd(two, xy, y0v);

        iter = _mm512_mask_add_pd(iter, mask, iter, _mm512_set1_pd(1.0));
      }

      const __m512d final_iter = iter;
      const __m512d x2 = _mm512_mul_pd(x, x);
      const __m512d y2 = _mm512_mul_pd(y, y);
      const __m512d zn = _mm512_add_pd(x2, y2);

      const __m512d log2_zn = fast_log2_avx512(zn);
      const __m512d log_zn =
          _mm512_mul_pd(log2_zn, _mm512_set1_pd(log(2.0) / 2.0));
      const __m512d log2_logzn = fast_log2_avx512(log_zn);
      const __m512d nu = _mm512_div_pd(log2_logzn, _mm512_set1_pd(log(2.0)));

      const __m512d smooth =
          _mm512_sub_pd(_mm512_add_pd(final_iter, _mm512_set1_pd(1.0)), nu);
      const __m512d t =
          _mm512_div_pd(smooth, _mm512_set1_pd((double)maxIterations));

      double smooth_vals[VEC_SIZE];
      double iter_vals[VEC_SIZE];
      _mm512_storeu_pd(smooth_vals, t);
      _mm512_storeu_pd(iter_vals, final_iter);

      for (int i = 0; i < VEC_SIZE; i++) {
        const int offset = ((py * width) + (px + i)) * 3;

        if ((int)iter_vals[i] >= maxIterations) {
          image[offset + 0] = 0;
          image[offset + 1] = 0;
          image[offset + 2] = 0;
        } else {
          const double t_shifted = fmod(smooth_vals[i] + colorShift, 1.0);
          const double ft = t_shifted * 255.0;
          int idx = (int)ft;
          const double frac = ft - idx;
          if (idx < 0)
            idx = 0;
          if (idx >= 255)
            idx = 254;

          for (int ch = 0; ch < 3; ++ch) {
            image[offset + ch] =
                (unsigned char)((1.0 - frac) * (*activePalette)[idx][ch] +
                                frac * (*activePalette)[idx + 1][ch]);
          }
        }
      }
    }
  }
}

int computeMaxIterations(const long double zoom) {
  const int baseIterations = 150;
  const double growthRate = 150.0; // How fast it grows

  int dynamicMax = (int)(baseIterations + log2l(zoom) * growthRate);

  // Clamp to reasonable bounds
  if (dynamicMax < baseIterations)
    dynamicMax = baseIterations;
  if (dynamicMax > MAX_ITERATIONS)
    dynamicMax = MAX_ITERATIONS;

  return dynamicMax;
}

// OpenCL standard (double) Mandelbrot call
void mandelbrot_opencl_standard(const double zoom, const double offsetX,
                                const double offsetY, const int width,
                                const int height, int maxIterations,
                                unsigned char *restrict image) {
  launch_opencl_kernel("mandelbrot_kernel", zoom, offsetX, offsetY, width,
                       height, maxIterations, image);
}

// OpenCL double-double Mandelbrot call
void mandelbrot_opencl_double_double(const double zoom, const double offsetX,
                                     const double offsetY, const int width,
                                     const int height, int maxIterations,
                                     unsigned char *restrict image) {
  launch_opencl_kernel("mandelbrot_kernel_dd", zoom, offsetX, offsetY, width,
                       height, maxIterations, image);
}

void mandelbrot(const long double zoom, const long double offsetX,
                const long double offsetY, const int width, const int height,
                unsigned char *restrict image) {
  colorShift += 0.0005; // Slow shift
  if (colorShift > 0.9)
    colorShift -= 0.9;

  const int dynamicIterations = computeMaxIterations(zoom);

#ifdef USE_OPENCL
  if (zoom < 1e12)
    mandelbrot_opencl_standard(zoom, offsetX, offsetY, width, height,
                               dynamicIterations, image);
  else
    mandelbrot_opencl_double_double(zoom, offsetX, offsetY, width, height,
                                    dynamicIterations, image);
#else
  if (zoom < 1e14)
    mandelbrot_avx512(zoom, offsetX, offsetY, width, height, dynamicIterations,
                      image);
  else
    mandelbrot_long_double(zoom, offsetX, offsetY, width, height,
                           dynamicIterations, image);
#endif
}
