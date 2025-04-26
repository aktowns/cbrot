__kernel void mandelbrot_kernel(
    __global uchar *output,
    const double zoom,
    const double offsetX,
    const double offsetY,
    const int width,
    const int height,
    const int maxIterations
)
{
    const int px = get_global_id(0);
    const int py = get_global_id(1);

    if (px >= width || py >= height)
        return;

    const double scale = 3.0 / zoom;
    const double real0 = ((double)px / width - 0.5) * scale + offsetX;
    const double imag0 = ((double)py / height - 0.5) * scale + offsetY;

    double real = real0;
    double imag = imag0;

    int iteration = 0;
    while (real*real + imag*imag <= 4.0 && iteration < maxIterations)
    {
        double realTemp = real * real - imag * imag + real0;
        imag = 2.0 * real * imag + imag0;
        real = realTemp;
        iteration++;
    }

    const int idx = (py * width + px) * 3;

    if (iteration == maxIterations) {
        output[idx + 0] = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 0;
    } else {
        float t = (float)iteration / (float)maxIterations;
        output[idx + 0] = (uchar)(9.0f * (1.0f - t) * t * t * t * 255.0f);
        output[idx + 1] = (uchar)(15.0f * (1.0f - t) * (1.0f - t) * t * t * 255.0f);
        output[idx + 2] = (uchar)(8.5f * (1.0f - t) * (1.0f - t) * (1.0f - t) * t * 255.0f);
    }
}

// Double Double
typedef struct {
    double hi;
    double lo;
} dd_real;

inline dd_real dd_add(dd_real a, dd_real b)
{
    double s = a.hi + b.hi;
    double v = s - a.hi;
    double t = ((b.hi - v) + (a.hi - (s - v))) + a.lo + b.lo;
    dd_real result;
    result.hi = s + t;
    result.lo = t - (result.hi - s);
    return result;
}

inline dd_real dd_sub(dd_real a, dd_real b)
{
    double s = a.hi - b.hi;
    double v = s - a.hi;
    double t = ((-b.hi - v) + (a.hi - (s - v))) + a.lo - b.lo;
    dd_real result;
    result.hi = s + t;
    result.lo = t - (result.hi - s);
    return result;
}

inline dd_real dd_mul(dd_real a, dd_real b)
{
    const double split = 134217729.0; // 2^27 + 1

    double cona = a.hi * split;
    double conb = b.hi * split;
    double a1 = cona - (cona - a.hi);
    double b1 = conb - (conb - b.hi);
    double a2 = a.hi - a1;
    double b2 = b.hi - b1;

    double c11 = a.hi * b.hi;
    double c21 = a2 * b2 - (((c11 - a1 * b1) - a2 * b1) - a1 * b2);

    double c2 = a.hi * b.lo + a.lo * b.hi;

    double t1 = c11 + c2;
    double e = t1 - c11;
    double t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 + a.lo * b.lo;

    dd_real result;
    result.hi = t1 + t2;
    result.lo = t2 - (result.hi - t1);
    return result;
}

inline dd_real dd_sqr(dd_real a)
{
    return dd_mul(a, a);
}

inline double dd_mag2(dd_real a, dd_real b)
{
    double r = a.hi * a.hi + b.hi * b.hi;
    return r;
}

__kernel void mandelbrot_kernel_dd(
    __global uchar *output,
    const double zoom,
    const double offsetX,
    const double offsetY,
    const int width,
    const int height,
    const int maxIterations
)
{
    const int px = get_global_id(0);
    const int py = get_global_id(1);

    if (px >= width || py >= height)
        return;

    const double scale = 3.0 / zoom;
    const double centerReal = offsetX;
    const double centerImag = offsetY;

    dd_real real0;
    real0.hi = ((double)px / width - 0.5) * scale + centerReal;
    real0.lo = 0.0;

    dd_real imag0;
    imag0.hi = ((double)py / height - 0.5) * scale + centerImag;
    imag0.lo = 0.0;

    dd_real real = real0;
    dd_real imag = imag0;

    int iteration = 0;
    while (dd_mag2(real, imag) <= 4.0 && iteration < maxIterations)
    {
        dd_real realSq = dd_sqr(real);
        dd_real imagSq = dd_sqr(imag);
        dd_real realTemp = dd_add(dd_sub(realSq, imagSq), real0);

        dd_real two;
        two.hi = 2.0;
        two.lo = 0.0;

        dd_real realImag = dd_mul(real, imag);
        imag = dd_add(dd_mul(two, realImag), imag0);

        real = realTemp;

        iteration++;
    }

    const int idx = (py * width + px) * 3;

    if (iteration == maxIterations) {
        output[idx + 0] = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 0;
    } else {
        float t = (float)iteration / (float)maxIterations;
        output[idx + 0] = (uchar)(9.0f * (1.0f - t) * t * t * t * 255.0f);
        output[idx + 1] = (uchar)(15.0f * (1.0f - t) * (1.0f - t) * t * t * 255.0f);
        output[idx + 2] = (uchar)(8.5f * (1.0f - t) * (1.0f - t) * (1.0f - t) * t * 255.0f);
    }
}
