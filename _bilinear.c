#include <math.h>
#include <assert.h>
#include <immintrin.h>
#include <xmmintrin.h>


float __interpolation_normal(const float *image, const int width,
                             const float cx, const float cy) {
    float lx = floor(cx);
    float ly = floor(cy);
    float ux = lx + 1.0;
    float uy = ly + 1.0;
    int lxi = (int)lx;
    int lyi = (int)ly;
    int uxi = (int)ux;
    int uyi = (int)uy;
    return (image[lyi * width + lxi] * (ux - cx) * (uy - cy) +
            image[lyi * width + uxi] * (cx - lx) * (uy - cy) +
            image[uyi * width + lxi] * (ux - cx) * (cy - ly) +
            image[uyi * width + uxi] * (cx - lx) * (cy - ly));
}


void _interpolation_normal(
    const float *restrict image, const int image_width,
    const float *restrict coordinates_x, const float *restrict coordinates_y,
    const int n_coordinates, float *restrict intensities) {
    for(int i = 0; i < n_coordinates; i++) {
        intensities[i] = __interpolation_normal(
            image, image_width, coordinates_x[i], coordinates_y[i]);
    }
}


__m256 __interpolation_simd(const float *image, const int image_width,
                            const __m256 cx, const __m256 cy) {
    __m256 lx = _mm256_round_ps(cx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    __m256 ly = _mm256_round_ps(cy, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    __m256 ux = _mm256_add_ps(lx, _mm256_set1_ps(1));
    __m256 uy = _mm256_add_ps(ly, _mm256_set1_ps(1));

    __m256 ucx = _mm256_sub_ps(ux, cx);
    __m256 ucy = _mm256_sub_ps(uy, cy);
    __m256 clx = _mm256_sub_ps(cx, lx);
    __m256 cly = _mm256_sub_ps(cy, ly);

    __m256i lxi = _mm256_cvtps_epi32(lx);
    __m256i lyi = _mm256_cvtps_epi32(ly);
    __m256i uxi = _mm256_cvtps_epi32(ux);
    __m256i uyi = _mm256_cvtps_epi32(uy);

    __m256i widths = _mm256_set1_epi32(image_width);
    __m256i wl = _mm256_mullo_epi32(widths, lyi);
    __m256i wu = _mm256_mullo_epi32(widths, uyi);
    __m256i lxlyi = _mm256_add_epi32(wl, lxi);
    __m256i uxlyi = _mm256_add_epi32(wl, uxi);
    __m256i lxuyi = _mm256_add_epi32(wu, lxi);
    __m256i uxuyi = _mm256_add_epi32(wu, uxi);

    __m256 intensities_lxly = _mm256_i32gather_ps(image, lxlyi, 4);
    __m256 intensities_uxly = _mm256_i32gather_ps(image, uxlyi, 4);
    __m256 intensities_lxuy = _mm256_i32gather_ps(image, lxuyi, 4);
    __m256 intensities_uxuy = _mm256_i32gather_ps(image, uxuyi, 4);

    __m256 ll = _mm256_mul_ps(intensities_lxly, _mm256_mul_ps(ucx, ucy));
    __m256 ul = _mm256_mul_ps(intensities_uxly, _mm256_mul_ps(clx, ucy));
    __m256 lu = _mm256_mul_ps(intensities_lxuy, _mm256_mul_ps(ucx, cly));
    __m256 uu = _mm256_mul_ps(intensities_uxuy, _mm256_mul_ps(clx, cly));

    return _mm256_add_ps(_mm256_add_ps(ll, ul), _mm256_add_ps(lu, uu));
}


const int N = 8;


void _interpolation_simd(
    const float *image, const int image_width,
    const float *coordinates_x, const float *coordinates_y,
    const int n_coordinates, float *intensities) {
    assert(n_coordinates % N == 0);

    // reversed when set
    // offsets[0] = 0, offsets[1] = 1, ..., offsets[7] = 7
    __m256i offsets = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    for(int i = 0; i < n_coordinates; i += N) {
        // indices = sliice(i, i+N)
        // xs = coordinates_x[indices]
        // ys = coordinates_y[indices]
        __m256i indices = _mm256_add_epi32(_mm256_set1_epi32(i), offsets);
        __m256 xs = _mm256_i32gather_ps(coordinates_x, indices, 4);
        __m256 ys = _mm256_i32gather_ps(coordinates_y, indices, 4);
        __m256 is = __interpolation_simd(image, image_width, xs, ys);
        _mm256_storeu_ps(&intensities[i], is);
    }
}
