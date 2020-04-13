// LICENSE: Apache 2.0
// Author : Takeshi Ishita
// How to build
// $ clang -g -Wall -mavx -mavx2 _bilinear.c -o bilinear

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>
#include <xmmintrin.h>


void print_float_array(const float *array, const int size) {
    for(int i = 0; i < size; i++) {
        printf("array[%d] = %f\n", i, array[i]);
    }
}


void print_m256(__m256 v) {
    float X[8];
    _mm256_store_ps(X, v);
    for(int i = 0; i < 8; i++) {
        printf("v[%d] = %f\n", i, X[i]);
    }
}


void print_m256i(__m256i v) {
    int *X = (int*)_mm_malloc(8 * sizeof(int), 32);
    _mm256_store_si256((__m256i *)X, v);
    for(int i = 0; i < 8; i++) {
        printf("v[%d] = %d\n", i, X[i]);
    }
    _mm_free(X);
}


float interpolation1d_(const float *image, const int width, float cx, float cy) {
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


__m256 __interpolation(const float *image, const int image_width,
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
    __m256i lxlyi = _mm256_add_epi32(_mm256_mullo_epi32(widths, lyi), lxi);
    __m256i uxlyi = _mm256_add_epi32(_mm256_mullo_epi32(widths, lyi), uxi);
    __m256i lxuyi = _mm256_add_epi32(_mm256_mullo_epi32(widths, uyi), lxi);
    __m256i uxuyi = _mm256_add_epi32(_mm256_mullo_epi32(widths, uyi), uxi);

    __m256 intensities_lxly = _mm256_i32gather_ps(image, lxlyi, 4);
    __m256 intensities_uxly = _mm256_i32gather_ps(image, uxlyi, 4);
    __m256 intensities_lxuy = _mm256_i32gather_ps(image, lxuyi, 4);
    __m256 intensities_uxuy = _mm256_i32gather_ps(image, uxuyi, 4);

    __m256 ll_uc_uc = _mm256_mul_ps(intensities_lxly, _mm256_mul_ps(ucx, ucy));
    __m256 ul_cl_uc = _mm256_mul_ps(intensities_uxly, _mm256_mul_ps(clx, ucy));
    __m256 lu_uc_cl = _mm256_mul_ps(intensities_lxuy, _mm256_mul_ps(ucx, cly));
    __m256 uu_cl_cl = _mm256_mul_ps(intensities_uxuy, _mm256_mul_ps(clx, cly));

    return _mm256_add_ps(
        _mm256_add_ps(ll_uc_uc, ul_cl_uc),
        _mm256_add_ps(lu_uc_cl, uu_cl_cl)
    );
}


const int N = 8;


void interpolation2d_(const float *image, const int image_width,
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
        __m256 is = __interpolation(image, image_width, xs, ys);
        _mm256_storeu_ps(&intensities[i], is);
    }
}


int main(void) {
    const float image[] = {
        2.0, 3.0, 2.0,
        1.0, 2.0, 4.0,
        5.0, 2.0, 3.0,
        4.0, 6.0, 3.0
    };

    const int image_width = 3;
    const float coordinates_x[] = {
        1.2, 0.0, 0.9, 0.3, 1.9, 0.5, 1.0, 0.4,
        0.1, 1.0, 0.0, 1.2, 0.9, 0.4, 0.8, 1.6
    };
    const float coordinates_y[] = {
        0.2, 1.2, 2.3, 2.9, 0.3, 0.4, 1.0, 2.0,
        2.0, 0.9, 0.2, 2.4, 0.0, 0.1, 1.2, 0.0
    };
    const int n_coordinates = 16;
    float intensities[n_coordinates];

    interpolation2d_(image, image_width,
                     coordinates_x, coordinates_y, n_coordinates,
                     intensities);
    for(int i = 0; i < n_coordinates; i++) {
        float intensity = interpolation1d_(image, image_width,
                                           coordinates_x[i], coordinates_y[i]);
        printf("simd   : intensities[%2d] = %f\n", i, intensities[i]);
        printf("normal : intensities[%2d] = %f\n", i, intensity);
    }
    return 0;
}
