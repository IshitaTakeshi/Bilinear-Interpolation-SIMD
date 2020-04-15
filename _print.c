#include <stdio.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include "_print.h"


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

