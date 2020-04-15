#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "_bilinear.h"


float random_in_range(float min, float max) {
    float scale = rand() / (float) RAND_MAX;
    return min + scale * (max - min);
}


void generate_image(float *image, const long n_pixels) {
    for(int i = 0; i < n_pixels; i++) {
        image[i] = random_in_range(0.0, 1.0);
    }
}


void generate_coordinates(float *coordinates, const int n_coordinates,
                          const int image_width, const int image_height) {
    for(int i = 0; i < n_coordinates; i++) {
        coordinates[i*2+0] = random_in_range(0.0, image_width);   // x
        coordinates[i*2+1] = random_in_range(0.0, image_height);  // y
    }
}


long diff_in_nanosecond(const struct timespec *start,
                        const struct timespec *end) {
    long dsec = (end->tv_sec - start->tv_sec);
    long dnano = (end->tv_nsec - start->tv_nsec);
    return dsec * (long)1e9 + dnano;
}

void gettime(struct timespec *time) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, time);
}


int main() {
    srand(time(NULL));

    const int image_width = 800;
    const int image_height = 600;
    const int n_pixels = image_width * image_height;
    float image[n_pixels];
    generate_image(image, n_pixels);

    const long n_attempts = 100000;
    const long n_coordinates = 8000;
    float coordinates[n_coordinates*2];  // flattened representation
    generate_coordinates(coordinates, n_coordinates,
                         image_width, image_height);

    float intensities_simd[n_coordinates];
    float intensities_normal[n_coordinates];

    struct timespec start, end;

    printf("n attempts    = %ld\n", n_attempts);
    printf("n coordinates = %ld\n", n_coordinates);

    gettime(&start);
    for(int i = 0; i < n_attempts; i++) {
        _interpolation_simd(
            image, image_width, coordinates,
            n_coordinates, intensities_simd);
    }
    gettime(&end);
    printf("(simd)   time : %ld [ns]\n", diff_in_nanosecond(&start, &end));

    gettime(&start);
    for(int i = 0; i < n_attempts; i++) {
        _interpolation_normal(
            image, image_width, coordinates,
            n_coordinates, intensities_normal);
    }
    gettime(&end);
    printf("(normal) time : %ld [ns]\n", diff_in_nanosecond(&start, &end));
    return 0;
}
