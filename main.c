#include <stdio.h>
#include "_bilinear.h"


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
    float intensities_simd[n_coordinates];
    float intensities_normal[n_coordinates];

    _interpolation_simd(
        image, image_width, coordinates_x, coordinates_y,
        n_coordinates, intensities_simd);
    _interpolation_normal(
        image, image_width, coordinates_x, coordinates_y,
        n_coordinates, intensities_normal);
    for(int i = 0; i < n_coordinates; i++) {
        printf("simd   : intensities[%2d] = %f\n", i, intensities_simd[i]);
        printf("normal : intensities[%2d] = %f\n", i, intensities_normal[i]);
    }
    return 0;
}
