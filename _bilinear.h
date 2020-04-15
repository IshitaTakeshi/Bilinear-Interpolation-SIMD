void interpolation_simd(
    const float *image, const int image_width,
    const float *coordinates_x, const float *coordinates_y,
    const int n_coordinates, float *intensities);

void interpolation_normal(
    const float *restrict image, const int image_width,
    const float *restrict coordinates_x, const float *restrict coordinates_y,
    const int n_coordinates, float *restrict intensities);
