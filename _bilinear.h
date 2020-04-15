void _interpolation_simd(
    const float *restrict image, const int image_width,
    const float *restrict coordinates, const int n_coordinates,
    float *restrict intensities);

void _interpolation_normal(
    const float *restrict image, const int image_width,
    const float *restrict coordinates, const int n_coordinates,
    float *restrict intensities);
