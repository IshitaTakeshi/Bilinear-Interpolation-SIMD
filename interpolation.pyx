cimport numpy as cnp
import numpy as np


cdef extern from "_bilinear.h":
    void interpolation_simd(const float *image, const int image_width,
                          const float *coordinates_x, const float *coordinates_y,
                          const int n_coordinates, float *intensities);


def interpolation(cnp.ndarray[cnp.float32_t, ndim=2] image,
                  cnp.ndarray[cnp.float32_t, ndim=2] coordinates):

    cdef float[:] image_view = image.flatten()
    cdef int width = image.shape[1]

    cdef float[:] xs = np.ascontiguousarray(coordinates[:, 0])
    cdef float[:] ys = np.ascontiguousarray(coordinates[:, 1])

    cdef int N = coordinates.shape[0]
    intensities = np.empty(N, dtype=np.float32)
    cdef float[:] intensities_view = intensities
    interpolation_simd(&image_view[0], width, &xs[0], &ys[0], N,
                     &intensities_view[0])
    return intensities
