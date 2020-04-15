cimport numpy as cnp
import numpy as np


cdef extern from "_bilinear.h":
    void _interpolation_simd(
        const float *image, const int image_width,
        const float *coordinates_x, const float *coordinates_y,
        const int n_coordinates, float *intensities);
    void _interpolation_normal(
        const float *image, const int image_width,
        const float *coordinates_x, const float *coordinates_y,
        const int n_coordinates, float *intensities);


def interpolation_simd(cnp.ndarray[cnp.float32_t, ndim=2] image,
                       cnp.ndarray[cnp.float32_t, ndim=2] coordinates):

    cdef float[:] image_view = image.flatten()
    cdef int width = image.shape[1]

    cdef float[:] xs = np.ascontiguousarray(coordinates[:, 0])
    cdef float[:] ys = np.ascontiguousarray(coordinates[:, 1])

    cdef int N = coordinates.shape[0]
    intensities = np.empty(N, dtype=np.float32)

    cdef float[:] intensities_view = intensities
    _interpolation_simd(&image_view[0], width, &xs[0], &ys[0], N,
                        &intensities_view[0])
    return intensities


def interpolation_normal(cnp.ndarray[cnp.float32_t, ndim=2] image,
                         cnp.ndarray[cnp.float32_t, ndim=2] coordinates):

    cdef float[:] image_view = image.flatten()
    cdef int width = image.shape[1]

    cdef float[:] xs = np.ascontiguousarray(coordinates[:, 0])
    cdef float[:] ys = np.ascontiguousarray(coordinates[:, 1])

    cdef int N = coordinates.shape[0]
    intensities = np.empty(N, dtype=np.float32)

    cdef float[:] intensities_view = intensities
    _interpolation_normal(&image_view[0], width, &xs[0], &ys[0], N,
                          &intensities_view[0])
    return intensities


def interpolation_cython(cnp.ndarray[cnp.float32_t, ndim=2] image,
                         cnp.ndarray[cnp.float32_t, ndim=2] coordinates):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] cx = coordinates[:, 0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] cy = coordinates[:, 1]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] lx = np.floor(cx)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] ly = np.floor(cy)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] ux = lx + 1.0
    cdef cnp.ndarray[cnp.float32_t, ndim=1] uy = ly + 1.0
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lxi = lx.astype(np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lyi = ly.astype(np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] uxi = ux.astype(np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] uyi = uy.astype(np.int64)
    return (image[lyi, lxi] * (ux - cx) * (uy - cy) +
            image[lyi, uxi] * (cx - lx) * (uy - cy) +
            image[uyi, lxi] * (ux - cx) * (cy - ly) +
            image[uyi, uxi] * (cx - lx) * (cy - ly))
