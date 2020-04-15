from numba import njit
import numpy as np


@njit
def interpolation__(image, c, lf, uf, li, ui):
    cx, cy = c

    lx, ly = lf
    lxi, lyi = li

    if lx == cx and ly == cy:
        return image[lyi, lxi]

    ux, uy = uf
    uxi, uyi = ui

    if lx == cx:
        return (image[lyi, lxi] * (ux - cx) * (uy - cy) +
                image[uyi, lxi] * (ux - cx) * (cy - ly))

    if ly == cy:
        return (image[lyi, lxi] * (ux - cx) * (uy - cy) +
                image[lyi, uxi] * (cx - lx) * (uy - cy))

    return (image[lyi, lxi] * (ux - cx) * (uy - cy) +
            image[lyi, uxi] * (cx - lx) * (uy - cy) +
            image[uyi, lxi] * (ux - cx) * (cy - ly) +
            image[uyi, uxi] * (cx - lx) * (cy - ly))


@njit
def interpolation_numba(image, C):
    LF = np.floor(C)
    UF = LF + 1.0
    LI = LF.astype(np.int64)
    UI = UF.astype(np.int64)

    N = C.shape[0]
    intensities = np.empty(N)
    for i in range(N):
        intensities[i] = interpolation__(image, C[i],
                                         LF[i], UF[i], LI[i], UI[i])
    return intensities
