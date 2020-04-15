from timeit import timeit


n_attempts = 1000
n_coordinates = 10000


setup = """
import numpy as np
from skimage.data import coins
from interpolation_numba import interpolation_numba
from interpolation import (interpolation_cython,
                           interpolation_simd, interpolation_normal)
image = coins().astype(np.float32)
height, width = image.shape

n_coordinates = {}
xs = np.random.uniform(0, width-1, n_coordinates)
ys = np.random.uniform(0, height-1, n_coordinates)

coordinates = np.column_stack((xs, ys)).astype(np.float32)

# precompile jit
interpolation_numba(image, coordinates)
""".format(n_coordinates)


print("number of attempts    :", n_attempts)
print("number of coordinates :", n_coordinates)

time = timeit("interpolation_normal(image, coordinates)",
              setup=setup, number=n_attempts)
print("C for loop : {:8f} [s]".format(time))

time = timeit("interpolation_simd(image, coordinates)",
              setup=setup, number=n_attempts)
print("C SIMD     : {:8f} [s]".format(time))

time = timeit("interpolation_cython(image, coordinates)",
              setup=setup, number=n_attempts)
print("Cython     : {:8f} [s]".format(time))

time = timeit("interpolation_numba(image, coordinates)",
              setup=setup, number=n_attempts)
print("Numba      : {:8f} [s]".format(time))
