import numpy as np
from scipy.ndimage import map_coordinates
from interpolation import interpolation_simd


def main():
    height, width = 300, 400
    N = 64
    image = np.random.randint(0, 100, (height, width)).astype(np.float32)
    xs = np.random.uniform(0, width-1, N)
    ys = np.random.uniform(0, height-1, N)
    C = np.column_stack((xs, ys)).astype(np.float32)
    intensities = interpolation_simd(image, C)
    print("interpolation result")
    print(intensities)
    print("ground truth")
    print(map_coordinates(image, np.vstack((ys, xs)), order=1))


main()
