from skimage import data
import numpy as np
from scipy.ndimage import map_coordinates
from interpolation import interpolation


def main():
    image = data.coins().astype(np.float32)
    height, width = image.shape
    N = 64
    xs = np.random.uniform(0, height-1, N)
    ys = np.random.uniform(0, width-1, N)
    C = np.column_stack((xs, ys)).astype(np.float32)
    intensities = interpolation(image, C)
    print("interpolation result")
    print(intensities)
    print("ground truth")
    print(map_coordinates(image, np.vstack((ys, xs)), order=1))


main()
