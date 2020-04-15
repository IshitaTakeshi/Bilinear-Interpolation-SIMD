import numpy as np
from scipy.ndimage import map_coordinates
from interpolation import interpolation_simd


def main():
    height, width = 300, 400
    N = 64
    image = np.random.randint(0, 10, (height, width)).astype(np.float32)
    xs = np.random.uniform(0, width-1, N)
    ys = np.random.uniform(0, height-1, N)

    intensities_true = map_coordinates(image, np.vstack((ys, xs)), order=1)

    C = np.column_stack((xs, ys)).astype(np.float32)
    intensities_pred = interpolation_simd(image, C)

    assert(len(intensities_pred) == len(intensities_true))
    for i in range(len(intensities_true)):
        print("true {:.3f}   pred {:.3f}".format(
            intensities_true[i], intensities_pred[i]
        ))


main()
