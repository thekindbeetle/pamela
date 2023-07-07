"""
Двумерное преобразование Хафа с вариациаями
"""
import numpy as np


def hough_line(img, theta=None):
    """
    Переписываем вручную преобразование Хафа так, чтобы оно учитывало значения в пикселах.
    img - изображение
    theta - диапазон углов
    """
    # Compute the array of angles and their sine and cosine
    if theta is None:
        # These values are approximations of pi/2
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    offset = np.ceil(np.sqrt(img.shape[0] * img.shape[0] +
                             img.shape[1] * img.shape[1])).astype(int)

    max_distance = int(2 * offset + 1)
    accum = np.zeros((max_distance, theta.shape[0]), dtype=np.float64)
    bins = np.linspace(-offset, offset, max_distance)

    # compute the nonzero indexes
    y_idxs, x_idxs = np.nonzero(img)

    # finally, run the transform

    nidxs = y_idxs.shape[0]  # x and y are the same shape
    nthetas = theta.shape[0]

    for i in range(nidxs):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(nthetas):
            accum_idx = round((ctheta[j] * x + stheta[j] * y)) + offset
            accum[accum_idx, j] += img[y, x]

    return accum, theta, bins
