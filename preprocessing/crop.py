import cv2
import numpy as np


def crop_black_border(image, threshold=10):
    """
    Remove large black background regions surrounding
    a retinal fundus image.

    Parameters
    ----------
    image : numpy.ndarray
        RGB fundus image.

    threshold : int
        Pixels darker than this threshold are treated
        as background.

    Returns
    -------
    numpy.ndarray
        Cropped RGB image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mask = gray > threshold
    coords = np.argwhere(mask)

    if coords.size == 0:
        return image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]