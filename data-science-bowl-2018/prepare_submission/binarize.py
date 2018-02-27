"""
Take our image of 0-1 floats and binarize it
"""
import numpy as np
from skimage import filters


def binarize(img, cutoff=None):
    if cutoff is None:
        try:
            cutoff = filters.threshold_otsu(img)
        except ValueError:
            print("otsu failure - needs multi colored image")
            return (img > 1).astype(np.uint8)
    img = (img > cutoff).astype(np.uint8)
    if cutoff  < 0.2:
        print("low cutoff: {}".format(cutoff))
    return img
