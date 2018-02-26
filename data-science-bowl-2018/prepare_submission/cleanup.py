"""
Pre submission cleanup utilities
"""
import numpy as np
# import cv2 Docs are hard to find/read?
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from skimage import filters



def cleanup(img):
    cutoff = filters.threshold_otsu(img)
    img = (img > cutoff).astype(np.uint8)
    img = _remove_tiny_regions(img)
    return img


def _remove_tiny_regions(img):
    # consider using circular kernels
    kernel1 = np.ones((3, 3)).astype(np.uint8)
    kernel2 = np.ones((3, 3)).astype(np.uint8)
    # Fill fully enclosed holes
    img = binary_fill_holes(img).astype(np.uint8)
    # Destroy small things
    img = binary_opening(img, kernel1).astype(np.uint8)
    # opening - erosion followed by dilation
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel1)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2, iterations=2)
    # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel1)


    return img
