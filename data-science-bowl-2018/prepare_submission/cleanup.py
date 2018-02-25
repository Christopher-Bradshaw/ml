"""
Pre submission cleanup utilities
"""
import numpy as np
from scipy.ndimage import label

def cleanup(img, cutoff):
    img = (img > cutoff).astype(np.int)
    img = _separate_masks(img)
    img = _remove_tiny_masks(img)
    return img


# given a 2d array with many masks, split it out into its components
def _separate_masks(img):
    regions, n = label(img)
    masks = []
    for i in range(1, n+1):
        masks.append(regions == i)
    return masks

def _remove_tiny_masks(masks):
    cutoff = 13
    big_masks = []
    for m in masks:
        if np.count_nonzero(m) > cutoff:
            big_masks.append(m)
    return(big_masks)
