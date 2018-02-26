"""
Tools to submit. No transforms, just the submission
"""
import numpy as np
from scipy.ndimage import label

def write_submission(fname, images, keys):
    print("Submitting for {} images".format(len(images)))
    f = open(fname, "w")
    f.write("ImageId,EncodedPixels\n")
    for i in range(len(images)):
        # Images with no nuclei found
        if len(images[i]) == 0:
            raise Exception("No nuclei found in image {}".format(i))
        for obj in _separate_masks(images[i]):
            f.write("{},{}\n".format(
                keys[i],
                " ".join([str(x) for x in _rle_encoding(obj)])
            ))
# from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def _rle_encoding(x):
    """
    x: numpy array of shape (height, width), nonzero - mask, 0 - background
    Returns run length as list
    """
    dots = np.where(x.T.flatten() !=0 )[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

# given a 2d array with many masks, split it out into its components
def _separate_masks(img):
    regions, n = label(img)
    masks = []
    for i in range(1, n+1):
        masks.append(regions == i)
    return masks
