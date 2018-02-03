import numpy as np
from scipy.ndimage import label

# from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), nonzero - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() !=0 )[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

# given a 2d array with many masks, split it out into its components
def separate_masks(img):
    regions, n = label(img)
    masks = []
    for i in range(1, n+1):
        masks.append(regions == i)
    return masks

def remove_tiny_masks(masks):
    cutoff = 4
    big_masks = []
    for m in masks:
        if np.count_nonzero(m) > cutoff:
            big_masks.append(m)
    return(big_masks)

def write_submission(fname, images, keys):
    print(len(images))
    f = open(fname, "w")
    f.write("ImageId,EncodedPixels\n")
    for i in range(len(images)):
        if len(images[i]) == 0:
            print(keys[i], i)
        for obj in images[i]:
            f.write("{},{}\n".format(
                keys[i],
                " ".join([str(x) for x in rle_encoding(obj)])
            ))
