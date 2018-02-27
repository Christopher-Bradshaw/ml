#!/usr/bin/env python3
"""
Script to take the segmented masks that are given and:
    * Create a single mask by ORing them together
    * Create a gap between different objects (that are connected)
    * Create a map of weights that encourages our code to learn the small gaps
    between nearby objects
Outputs this somewhere sane.
"""
import os
import numpy as np
import skimage.io
from scipy.ndimage import distance_transform_edt, label, binary_erosion

datadir = "/home/christopher/Data/data/ml/data-science-bowl-2018/"
weightsdir = datadir + "weights/"

def main():
    wdir = weightsdir + "enforce_separation/" # must end in a slash
    # wdir = weightsdir + "weights_test/" # must end in a slash
    try:
        os.listdir(wdir)
    except FileNotFoundError:
        pass
    else:
        raise Exception("weights dir {} exists".format(wdir))

    summary_masks, weights = weights_from_all_masks(load_masks())
    save_masks_and_weights(summary_masks, weights, wdir)

def load_masks():
    keys = os.listdir(datadir + "train")
    res = []
    for key in keys:
        f_dir = datadir + "train/" + key + "/masks/"
        fnames = [f_dir + fname for fname in os.listdir(f_dir)]
        f_res = []
        for img in skimage.io.imread_collection(fnames):
            assert len(img.shape) == 2
            assert img.shape[0] >= 256 and img.shape[1] >= 256
            f_res.append(img)
        res.append(np.array(f_res, dtype=np.bool))
    return res

# mask_list is a list( np.array(mask1, mask2), np.array(mask1, mask2), ... )
def weights_from_all_masks(mask_list):
    summary_masks, weights = [], []
    for (i, masks) in enumerate(mask_list):
        s, w = weights_from_mask(masks)
        summary_masks.append(s)
        weights.append(w)
        if i % 10 == 0:
            print(i)
    return summary_masks, weights

# masks is a np.array( np.array(mask1), np.array(mask2), ... ) with type bool
def weights_from_mask(masks):
    sigma, w_two, w_one = 5, 5, 5
    width, height = len(masks[0]), len(masks[0][0])

    seg_summary_mask = get_segmented_summary_mask(masks)

    # weight nuceli vs empty space
    empty_weight = 1
    nuclei_weight = 10

    # get distances from each pixel to the nearest true region. Remove one true region each
    # time to get two distances
    labelled_summary_mask, n_features = label(seg_summary_mask)
    distances = np.zeros((n_features, width, height))
    for feature in range(n_features):
        distances[feature] = distance_transform_edt(
                ((labelled_summary_mask == 0) | (labelled_summary_mask == feature + 1))
        )
    distances = np.rollaxis(distances, 0, 3)

    # compute weights
    weights = np.zeros(seg_summary_mask.shape, dtype=np.float)
    for i in range(width):
        for j in range(height):
            if seg_summary_mask[i][j] != 0:
                weights[i][j] = nuclei_weight
            else:
                d = np.sort(np.unique(distances[i][j]))
                assert (len(d) in (1, 2))
                weights[i][j] = (empty_weight +
                    w_two * nuclei_weight * np.exp( -((d[0]+d[-1])**2) / (2 * sigma**2) ) +
                    w_one * nuclei_weight * np.exp( -(d[0]**2) / (sigma**2) ))
    return seg_summary_mask, weights

def get_segmented_summary_mask(masks):
    res = np.zeros(masks[0].shape)
    kernel = np.ones((3, 3)).astype(np.uint8)
    for i, mask in enumerate(masks):
        res += binary_erosion(mask, kernel)
    return res.astype(np.bool)

def save_masks_and_weights(summary_masks, weights, wdir):
    try:
        os.mkdir(wdir)
    except FileExistsError:
        raise Exception("You have already saved weights. Go remove them before saving new ones!")

    np.save(wdir + "weights", weights)
    np.save(wdir + "summary_masks", summary_masks)

if __name__ == "__main__":
    main()
