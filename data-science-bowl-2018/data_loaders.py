import os
import numpy as np
import skimage.io

datadir = "/home/christopher/Data/data/ml/data-science-bowl-2018/"
weightsdir = datadir + "weights/"

def training_images():
    return _get_images("train")

def test_images():
    return _get_images("test")

def masks_and_weights(wdir):
    masks = np.load(weightsdir + wdir + "/summary_masks.npy")
    masks = [i.astype(np.uint8) for i in masks]
    weights = np.load(weightsdir + wdir + "/weights.npy")
    weights = [i.astype(np.uint8) for i in weights]
    return masks, weights

def _get_images(subdir):
    if subdir not in set(["train", "test"]):
        raise Exception("You probably want test or train images, not {}".format(subdir))
    keys = os.listdir(datadir + subdir)
    fnames = [datadir + subdir + "/" + key + "/images/" + key + ".png" for key in keys]
    res = []
    for img in skimage.io.imread_collection(fnames):
        assert len(img.shape) == 3
        assert img.shape[0] >= 160 and img.shape[1] >= 160
        # Could add some assertions here that A is boring (if it exists)
        assert img.shape[2] in set([3, 4]) # RGB(A)

        img = np.rollaxis(img, 2, 0)
        img = np.mean(img[:3], axis=0).astype(np.uint8)
        res.append(img)
    return res, keys
