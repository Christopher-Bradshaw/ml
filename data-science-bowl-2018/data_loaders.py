import os
import numpy as np
import skimage.io

datadir = "/home/christopher/Data/data/ml/data-science-bowl-2018/"
wdir = "saved_weights/"

def training_images():
    return _get_images("train")

def test_images():
    return _get_images("test")

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

def training_masks():
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

def save_weights(weights):
    try:
        os.mkdir(datadir + wdir)
    except FileExistsError:
        raise Exception("You have already saved weights. Go remove them before saving new ones!")

    np.save(datadir + wdir + "weights", weights)

def weights():
    return np.load(datadir + wdir + "weights.npy")
