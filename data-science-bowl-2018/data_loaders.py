import os
import numpy as np
import skimage.io

datadir = "/home/christopher/Data/data/ml/data-science-bowl-2018/"

def training_images():
    return get_images("train")

def test_images():
    return get_images("test")

def get_images(subdir):
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
        res.append(np.array(f_res))
    return res

# given masks returned from training_masks, returns them without any of the border masks
# useful for stats
def filter_border_masks(masks):
    res = []
    for image_masks in masks:
        f_res = []
        for mask in image_masks:
            for edge in [mask[0], mask[-1], mask[:,0], mask[:,-1]]:
                if not np.all(edge == 0):
                    break
            else: # no break
                f_res.append(np.copy(mask))
        res.append(np.array(f_res))
    return res

# create a single mask for each image
def single_masks(masks):
    res = []
    for image_masks in masks:
        res.append(np.sum(image_masks, axis=0))
    return res
