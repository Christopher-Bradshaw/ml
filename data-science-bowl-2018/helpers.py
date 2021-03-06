import numpy as np
import torch

# takes a normal list of np.arrays and returns a 4d variables
def tensorfy(x):
    xv = torch.from_numpy(np.array(x).astype(np.float32))
    xv.unsqueeze_(1)
    return xv

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
