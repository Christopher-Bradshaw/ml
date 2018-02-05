"""
Set of helpers to make our masks useful to u net
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, label
# from multiprocessing import Pool
# import tqdm

# mask_list is a list( np.array(mask1, mask2), np.array(mask1, mask2), ... )
def weights_from_all_masks(mask_list):
    summary_masks, weights = [], []
    # with Pool(8) as p:
    #     for v in tqdm.tqdm(p.map(weights_from_mask, mask_list), total=len(mask_list)):
    #         summary_masks.append(v[0])
    #         weights.append(v[1])
    for (i, masks) in enumerate(mask_list):
        s, w = weights_from_mask(masks)
        summary_masks.append(s)
        weights.append(w)
        if i % 10 == 0:
            print(i)
    return summary_masks, weights

# masks is a np.array( np.array(mask1), np.array(mask2), ... ) with type bool
def weights_from_mask(masks):
    sigma, w0 = 5, 10
    width, height = len(masks[0]), len(masks[0][0])

    seg_summary_mask = get_segmented_summary_mask(masks)

    # weight nuceli vs empty space
    empty_weight = 1
    nuclei_weight = empty_weight * np.size(seg_summary_mask) / np.count_nonzero(seg_summary_mask)

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
                d = np.unique(distances[i][j])
                assert (len(d) in (1, 2))
                weights[i][j] = empty_weight + w0 * np.exp( -((d[0]+d[-1])**2) / (2 * sigma**2) )
    return seg_summary_mask, weights

def get_segmented_summary_mask(masks):
    res = np.zeros(masks[0].shape)
    for i, mask in enumerate(masks):
        res += (i+1)*mask
    width, height = len(res), len(res[0])

    for i in range(width):
        for j in range(height):
            p = res[i][j]
            if p == 0:
                continue
            for (x, y) in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if not ((0 < x < width) and (0 < y < height)):
                    continue
                if res[x][y] != 0 and res[x][y] != p:
                    res[i][j] = 0
    return res.astype(np.bool)
