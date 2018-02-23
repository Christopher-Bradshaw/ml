import math
import torch

# import matplotlib.pyplot
# given an image of uncertain dimensions
# we want to split it into 256x256 (ideally) image segments, with 46x46 (348-256)//2 padding
# idea: provide a list of (startx, starty, len, height). Then we just pad out to 348
# Knowing the weight + height and that we symmetrically padded, we can get back the original image
def patchify(shape):
    size = 256

    height, width = shape

    num_rows = math.ceil(height / size)
    num_cols = math.ceil(width / size)

    x_vals, y_vals = [], []
    width_step = width / num_cols
    height_step = height / num_rows
    for i in range(num_cols+1):
        x_vals.append(width_step * i)
    for i in range(num_rows+1):
        y_vals.append(height_step * i)

    x_vals = norm(x_vals)
    y_vals = norm(y_vals)

    return y_vals, x_vals



# The differences all need to be multiples of 2
def norm(vals):
    n = []
    n.append(vals[0])
    for i in range(len(vals) - 1):
        diff = int(vals[i+1] - n[i])
        if diff % 2 == 0:
            n.append(n[i] + diff)
            continue
        else:
            # -1 to solve the case where it is an odd number - just cut the final pixel
            n.append(n[i] + diff - 1)
    return [int(i) for i in n]


if __name__ == "__main__":
    x = torch.randn(256, 550)
    print(patchify(x.shape))
