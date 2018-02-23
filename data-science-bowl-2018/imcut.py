import math
import torch



size = 256
def training_patchify(shape):
    height, width = shape

    num_rows = math.ceil(height / size)
    num_cols = math.ceil(width / size)

    x_vals, y_vals = [], []
    for i in range(num_cols-1):
        x_vals.append(size * i)
    x_vals.append(width - size)
    for i in range(num_rows-1):
        y_vals.append(size * i)
    y_vals.append(height - size)

    return y_vals, x_vals

# This is used for the test set. It works out how many cuts we need to make.
# Then it splits the image into equal sized pieces. We pad it out to full width, symmetrically.
#
# This is probably not ideal - we are losing some info. Better would be to always create 256x256 images
# And then when putting them back together, decide where to cut.
def patchify(shape):
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
    print(training_patchify(x.shape))
