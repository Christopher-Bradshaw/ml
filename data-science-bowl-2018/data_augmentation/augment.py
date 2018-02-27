"""
Contains funcs to augment our data
"""

import numpy as np
import torchvision.transforms as tr
import functools
from PIL import Image
from scipy.ndimage.interpolation import zoom

# taks multiple images (train, mask, weight) and returns [ (t, m, w)*, (t, m, w)^, ...]
# where *, ^ etc mean that the images have been slightly transformed
def augment(img):
    _transforms = [
            _compose(_rt[i], _zt[j]) for i in range(len(_rt)) for j in range(len(_zt))
    ]
    return [t(img) for t in _transforms]

def _compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def _color_transform(brightness, contrast, saturation, hue):
    def x(img):
        return np.array(
                tr.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue,
                )(Image.fromarray(img))
        )
    return x

def _zoom(img, z_factor):
    return np.array(
            tr.CenterCrop(256)(Image.fromarray(zoom(img, z_factor, order=1)))
            # tr.CenterCrop(256)(Image.fromarray(img))
    )

# rotate transforms
_rt = (
        np.fliplr,
        np.flipud,
        _compose(np.fliplr, np.flipud), # equivalent to 180 deg rot
        np.rot90,
        lambda img: np.rot90(img, -1),
)

# zoom transforms
_zt = (
        # lambda img: img,
        lambda img: _zoom(img, 1.1),
        # lambda img: _zoom(img, 2),
)

# color transforms
_ct = (
        _color_transform(0, 0, 0, 0), # null
        _color_transform(1, 1, 1, 0),
)

