#图像可视化模块
from __future__ import annotations
import numpy as np
import scipy.sparse
from PIL import Image
import cv2
import seaborn as sns
import skimage.segmentation
from .image_utils import load_image, normalize

#用于在图像上可视化分割标签。可以选择裁剪图像、归一化、填充标签颜色以及绘制边框。
def view_labels(image_path, labels_npz_path,
                crop=None,
                stardist_normalize=False,
                fill=False,
                border=True,
                fill_palette=None,
                fill_label_weight=0.5,
                border_color=[255, 255, 0]
                ):

    # load the sparse labels
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    # determine memory efficient dtype to load the image as
    # if we'll be normalising, we want np.float16 for optimal RAM footprint
    # otherwise use np.uint8
    if stardist_normalize:
        dtype = np.float16
    else:
        dtype = np.uint8
    if crop is None:
        # this will load greyscale as 3 channel, which is what we want here
        img = load_image(image_path, dtype=dtype)
    else:
        # PIL is better at handling crops memory efficiently than cv2
        img = Image.open(image_path)
        # ensure that it's in RGB (otherwise there's a single channel for greyscale)
        img = np.array(img.crop(crop).convert('RGB'), dtype=dtype)
        # subset labels to area of interest
        # crop is (left, upper, right, lower)
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
        # upper:lower, left:right
        labels_sparse = labels_sparse[crop[1]:crop[3], crop[0]:crop[2]]
    # optionally normalise image
    if stardist_normalize:
        img = normalize(img)
        # actually cap the values - currently there are sub 0 and above 1 entries
        img[img < 0] = 0
        img[img > 1] = 1
        # turn back to uint8 for internal consistency
        img = (255 * img).astype(np.uint8)
    # turn labels to COO for ease of position retrieval
    labels_sparse = labels_sparse.tocoo()
    if fill:
        if fill_palette is None:
            # use the seaborn bright palette, but remove the pink from it
            # as it blends too much into the H&E background
            fill_palette = (np.array(sns.color_palette("bright")) * 255).astype(np.uint8)
            fill_palette = np.delete(fill_palette, 6, 0)
        # now we have a master list of pixels with objects to show
        # .row is [:,0], .col is [:,1]
        # extract the existing values from the image
        # and simultaneously get a fill colour by doing % on number of fill colours
        # weight the two together to get the new pixel value
        img[labels_sparse.row, labels_sparse.col, :] = \
            (1 - fill_label_weight) * img[labels_sparse.row, labels_sparse.col, :] + \
            fill_label_weight * fill_palette[labels_sparse.data % fill_palette.shape[0], :]
    if border:
        # unfortunately the boundary finder wants a dense matrix, so turn our labels to it for a sec
        # turn the output back into a sparse COO, both for memory efficiency and pure convenience
        border_sparse = scipy.sparse.coo_matrix(skimage.segmentation.find_boundaries(np.array(labels_sparse.todense())))
        # can now easily colour the borders similar to what was done for the fill
        img[border_sparse.row, border_sparse.col, :] = border_color
    return img

print("visualization_utils.py loaded")