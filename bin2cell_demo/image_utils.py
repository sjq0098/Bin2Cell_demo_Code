from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
import skimage.filters
import os
import seaborn as sns
import scipy
import tifffile as tf
#本脚本负责图像输入输出、图像预处理功能
from .coordinate_utils import actual_vs_inferred_image_shape, get_crop, mpp_to_scalef,check_array_coordinates

Image.MAX_IMAGE_PIXELS = None
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
#加载图像
def load_image(image_path, gray=False, dtype=np.uint8):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(dtype, copy=False)

#标准化图像的对比度
def normalize(img):
    eps = 1e-20
    mi = np.percentile(img, 3)
    ma = np.percentile(img, 99.8)
    return ((img - mi) / (ma - mi + eps))

#创建一个调整后的 H&E（Hematoxylin and Eosin）组织图像
def scaled_he_image(adata, mpp=1, crop=True, buffer=150, spatial_cropped_key=None, store=True, img_key=None,
                    save_path=None):
    # identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    # retrieve specified source image path and load it
    img = load_image(adata.uns['spatial'][library]['metadata']['source_image_path'])
    # assess that the image dimensions match what they're supposed to be
    # if not, inform the user what image they should retrieve and use
    actual_vs_inferred_image_shape(adata, img)
    # crop image if necessary
    if crop:
        crop_coords = get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=buffer)
        # this is already capped at a minimum of 0, so can just subset freely
        # left, upper, right, lower; image is up-down, left-right
        img = img[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2], :]
        # set up the spatial cropped key if one is not passed
        if spatial_cropped_key is None:
            spatial_cropped_key = "spatial_cropped_" + str(buffer) + "_buffer"
        # need to move spatial so it starts at the new crop top left point
        # spatial[:,1] is up-down, spatial[:,0] is left-right
        adata.obsm[spatial_cropped_key] = adata.obsm["spatial"].copy()
        adata.obsm[spatial_cropped_key][:, 0] -= crop_coords[0]
        adata.obsm[spatial_cropped_key][:, 1] -= crop_coords[1]
        # print off the spatial cropped key just in case
        print("Cropped spatial coordinates key: " + spatial_cropped_key)
    # reshape image to desired microns per pixel
    # get necessary scale factor for the custom mpp
    # multiply dimensions by this to get the shrunken image size
    # multiply .obsm['spatial'] by this to get coordinates matching the image
    scalef = mpp_to_scalef(adata, mpp=mpp)
    # need to reverse dimension order and turn to int for cv2
    dim = (np.array(img.shape[:2]) * scalef).astype(int)[::-1]
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # we have everything we need. store in object
    if store:
        if img_key is None:
            img_key = str(mpp) + "_mpp"
            if crop:
                img_key = img_key + "_" + str(buffer) + "_buffer"
        adata.uns['spatial'][library]['images'][img_key] = img
        # the scale factor needs to be prefaced with "tissue_"
        adata.uns['spatial'][library]['scalefactors']['tissue_' + img_key + "_scalef"] = scalef
        # print off the image key just in case
        print("Image key: " + img_key)
    if save_path is not None:
        # cv2 expects BGR channel order, we're working with RGB
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


#根据指定的微米每像素（microns per pixel, mpp）比例，调整和处理免疫荧光（IF）图像
def scaled_if_image(adata, channel, mpp=1, crop=True, buffer=150, spatial_cropped_key=None, store=True, img_key=None,
                    save_path=None):

    # identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    # pull out specified channel from IF tiff via tifffile
    # pretype to float32 for space while working with plots (float16 does not)
    img = tf.imread(adata.uns['spatial'][library]['metadata']['source_image_path'], key=channel).astype(np.float32)
    # assess that the image dimensions match what they're supposed to be
    # if not, inform the user what image they should retrieve and use
    actual_vs_inferred_image_shape(adata, img)
    # this can be dark, apply stardist normalisation to fix
    img = normalize(img)
    # actually cap the values - currently there are sub 0 and above 1 entries
    img[img < 0] = 0
    img[img > 1] = 1
    # crop image if necessary
    if crop:
        crop_coords = get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=buffer)
        # this is already capped at a minimum of 0, so can just subset freely
        # left, upper, right, lower; image is up-down, left-right
        img = img[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]
        # set up the spatial cropped key if one is not passed
        if spatial_cropped_key is None:
            spatial_cropped_key = "spatial_cropped_" + str(buffer) + "_buffer"
        # need to move spatial so it starts at the new crop top left point
        # spatial[:,1] is up-down, spatial[:,0] is left-right
        adata.obsm[spatial_cropped_key] = adata.obsm["spatial"].copy()
        adata.obsm[spatial_cropped_key][:, 0] -= crop_coords[0]
        adata.obsm[spatial_cropped_key][:, 1] -= crop_coords[1]
        # print off the spatial cropped key just in case
        print("Cropped spatial coordinates key: " + spatial_cropped_key)
    # reshape image to desired microns per pixel
    # get necessary scale factor for the custom mpp
    # multiply dimensions by this to get the shrunken image size
    # multiply .obsm['spatial'] by this to get coordinates matching the image
    scalef = mpp_to_scalef(adata, mpp=mpp)
    # need to reverse dimension order and turn to int for cv2
    dim = (np.array(img.shape[:2]) * scalef).astype(int)[::-1]
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # we have everything we need. store in object
    if store:
        if img_key is None:
            img_key = str(mpp) + "_mpp"
            if crop:
                img_key = img_key + "_" + str(buffer) + "_buffer"
        adata.uns['spatial'][library]['images'][img_key] = img
        # the scale factor needs to be prefaced with "tissue_"
        adata.uns['spatial'][library]['scalefactors']['tissue_' + img_key + "_scalef"] = scalef
        # print off the image key just in case
        print("Image key: " + img_key)
    if save_path is not None:
        # cv2 expects BGR channel order, we have a greyscale image
        # oh also we should make it a uint8 as otherwise stuff won't work
        cv2.imwrite(save_path, cv2.cvtColor((255 * img).astype(np.uint8), cv2.COLOR_GRAY2BGR))


#在数组坐标网格上创建指定 val 的图像。
def grid_image(adata, val, log1p=False, mpp=2, sigma=None, save_path=None):
    # pull out the values for the image. start by checking .obs
    if val in adata.obs.columns:
        vals = adata.obs[val].values.copy()
    elif val in adata.var_names:
        # if not in obs, it's presumably in the feature space
        vals = adata[:, val].X
        # may be sparse
        if scipy.sparse.issparse(vals):
            vals = vals.todense()
        # turn it to a flattened numpy array so it plays nice
        vals = np.asarray(vals).flatten()
    else:
        # failed to find
        raise ValueError('"' + val + '" not located in ``.obs`` or ``.var_names``')
    # make the values span from 0 to 255
    vals = (255 * (vals - np.min(vals)) / (np.max(vals) - np.min(vals))).astype(np.uint8)
    # optionally log1p
    if log1p:
        vals = np.log1p(vals)
        vals = (255 * (vals - np.min(vals)) / (np.max(vals) - np.min(vals))).astype(np.uint8)
    # spatial coordinates match what's going on in the image, array coordinates may not
    # have we checked if the array row/col need flipping?
    if not "bin2cell" in adata.uns:
        check_array_coordinates(adata)
    elif not "array_check" in adata.uns["bin2cell"]:
        check_array_coordinates(adata)
    # can now create an empty image the shape of the grid and stick the values in based on the coordinates
    # need to nudge up the dimensions by 1 as python is zero-indexed
    img = np.zeros((adata.uns["bin2cell"]["array_check"]["row"]["max"] + 1,
                    adata.uns["bin2cell"]["array_check"]["col"]["max"] + 1),
                   dtype=np.uint8)
    img[adata.obs['array_row'], adata.obs['array_col']] = vals
    # check if the row or column need flipping
    if adata.uns["bin2cell"]["array_check"]["row"]["flipped"]:
        img = np.flip(img, axis=0)
    if adata.uns["bin2cell"]["array_check"]["col"]["flipped"]:
        img = np.flip(img, axis=1)
    # resize image to appropriate mpp. bins are 2um apart, so current mpp is 2
    # need to reverse dimensions relative to the array for cv2, and turn to int
    if mpp != 2:
        dim = np.round(np.array(img.shape) * 2 / mpp).astype(int)[::-1]
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    # run through the gaussian filter if need be
    if sigma is not None:
        img = skimage.filters.gaussian(img, sigma=sigma)
        img = (255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.uint8)
    # save or return image
    if save_path is not None:
        cv2.imwrite(save_path, img)
    else:
        return img

print("image_utils.py loaded")