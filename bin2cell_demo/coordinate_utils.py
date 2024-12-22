#本脚本负责坐标系相关操作
import numpy as np
import scipy.stats
#验证加载的高分辨率（hires）组织形态图像的实际尺寸是否与从 AnnData 对象中推断出的尺寸相匹配
def actual_vs_inferred_image_shape(adata, img, ratio_threshold=0.99):
    # identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    # infer the dimensions as the shape of the hires tissue image
    # divided by the hires scale factor
    inferred_dim = np.array(adata.uns['spatial'][library]['images']['hires'].shape)[:2] / \
                   adata.uns['spatial'][library]['scalefactors']['tissue_hires_scalef']
    # retrieve actual dimension as we have the full morphology image loaded
    actual_dim = np.array(img.shape)[:2]
    # do the two match, within some tolerance of rounding etc?
    # divide both ways just in case
    if np.min(np.hstack((actual_dim / inferred_dim, inferred_dim / actual_dim))) < ratio_threshold:
        raise ValueError("Morphology image dimension mismatch. Dimensions inferred from Spaceranger output: " + str(
            inferred_dim) + ", actual image dimensions: " + str(
            actual_dim) + ". Are you running with ``source_image_path`` set to the full resolution morphology image, as used for ``--image`` in Spaceranger?")

#根据给定的微米每像素（microns per pixel，简称 mpp）和指定的坐标基准（basis），从 AnnData 对象中提取并转换空间坐标
def get_mpp_coords(adata, basis="spatial", spatial_key="spatial", mpp=None):
    # if we're using array coordinates, is there an mpp provided?
    if basis == "array" and mpp is None:
        raise ValueError("Need to specify mpp if working with array coordinates.")
    if basis == "spatial":
        if mpp is not None:
            # get necessary scale factor
            scalef = mpp_to_scalef(adata, mpp=mpp)
        else:
            # no mpp implies full blown morphology image, so scalef is 1
            scalef = 1
        # get the matching coordinates, rounding to integers makes this agree
        # need to reverse them here to make the coordinates match the image, as per note at start
        # multiply by the scale factor to account for possible custom mpp morphology image
        coords = (adata.obsm[spatial_key] * scalef).astype(int)[:, ::-1]
    elif basis == "array":
        # generate the pixels in the GEX image at the specified mpp
        # which actually correspond to the locations of the bins
        # easy to define scale factor as starting array mpp is 2
        scalef = 2 / mpp
        coords = np.round(adata.obs[['array_row', 'array_col']].values * scalef).astype(int)
        # need to flip axes maybe
        # need to scale up maximum appropriately
        if adata.uns["bin2cell"]["array_check"]["row"]["flipped"]:
            coords[:, 0] = np.round(adata.uns["bin2cell"]["array_check"]["row"]["max"] * scalef).astype(int) - coords[:,
                                                                                                               0]
        if adata.uns["bin2cell"]["array_check"]["col"]["flipped"]:
            coords[:, 1] = np.round(adata.uns["bin2cell"]["array_check"]["col"]["max"] * scalef).astype(int) - coords[:,
                                                                                                               1]
    return coords

#根据给定的空间坐标信息，计算出适合用于图像裁剪的边界框
def get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=0):
    # get the appropriate coordinates, be they spatial or array, at appropriate mpp
    coords = get_mpp_coords(adata, basis=basis, spatial_key=spatial_key, mpp=mpp)
    # PIL crop is defined as a tuple of (left, upper, right, lower) coordinates
    # coords[:,0] is up-down, coords[:,1] is left-right
    # don't forget to add/remove buffer, and to not go past 0
    return (np.max([np.min(coords[:, 1]) - buffer, 0]),
            np.max([np.min(coords[:, 0]) - buffer, 0]),
            np.max(coords[:, 1]) + buffer,
            np.max(coords[:, 0]) + buffer
            )


#根据指定的新每像素微米数（microns per pixel，简称 mpp），计算出一个缩放因子（scale factor）
def mpp_to_scalef(adata, mpp):
    # identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    # get original image mpp value
    mpp_source = adata.uns['spatial'][library]['scalefactors']['microns_per_pixel']
    # our scale factor is the original mpp divided by the new mpp
    return mpp_source / mpp

#验证和记录单细胞空间数据中数组坐标（行和列）与实际空间坐标（空间轴）的相关性
def check_array_coordinates(adata, row_max=3349, col_max=3349):
    # store the calls here
    if not "bin2cell" in adata.uns:
        adata.uns["bin2cell"] = {}
    adata.uns["bin2cell"]["array_check"] = {}
    # we'll need to check both the rows and columns
    for axis in ["row", "col"]:
        # we may as well store the maximum immediately
        adata.uns["bin2cell"]["array_check"][axis] = {}
        if axis == "row":
            adata.uns["bin2cell"]["array_check"][axis]["max"] = row_max
        elif axis == "col":
            adata.uns["bin2cell"]["array_check"][axis]["max"] = col_max
        # are we going to be extracting values for a single col or row?
        # set up where we'll be looking to get values to correlate
        if axis == "col":
            single_axis = "row"
            # spatial[:,0] matches axis_col (note at start)
            spatial_axis = 0
        elif axis == "row":
            single_axis = "col"
            # spatial[:,1] matches axis_row (note at start)
            spatial_axis = 1
        # get the value of the other axis with the highest number of bins present
        val = adata.obs['array_' + single_axis].value_counts().index[0]
        # get a boolean mask of the bins of that value
        mask = (adata.obs['array_' + single_axis] == val)
        # use the mask to get the spatial and array coordinates to compare
        array_vals = adata.obs.loc[mask, 'array_' + axis].values
        spatial_vals = adata.obsm['spatial'][mask, spatial_axis]
        # check whether they're positively or negatively correlated
        if scipy.stats.pearsonr(array_vals, spatial_vals)[0] < 0:
            adata.uns["bin2cell"]["array_check"][axis]["flipped"] = True
        else:
            adata.uns["bin2cell"]["array_check"][axis]["flipped"] = False

print("coordinate_utils.py loaded")