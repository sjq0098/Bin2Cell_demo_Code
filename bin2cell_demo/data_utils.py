
from __future__ import annotations
import json
from pathlib import Path, PurePath
from typing import BinaryIO, Literal
import pandas as pd
from anndata import AnnData
from matplotlib.image import imread
from scanpy import read_10x_h5

#actual bin2cell dependencies start here
#the ones above are for read_visium()
from stardist.plot import render_label
from copy import deepcopy
import skimage.segmentation
import tifffile as tf
import seaborn as sns
import scipy.spatial
import scipy.sparse
import scipy.stats
import anndata as ad
import skimage
import scanpy as sc
import numpy as np
import os
import logging
from imageio import imread
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# 创建一个日志记录器
logg = logging.getLogger(__name__)
# 设置日志级别（根据需要调整，例如 DEBUG, INFO, WARNING, ERROR, CRITICAL）
logging.basicConfig(level=logging.INFO)

#
def read_visium(
    path: Path | str,
    genome: str | None = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool | None = True,
    source_image_path: Path | str | None = None,
    spaceranger_image_path: Path | str | None = None,
) -> AnnData:
    """\
    Read 10x-Genomics-formatted visum dataset.

    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.

    See :func:`~scanpy.pl.spatial` for a compatible plotting function.

    .. _Space Ranger output docs: https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        Path to directory for visium datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    source_image_path
        Path to the high-resolution tissue image. Path will be included in
        `.uns["spatial"][library_id]["metadata"]["source_image_path"]`.
    spaceranger_image_path
        Path to the folder containing the spaceranger output hires/lowres tissue images. If `None`,
        will go with the `spatial` folder of the provided `path`.

    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:

    :param source_image_path:
    :param spaceranger_image_path:
    :param count_file:
    :param library_id:
    :param load_images:
    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names for a feature barcode matrix, probe names for a probe bc matrix
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    :attr:`~anndata.AnnData.obs`\\ `[filtered_barcodes]`
        filtered barcodes if present in the matrix
    :attr:`~anndata.AnnData.var`
        Any additional metadata present in /matrix/features is read in.
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of spaceranger output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['images']`
        Dict of images (`'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version', 'source_image_path'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """
    path = Path(path)
    #if not provided, assume the hires/lowres images are in the same folder as everything
    #except in the spatial subdirectory
    if spaceranger_image_path is None:
        spaceranger_image_path = path / "spatial"
    else:
        spaceranger_image_path = Path(spaceranger_image_path)
    adata = read_10x_h5(path / count_file, genome=genome)

    adata.uns["spatial"] = dict()

    from h5py import File

    with File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)
    if library_id is None:
        library_id = str(attrs.pop("library_ids")[0], "utf-8")

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        tissue_positions_file = (
            path / "spatial/tissue_positions.csv"
            if (path / "spatial/tissue_positions.csv").exists()
            else path / "spatial/tissue_positions.parquet" if (path / "spatial/tissue_positions.parquet").exists()
            else path / "spatial/tissue_positions_list.csv"
        )
        files = dict(
            tissue_positions_file=tissue_positions_file,
            scalefactors_json_file=path / "spatial/scalefactors_json.json",
            hires_image=spaceranger_image_path / "tissue_hires_image.png",
            lowres_image=spaceranger_image_path / "tissue_lowres_image.png",
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    logg.warning(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]["images"] = dict()
        for res in ["hires", "lowres"]:
            try:
                adata.uns["spatial"][library_id]["images"][res] = imread(
                    str(files[f"{res}_image"])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
            files["scalefactors_json_file"].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }

        # read coordinates
        if files["tissue_positions_file"].name.endswith(".csv"):
            positions = pd.read_csv(
                files["tissue_positions_file"],
                header=0 if tissue_positions_file.name == "tissue_positions.csv" else None,
                index_col=0,
            )
        elif files["tissue_positions_file"].name.endswith(".parquet"):
            positions = pd.read_parquet(files["tissue_positions_file"])
            #need to set the barcode to be the index
            positions.set_index("barcode", inplace=True)
        positions.columns = [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm["spatial"] = adata.obs[
            ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        ].to_numpy()
        adata.obs.drop(
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata


def destripe_counts(adata, counts_key="n_counts", adjusted_counts_key="n_counts_adjusted"):
    '''
    Scale each row (bin) of ``adata.X`` to have ``adjusted_counts_key``
    rather than ``counts_key`` total counts.

    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key``
        and ``adjusted_counts_key`` in ``.obs``.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column storing the desired destriped counts per bin.
    '''
    # scanpy's utility function to make sure the anndata is not a view
    # if it is a view then weird stuff happens when you try to write to its .X
    sc._utils.view_to_actual(adata)
    # adjust the count matrix to have n_counts_adjusted sum per bin (row)
    # premultiplying by a diagonal matrix multiplies each row by a value: https://solitaryroad.com/c108.html
    bin_scaling = scipy.sparse.diags(adata.obs[adjusted_counts_key] / adata.obs[counts_key])
    adata.X = bin_scaling.dot(adata.X)


def destripe(adata, quantile=0.99, counts_key="n_counts", factor_key="destripe_factor",
             adjusted_counts_key="n_counts_adjusted", adjust_counts=True):
    '''
    Correct the raw counts of the input object for known variable width of
    VisiumHD 2um bins. Scales the total UMIs per bin on a per-row and
    per-column basis, dividing by the specified ``quantile``. The resulting
    value is stored in ``.obs[factor_key]``, and is multiplied by the
    corresponding total UMI ``quantile`` to get ``.obs[adjusted_counts_key]``.

    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key`` in
        ``.obs``.
    quantile : ``float``, optional (default: 0.99)
        Which row/column quantile to use for the computation.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    factor_key : ``str``, optional (default: ``"destripe_factor"``)
        Name of ``.obs`` column to hold computed factor prior to reversing to
        count space.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column for storing the destriped counts per bin.
    adjust_counts : ``bool``, optional (default: ``True``)
        Whether to use the computed adjusted count total to adjust the counts in
        ``adata.X``.
    '''
    # apply destriping via sequential quantile scaling
    # get specified quantile per row
    quant = adata.obs.groupby("array_row")[counts_key].quantile(quantile)
    # divide each row by its quantile (order of obs[counts_key] and obs[array_row] match)
    adata.obs[factor_key] = adata.obs[counts_key] / adata.obs["array_row"].map(quant)
    # repeat on columns
    quant = adata.obs.groupby("array_col")[factor_key].quantile(quantile)
    adata.obs[factor_key] /= adata.obs["array_col"].map(quant)
    # propose adjusted counts as the global quantile multipled by the destripe factor
    adata.obs[adjusted_counts_key] = adata.obs[factor_key] * np.quantile(adata.obs[counts_key], quantile)
    # correct the count space unless told not to
    if adjust_counts:
        destripe_counts(adata, counts_key=counts_key, adjusted_counts_key=adjusted_counts_key)


def expand_labels(adata, labels_key="labels", expanded_labels_key="labels_expanded", algorithm="max_bin_distance",
                  max_bin_distance=2, volume_ratio=4, k=4, subset_pca=True):
    '''
    Expand StarDist segmentation results to bins a maximum distance away in
    the array coordinates. In the event of multiple equidistant bins with
    different labels, ties are broken by choosing the closest bin in a PCA
    representation of gene expression. The resulting labels will be integers,
    with 0 being unassigned to an object.

    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw or destriped counts.
    labels_key : ``str``, optional (default: ``"labels"``)
        ``.obs`` key holding the labels to be expanded. Integers, with 0 being
        unassigned to an object.
    expanded_labels_key : ``str``, optional (default: ``"labels_expanded"``)
        ``.obs`` key to store the expanded labels under.
    algorithm : ``str``, optional (default: ``"max_bin_distance"``)
        Toggle between ``max_bin_distance`` or ``volume_ratio`` based label
        expansion.
    max_bin_distance : ``int`` or ``None``, optional (default: 2)
        Maximum number of bins to expand the nuclear labels by.
    volume_ratio : ``float``, optional (default: 4)
        A per-label expansion distance will be proposed as
        ``ceil((volume_ratio**(1/3)-1) * sqrt(n_bins/pi))``, where
        ``n_bins`` is the number of bins for the corresponding pre-expansion
        label. Default based on cell line
        `data <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8893647/>`_
    k : ``int``, optional (default: 4)
        Number of assigned spatial coordinate bins to find as potential nearest
        neighbours for each unassigned bin.
    subset_pca : ``bool``, optional (default: ``True``)
        If ``True``, will obtain the PCA representation of just the bins
        involved in the tie breaks rather than the full bin space. Results in
        a slightly different embedding at a lower resource footprint.
    '''
    # this is where the labels will go
    adata.obs[expanded_labels_key] = adata.obs[labels_key].values.copy()
    # get out our array grid, and preexisting labels
    coords = adata.obs[["array_row", "array_col"]].values
    labels = adata.obs[labels_key].values
    # we'll be splitting the space in two - the bins with labels, and those without
    object_mask = (labels != 0)
    # get their indices in cell space
    full_reference_inds = np.arange(adata.shape[0])[object_mask]
    full_query_inds = np.arange(adata.shape[0])[~object_mask]
    # for each unassigned bin, we'll find its k nearest neighbours in the assigned space
    # build a reference using the assigned bins' coordinates
    ckd = scipy.spatial.cKDTree(coords[object_mask, :])
    # query it using the unassigned bins' coordinates
    dists, hits = ckd.query(x=coords[~object_mask, :], k=k, workers=-1)
    # convert the identified indices back to the full cell space
    hits = full_reference_inds[hits]
    # get the label calls for each of the hits
    calls = labels[hits]
    # get the area (bin count) of each object
    label_values, label_counts = np.unique(labels, return_counts=True)
    # this is how the algorithm was toggled early on
    # switched to an argument to avoid potential future spaghetti
    if max_bin_distance is None:
        raise ValueError("Use ``algorithm`` to toggle between algorithms")
    if algorithm == "volume_ratio":
        # compute the object's sphere's radius as sqrt(nbin/pi)
        # scale to radius of cell by multiplying by volume_ratio^(1/3)
        # and subtract away the original radius to account for presence of nucleus
        # do a ceiling to compensate for possible reduction of area in slice
        label_distances = np.ceil((volume_ratio ** (1 / 3) - 1) * np.sqrt(label_counts / np.pi))
        # get an array where you can index on object and get the distance
        # needs +1 as the max value of label_values is actually present in the data
        label_distance_array = np.zeros((np.max(label_values) + 1,))
        label_distance_array[label_values] = label_distances
    elif algorithm == "max_bin_distance":
        # just use the provided value
        label_distance_array = np.ones((np.max(label_values) + 1,)) * max_bin_distance
    else:
        raise ValueError("``algorithm`` must be ``'max_bin_distance'`` or ``'volume_ratio'``")
    # construct a matching dimensionality array of max distance allowed per call
    max_call_distance = label_distance_array[calls]
    # mask bins too far away from call with arbitrary high value
    dist_mask = 1000
    dists[dists > max_call_distance] = dist_mask
    # evaluate the minima in each row. start by getting said minima
    min_per_bin = np.min(dists, axis=1)[:, None]
    # now get positions in each row that have the minimum (and aren't the mask)
    is_hit = (dists == min_per_bin) & (min_per_bin < dist_mask)
    # case one - we have a solitary hit of the minimum
    clear_mask = (np.sum(is_hit, axis=1) == 1)
    # get out the indices of the bins
    clear_query_inds = full_query_inds[clear_mask]
    # np.argmin(axis=1) finds the column of the minimum per row
    # subsequently retrieve the matching hit from calls
    clear_query_labels = calls[clear_mask, np.argmin(dists[clear_mask, :], axis=1)]
    # insert calls into object
    adata.obs.loc[adata.obs_names[clear_query_inds], expanded_labels_key] = clear_query_labels
    # case two - 2+ assigned bins are equidistant
    ambiguous_mask = (np.sum(is_hit, axis=1) > 1)
    if np.sum(ambiguous_mask) > 0:
        # get their indices in the original cell space
        ambiguous_query_inds = full_query_inds[ambiguous_mask]
        if subset_pca:
            # in preparation of PCA, get a master list of all the bins to PCA
            # we've got two sets - the query bins, and their k hits
            # the hits needs to be .flatten()ed after masking to become 1d again
            # np.unique sorts in an ascending fashion, which is convenient
            smol = np.unique(np.concatenate([hits[ambiguous_mask, :].flatten(), ambiguous_query_inds]))
            # prepare a PCA as a representation of the GEX space for solving ties
            # can just run straight on an array to get a PCA matrix back. convenient!
            # keep the object's X raw for subsequent cell creation
            pca_smol = sc.pp.pca(np.log1p(adata.X[smol, :]))
            # mock up a "full-scale" PCA matrix to not have to worry about different indices
            pca = np.zeros((adata.shape[0], pca_smol.shape[1]))
            pca[smol, :] = pca_smol
        else:
            # just run a full space PCA
            pca = sc.pp.pca(np.log1p(adata.X))
        # compute the distances between the expression profiles of the undecided bin and the neighbours
        # np.linalg.norm is the fastest way to get euclidean, subtract two point sets beforehand
        # pca[hits[ambiguous_mask, :]] is bins by k by num_pcs
        # pca[ambiguous_query_inds, :] is bins by num_pcs
        # add the [:, None, :] and it's bins by 1 by num_pcs, and subtracts as you'd hope
        eucl_input = pca[hits[ambiguous_mask, :]] - pca[ambiguous_query_inds, :][:, None, :]
        # can just do this along axis=2 and get all the distances at once
        eucl_dists = np.linalg.norm(eucl_input, axis=2)
        # mask ineligible bins with arbitrary high value
        eucl_mask = 1000
        eucl_dists[~is_hit[ambiguous_mask, :]] = eucl_mask
        # define calls based on euclidean minimum
        # same argmin/mask logic as with clear before
        ambiguous_query_labels = calls[ambiguous_mask, np.argmin(eucl_dists, axis=1)]
        # insert calls into object
        adata.obs.loc[adata.obs_names[ambiguous_query_inds], expanded_labels_key] = ambiguous_query_labels


def salvage_secondary_labels(adata, primary_label="labels_he_expanded", secondary_label="labels_gex",
                             labels_key="labels_joint"):
    '''
    Create a joint ``labels_key`` that takes the ``primary_label`` and fills in
    unassigned bins based on calls from ``secondary_label``. Only objects that do not
    overlap with any bins called as part of ``primary_label`` are transferred over.

    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Needs ``primary_key`` and ``secodary_key`` in ``.obs``.
    primary_label : ``str``, optional (default: ``"labels_he_expanded"``)
        ``.obs`` key holding the main labels. Integers, with 0 being unassigned to an
        object.
    secondary_label : ``str``, optional (default: ``"labels_gex"``)
        ``.obs`` key holding the labels to be inserted into unassigned bins. Integers,
        with 0 being unassigned to an object.
    labels_key : ``str``, optional (default: ``"labels_joint"``)
        ``.obs`` key to store the combined label information into. Will also add a
        second column with ``"_source"`` appended to differentiate whether the bin was
        tagged from the primary or secondary label.
    '''
    # these are the bins that have the primary label assigned
    primary = adata.obs.loc[adata.obs[primary_label] > 0, :]
    # these are the bins that lack the primary label, but have the secondary label
    secondary = adata.obs.loc[adata.obs[primary_label] == 0, :]
    secondary = secondary.loc[secondary[secondary_label] > 0, :]
    # kick out any secondary labels that appear in primary-labelled bins
    # we are just interested in ones that are unique to bins without primary labelling
    secondary_to_take = np.array(list(set(secondary[secondary_label]).difference(set(primary[secondary_label]))))
    # both of these labels are integers, starting from 1
    # offset the new secondary labels by however much the maximum primary label is
    offset = np.max(adata.obs[primary_label])
    # use the primary labels as a basis
    adata.obs[labels_key] = adata.obs[primary_label].copy()
    # flag any bins that are assigned to our secondary labels of interest
    mask = np.isin(adata.obs[secondary_label], secondary_to_take)
    adata.obs.loc[mask, labels_key] = adata.obs.loc[mask, secondary_label] + offset
    # store information on origin of call
    adata.obs[labels_key + "_source"] = "none"
    adata.obs.loc[adata.obs[primary_label] > 0, labels_key + "_source"] = "primary"
    adata.obs.loc[mask, labels_key + "_source"] = "secondary"
    # stash secondary label offset as that seems potentially useful
    if "bin2cell" not in adata.uns:
        adata.uns["bin2cell"] = {}
    if "secondary_label_offset" not in adata.uns["bin2cell"]:
        adata.uns["bin2cell"]["secondary_label_offset"] = {}
    adata.uns["bin2cell"]["secondary_label_offset"][labels_key] = offset
    # notify of how much was salvaged
    print("Salvaged " + str(len(secondary_to_take)) + " secondary labels")


def bin_to_cell(adata, labels_key="labels_expanded", spatial_keys=["spatial"], diameter_scale_factor=None):
    '''
    Collapse all bins for a given nonzero ``labels_key`` into a single cell.
    Gene expression added up, array coordinates and ``spatial_keys`` averaged out.
    ``"spot_diameter_fullres"`` in the scale factors multiplied by
    ``diameter_scale_factor`` to reflect increased unit size. Returns cell level AnnData,
    including ``.obs["bin_count"]`` reporting how many bins went into creating the cell.

    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw or destriped counts. Needs ``labels_key`` in ``.obs``
        and ``spatial_keys`` in ``.obsm``.
    labels_key : ``str``, optional (default: ``"labels_expanded"``)
        Which ``.obs`` key to use for grouping 2um bins into cells. Integers, with 0 being
        unassigned to an object. If an extra ``"_source"`` column is detected as a result
        of ``b2c.salvage_secondary_labels()`` calling, its info will be propagated per
        label.
    spatial_keys : list of ``str``, optional (default: ``["spatial"]``)
        Which ``.obsm`` keys to average out across all bins falling into a cell to get a
        cell's respective spatial coordinates.
    diameter_scale_factor : ``float`` or ``None``, optional (default: ``None``)
        The object's ``"spot_diameter_fullres"`` will be multiplied by this much to reflect
        the change in unit per observation. If ``None``, will default to the square root of
        the mean of the per-cell bin counts.
    '''
    # a label of 0 means there's nothing there, ditch those bins from this operation
    adata = adata[adata.obs[labels_key] != 0]
    # use the newly inserted labels to make pandas dummies, as sparse because the data is huge
    cell_to_bin = pd.get_dummies(adata.obs[labels_key], sparse=True)
    # take a quick detour to save the cell labels as they appear in the dummies
    # they're likely to be integers, make them strings to avoid complications in the downstream AnnData
    cell_names = [str(i) for i in cell_to_bin.columns]
    # then pull out the actual internal sparse matrix (.sparse) as a scipy COO one, turn to CSR
    # this has bins as rows, transpose so cells are as rows (and CSR becomes CSC for .dot())
    cell_to_bin = cell_to_bin.sparse.to_coo().tocsr().T
    # can now generate the cell expression matrix by adding up the bins (via matrix multiplication)
    # cell-bin * bin-gene = cell-gene
    # (turn it to CSR at the end as somehow it comes out CSC)
    X = cell_to_bin.dot(adata.X).tocsr()
    # create object, stash stuff
    cell_adata = ad.AnnData(X, var=adata.var)
    cell_adata.obs_names = cell_names
    # turn the cell names back to int and stash that as metadata too
    cell_adata.obs['object_id'] = [int(i) for i in cell_names]
    # need to bust out deepcopy here as otherwise altering the spot diameter gets back-propagated
    cell_adata.uns['spatial'] = deepcopy(adata.uns['spatial'])
    # getting the centroids (means of bin coords) involves computing a mean of each cell_to_bin row
    # premultiplying by a diagonal matrix multiplies each row by a value: https://solitaryroad.com/c108.html
    # use that to divide each row by it sum (.sum(axis=1)), then matrix multiply the result by bin coords
    # stash the sum into a separate variable for subsequent object storage
    # cell-cell * cell-bin * bin-coord = cell-coord
    bin_count = np.asarray(cell_to_bin.sum(axis=1)).flatten()
    row_means = scipy.sparse.diags(1 / bin_count)
    cell_adata.obs['bin_count'] = bin_count
    # take the thing out for a spin with array coordinates
    cell_adata.obs["array_row"] = row_means.dot(cell_to_bin).dot(adata.obs["array_row"].values)
    cell_adata.obs["array_col"] = row_means.dot(cell_to_bin).dot(adata.obs["array_col"].values)
    # generate the various spatial coordinate systems
    # just in case a single is passed as a string
    if type(spatial_keys) is not list:
        spatial_keys = [spatial_keys]
    for spatial_key in spatial_keys:
        cell_adata.obsm[spatial_key] = row_means.dot(cell_to_bin).dot(adata.obsm[spatial_key])
    # of note, the default scale factor bin diameter at 2um resolution stops rendering sensibly in plots
    # by default estimate it as the sqrt of the bin count mean
    if diameter_scale_factor is None:
        diameter_scale_factor = np.sqrt(np.mean(bin_count))
    # bump it up to something a bit more sensible
    library = list(adata.uns['spatial'].keys())[0]
    cell_adata.uns['spatial'][library]['scalefactors']['spot_diameter_fullres'] *= diameter_scale_factor
    # if we can find a source column, transfer that
    if labels_key + "_source" in adata.obs.columns:
        # hell of a one liner. the premise is to turn two columns of obs into a translation dictionary
        # so pull them out, keep unique rows, turn everything to string (as labels are strings in cells)
        # then set the index to be the label names, turn the thing to dict
        # pd.DataFrame -> dict makes one entry per column (even if we just have the one column here)
        # so pull out our column's entry and we have what we're after
        mapping = \
        adata.obs[[labels_key, labels_key + "_source"]].drop_duplicates().astype(str).set_index(labels_key).to_dict()[
            labels_key + "_source"]
        # translate the labels from the cell object
        cell_adata.obs[labels_key + "_source"] = [mapping[i] for i in cell_adata.obs_names]
    return cell_adata

print("data_utils.py loaded")