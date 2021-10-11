import numpy as np
import pandas as pd
import random
import progressbar
from sklearn.decomposition import PCA
import torch

def simulate(model,
             pca_rad, rad_cols_means,
             pca_size, size_cols_means,
             device,
             size_mtx_sim_NA,
             rad_mtx_sim_NA,
             extra_feats_size_mtx_sim_NA,
             extra_feats_rad_mtx_sim_NA,
             rows_start,
             seed,
             verbose=True):
    """
    Arguments
    ---------
    model : PixelCNNMultiHead model
        pre-trained PixelCNNMultiHead model
    pca_rad : sklearn.decomposition._pca.PCA
        pretrained centered pca compression for radius (3 PC)
    rad_cols_means : numpy array
        centering vector for pca_rad
    pac_size : sklearn.decomposition._pca.PCA
        pretrained centered pca compression for size (2 PC)
    size_cols_means : numpy array
        centering vector for pca_rad
    device : torch.device
        connection to cpu machine
    size_mtx_sim_NA: numpy array
        numpy array (n + m, a1) with size structure for X with n rows and m rows of NAs for the Y
    rad_mtx_sim_NA: numpy array
        numpy array (n + m, a2) with rad structure for X with n rows and m rows of NAs for the Y
    extra_feats_size_mtx_sim_NA : numpy array
        numpy array (n + m, a3) of additional features associated with size (for
        X with n+1 rows and m-1 rows of NAs for the Y)
    extra_feats_rad_mtx_sim_NA : numpy array
        numpy array (n + m, a4) for additional features associated with rad (for
        X with n+1 rows  and m-1 rows of NAs for the Y)
    rows_start : int
        number of rows to simulate beyond (aka size of X, n)
    seed : int
        seed for random.seed and np.random.seed setting
    verbose : boolean
        if progress should be presented with a progress bar

    Returns
    -------
    size_mtx_Y : numpy array
        numpy array (m, a1) of simulated size matrix for
        Y (rows_simulation_beyond number of steps away from end of X's size
        matrix)
    rad_mtx_Y : numpy array
        numpy array (m, a2) of simulated rad matrix for
        Y (rows_simulation_beyond number of steps away from end of X's rad
        matrix)
    """
    # setting seeds (across both known options)...
    random.seed(seed)
    np.random.seed(seed)

    # sizes
    num_full_rows = size_mtx_sim_NA.shape[0]
    num_size_cols = size_mtx_sim_NA.shape[1]
    num_rad_cols = rad_mtx_sim_NA.shape[1]
    num_extra_size_cols = extra_feats_size_mtx_sim_NA.shape[2]
    num_extra_rad_cols = extra_feats_rad_mtx_sim_NA.shape[2]

    # size of Y
    rows_simulate_beyond = num_full_rows - rows_start

    # basic checks
    assert num_full_rows == rad_mtx_sim_NA.shape[0] and \
        num_full_rows == extra_feats_size_mtx_sim_NA.shape[1] and \
        num_full_rows == extra_feats_rad_mtx_sim_NA.shape[1], \
        "expected the same number of rows in all sim_NA structures, " + \
        "(size_mtx_sim_NA.shape[0], rad_mtx_sim_NA.shape[0], "+ \
        "extra_feats_size_mtx_sim_NA.shape[1], extra_feats_rad_mtx_sim_NA.shape[1])"

    # creating larger desired structure to pass into model
    # we are putting in nans to rows not yet predicted...


    # actual prediction process

    if verbose:
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(),
                                           progressbar.Percentage(),
                                           ' ',
                                           progressbar.ETA()])

    with torch.no_grad():
        # double for loop through all pixels starting from simulation row
        row_iter = np.arange(rows_start, num_full_rows)
        if verbose:
            row_iter = bar(row_iter)
        for row in row_iter:
            for col in range(num_rad_cols):
                # interested in predicting pixel (row, col)
                # BPL: the below prediction will update as we update the
                # *_mtx extra_feats_*_mtx for each inner loop step
                out_size, out_rad = model.forward(
                    torch.Tensor([[size_mtx_sim_NA]]).to(device),  # would it be cheaper to only provide up to necessary rows?
                    torch.Tensor([[rad_mtx_sim_NA]]).to(device),
                    torch.Tensor([extra_feats_size_mtx_sim_NA]).to(device),
                    torch.Tensor([extra_feats_rad_mtx_sim_NA]).to(device),
                )
                # BPL: since we're looking over both rad and size columns we
                # need to only update size when not too extreme...

                # compute mean/stdev of both orbs for pixel (row, col)
                if col < num_size_cols:
                    mean_size = out_size[0, 0, row, col] / 1e4
                    stdev_size = np.exp(torch.Tensor.cpu(out_size[0, 1, row, col]) / 1e5) * 0.1

                mean_rad = out_rad[0, 0, row, col] / 1e4
                stdev_rad = np.exp(torch.Tensor.cpu(out_rad[0, 1, row, col]) / 1e5) * 10

                # sample pixel value from each orb
                # set (row, col) pixel value in orb to this sampled one
                if col < num_size_cols:
                    size_val = np.random.normal(torch.Tensor.cpu(mean_size),
                                                torch.Tensor.cpu(stdev_size))
                    size_mtx_sim_NA[row, col] = size_val

                rad_val = np.random.normal(torch.Tensor.cpu(mean_rad),
                                           torch.Tensor.cpu(stdev_rad))
                rad_mtx_sim_NA[row, col] = rad_val  # BPL: updating matrix

            # take PCA after row is fully simulated and set feature values
            if row + 1 < num_full_rows: # BPL: this is out of first loop
                rad_pcas = pca_rad.transform((rad_mtx_sim_NA[row, :] - \
                                                np.array(rad_cols_means)).reshape(1, -1))[0]
                rad_pca1_val, rad_pca2_val = rad_pcas[0], rad_pcas[1]
                # set PCA coeffs for next row
                extra_feats_size_mtx_sim_NA[2, row + 1, :] = rad_pca1_val
                extra_feats_size_mtx_sim_NA[3, row + 1, :] = rad_pca2_val

                size_pcas = pca_size.transform((size_mtx_sim_NA[row, :] - \
                                                    np.array(size_cols_means)).reshape(1, -1))[0]
                size_pca1_val, size_pca2_val, size_pca3_val = size_pcas[0], size_pcas[1], size_pcas[2]
                extra_feats_rad_mtx_sim_NA[2, row + 1, :] = size_pca1_val
                extra_feats_rad_mtx_sim_NA[3, row + 1, :] = size_pca2_val
                extra_feats_rad_mtx_sim_NA[4, row + 1, :] = size_pca3_val

        size_mtx_Y = size_mtx_sim_NA[rows_start:,]
        rad_mtx_Y = rad_mtx_sim_NA[rows_start:,]

        return size_mtx_Y, rad_mtx_Y
