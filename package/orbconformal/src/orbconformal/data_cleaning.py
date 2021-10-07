import numpy as np
import pandas as pd

def linear_interp(mat, na_vec):
    """
    Linear interpolate a matrix's row values.

    Parameters
    ----------
    mat : numpy.ndarray
        numpy array (n, p) numeric matrix with rows that need to be
        interpolated
    na_vec : numpy.ndarray
        boolean numpy vector (n, ) if row should be interpolated (True)

    Returns
    -------
    numpy.ndarray
        updated matrix, with interpolated rows

    Notes
    -----
        If the beginning or end of the matrix needs to be interpolated this
        should be done outside this function.
    """
    assert mat.shape[0] == na_vec.shape[0], \
        "expect mat rows and na_vec to have same number of obs"

    # track beginning and end of missing data segments
    out_mat = mat.copy()
    starts = []
    ends = []
    tracking_sequence = False
    for right in range(na_vec.shape[0]):
        current_nas = na_vec[right]
        if tracking_sequence:
            if not current_nas:
                ends += [right-1]
        else:
            if current_nas:
                starts += [right]

        tracking_sequence = current_nas

    # calculate length of these segments
    ends_np = np.array(ends)
    starts_np = np.array(starts)
    len_segments = ends_np-starts_np + 1

    for segment_idx in np.arange(starts_np.shape[0]):

        before = mat[starts_np[segment_idx]-1,:]
        after = mat[ends_np[segment_idx]+1,:]
        current_length = len_segments[segment_idx]


        for update_idx, update_row_idx in \
            enumerate(np.arange(starts_np[segment_idx],
                                ends_np[segment_idx]+1)):
            inner_frac = (update_idx + 1)/(current_length + 1)
            out_mat[update_row_idx,:] = inner_frac*after + (1-inner_frac)*before

    return out_mat

def linear_interp_df(df, cols, row_update="any"):
    """
    wrapper for linear_interp from data frames

    Arguments
    ---------
    df : pd.DataFrame
        data frame of tropical cyclone information
    cols : list
        list of column names of df that will use interpolated if need be
    row_update : string
        currently must be "any", but captures the idea that one could update

    Returns
    -------
    updated df based on linear interp
    """
    df_inner = df.copy()
    mat = np.array(df_inner.loc[:, cols])

    if row_update == "any":
        na_info = np.isnan(mat).mean(axis = 1)

    mat_update = linear_interp(mat, na_info)

    df_inner.loc[:,cols] = mat_update

    return df_inner


def remove_init_and_final_missing_rows_for_interp(single_df, size_cols,
                                                  rad_cols,
                                                  cutoff=1):
    """
    Removes final rows of data frame if columns of either size_cols or rad_cols
    are all NANs

    Arguments
    ---------
    single_df: pd.DataFrame
        data frame (assumed for a single TC) with columns size_cols and rad_cols
        to be examined
    size_cols: list
        list of columns for the size function
    rad_cols: list
        list of columns for the rad function
    cutoff: float
        float between 0 and 1 (inclusive) where we remove

    Returns:
    --------
    updated single_df data frame with final rows removed if they are all NANs

    Details:
    --------
    It may make sense to also remove to rows until no NAN is observed...
    """
    # size ---------

    single_df_idx = single_df.copy().reset_index(drop = True)

    single_mat = np.array(single_df_idx.loc[:,size_cols])

    na_info = np.isnan(single_mat).mean(axis = 1)

    if np.all(na_info >= cutoff):
        return pd.DataFrame(columns = single_df_idx.columns)

    if (na_info >= cutoff)[-1] or (na_info >= cutoff)[0]:
        if (na_info >= 1)[-1]: #bottom
            empty_final_rows = np.sum(np.cumsum((na_info >= cutoff)[::-1]) ==                                   np.arange(1,na_info.shape[0]+1, dtype = int))
            single_df_updated = single_df_idx.drop(
                                    list(np.arange(na_info.shape[0]-empty_final_rows,
                                                   na_info.shape[0])))
        else:
            single_df_updated = single_df_idx

        if (na_info >= cutoff)[0]: #top
            empty_init_rows = np.sum(np.cumsum((na_info >= cutoff)) ==                                   np.arange(1,na_info.shape[0]+1, dtype = int))
            single_df_updated = single_df_updated.drop(
                                    list(np.arange(empty_init_rows))) #0:(empty_init_rows-1)

    else:
        single_df_updated = single_df_idx

    single_df_updated = single_df_updated.reset_index(drop = True) # otherwise we'll drop the wrong rows
    # rad -----------
    single_mat2 = np.array(single_df_updated.loc[:,rad_cols])

    na_info2 = np.isnan(single_mat2).mean(axis = 1)

    if np.all(na_info2 >= cutoff):
        return pd.DataFrame(columns = single_df_idx.columns)


    if (na_info2 >= cutoff)[-1] or (na_info2 >= cutoff)[0]:
        if (na_info2 >= cutoff)[-1]: #bottom
            empty_final_rows2 = np.sum(np.cumsum((na_info2 >= cutoff)[::-1]) ==                                   np.arange(1,na_info2.shape[0]+1, dtype = int))

            single_df_updated2 = single_df_updated.drop(
                                    list(np.arange(na_info2.shape[0]-empty_final_rows2, \
                                                   na_info2.shape[0])))

        else:
            single_df_updated2 = single_df_updated

        if (na_info2 >= cutoff)[0]: #top
            empty_init_rows2 = np.sum(np.cumsum((na_info2 >= cutoff)) ==                                   np.arange(1,na_info2.shape[0]+1, dtype = int))

            single_df_updated2 = single_df_updated2.drop(
                                    list(np.arange(empty_init_rows2)))#0:(empty_init_rows2-1)
    else:
        single_df_updated2 = single_df_updated

    return single_df_updated2


