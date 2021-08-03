import numpy as np

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
