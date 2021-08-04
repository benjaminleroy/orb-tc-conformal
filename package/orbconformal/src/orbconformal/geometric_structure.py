import numpy as np
from .distances import l2_dist_lots2one_pointwise


def calc_small_ball_rad_multidim_func(X_array,
                                      pd_vec):
    """
    calculate the sequence of small-ball-radius coverage values for
    multidimensional functions

    Parameters
    ----------
    X_array : numpy.ndarray
        numpy array (r, n, p), each row is a representation of a
        multidimensional function.
    pd_vec : numpy.ndarray
        numpy array (r,) of psuedo-density values (associated with the
        rows of X_array)

    Returns
    -------
    rad_vec : numpy.ndarray
        numpy array (r,) of small-ball-radius values as we increasely
        add the rth highest psuedo-density to the grouping
    """

    n = pd_vec.shape[0]

    assert X_array.shape[0] == n, \
        "expect X_array rows and pd_vec length to be the same size"

    pd_order = np.argsort(pd_vec)[::-1]

    X_array2 = X_array[pd_order]


    rad_vec = -1 * np.ones(n)
    rad_vec[0] = 0

    min_dist_mat = np.inf * np.ones(X_array.shape)

    for r_idx in np.arange(1, n):

        pointwise_dist = l2_dist_lots2one_pointwise(X_array2[r_idx],
                                                    X_array2[:r_idx])
        if r_idx == 1:
            min_dist_mat[0] = pointwise_dist.reshape((pointwise_dist.shape[1],
                                                      pointwise_dist.shape[2]))
            min_dist_mat[1] = pointwise_dist.reshape((pointwise_dist.shape[1],
                                                      pointwise_dist.shape[2]))
        else:
            min_dist_mat[r_idx] = pointwise_dist.min(axis = 0)
            inner_min_logic = pointwise_dist < min_dist_mat[0:r_idx]
            new_min = min_dist_mat[:r_idx] * np.logical_not(inner_min_logic) + \
                inner_min_logic * pointwise_dist
            min_dist_mat[:r_idx] = new_min

        rad_vec[r_idx] = np.max(min_dist_mat[:(r_idx+1)])

    return rad_vec
