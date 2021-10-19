import numpy as np
import progressbar
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


def when_contained(true_z,
                   simulated_Z_array,
                   pd_vec,
                   sbr,
                   verbose=False):
    """
    Calculate the relative rank of the simulated_Z_array when the observation
    is contained in a union of small balls around the simulated z. Note this
    containment is pointwise relative to the small-ball-radius (sbr)

    Arguments
    ---------
    true_z : numpy array
        array (n,m) of true observation (represented as a surface)
    simulated_Z_array : numpy array
        array (B,n,m), where each row is a simulated observation (represented
        as a surface)
    pd_vec : numpy array
        array (B,) of psuedo-density values for the simulated_Z_array
        simulations (relative to the rows of said array)
    sbr : float
        small-ball-radius
    verbose : boolean
        if progress should be reported verbosely

    Return
    ------
    contained_idx : int
        index of which simulation (added relative to pd_vec ordering) the
        true_z is first contained relative to the union of balls with radius
        sbr. Note if the value np.inf is returned, then the true_z is never
        completely contained but the simulations.
    """

    # logic checks ---------------
    assert true_z.shape == (simulated_Z_array.shape[1],
                            simulated_Z_array.shape[2]), \
        "expected true_z to have same shape of the rows in the "+\
        "simulated_Z_array"

    assert simulated_Z_array.shape[0] == pd_vec.shape[0], \
        "expected simuated_Z_array number of rows to the same as the "+\
        "length of the pd_vec."

    order = pd_vec.argsort()[::-1]

    s_iter = np.arange(pd_vec.shape[0])
    if verbose:
        bar = progressbar.ProgressBar()
        s_iter = bar(s_iter)

    containment_mat = False * np.ones(true_z.shape)
    min_distance = np.inf * np.ones(true_z.shape)

    for s_idx in s_iter:
        ordered_index = order[s_idx]
        pointwise_dist = l2_dist_lots2one_pointwise(true_z,
                                                    simulated_Z_array[
                                                        np.newaxis,
                                                        ordered_index,:])

        min_distance = np.array([min_distance,
                                 pointwise_dist.reshape(min_distance.shape)
                                ]).min(axis=0)


        containment_mat[min_distance <= sbr] = True

        if np.all(containment_mat):
            return(s_idx)

    if not np.all(containment_mat):
        return(np.inf)

