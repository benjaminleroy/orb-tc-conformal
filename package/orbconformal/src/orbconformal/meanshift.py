import numpy as np
from .distances import l2_dist_lots2one, l2_dist, l2_dist_matrix
import progressbar
import joblib
import networkx as nx
import pandas as pd

def meanshift_multidim_funct_single(X_array,
                                    current_obs,
                                    sigma, eps, maxT):
    """
    using guassian kernel meanshift algorithm to shift a single observation
    to their mode along the kernel pseudo-density.

    Parameters
    ----------
    X_array: numpy.ndarray
        numpy array (n, p, q), a set of multivariate functions, rows
        correspond to new observations
    current_obs: numpy.ndarray
        numpy array (p, q), single multivariate function to move up to
        a mode
    sigma: float
        scale for guassian kernel pseudo-density
    eps: float
        positive error bound to declare progression towards the mode complete
    maxT: int
        maximum number of steps to take if eps doesn't stop the
        progression towards the mode.

    Returns
    -------
    t: int
        number of interations actually taken
    current_obs_inner: numpy.ndarray
        final step taken by the current_obs toward the mode
    """


    current_obs_inner = current_obs.copy()

    if maxT == 0: # so that we don't think we're at least doing 1 step
        return -1, current_obs_inner

    for t in np.arange(maxT):
        d_inner = l2_dist_lots2one(current_obs_inner, X_array)

        kernel_weights_inner = np.exp(-1*d_inner**2/sigma)
        kernel_weights_inner_standardized = kernel_weights_inner/\
            kernel_weights_inner.sum()

        # next step towards mode:
        multi_mat = np.concatenate([m*np.eye(X_array.shape[1]).reshape(
                                                                1,
                                                                X_array.shape[1],
                                                                X_array.shape[1])
                                    for m in kernel_weights_inner_standardized])

        temp_inner = np.matmul(multi_mat, X_array).sum(axis = 0)
        diff_inner = l2_dist(temp_inner, current_obs_inner)
        current_obs_inner = temp_inner

        if diff_inner < eps:
            break

    return t, current_obs_inner


def meanshift_multidim_funct(X_array, G_array = None,
                             sigma = 1, eps = 1e-05, maxT = 40,
                             parallel = 1, verbose = True):
    """
    using guassian kernel meanshift algorithm to shift a observations
    to their mode along the kernel pseudo-density.

    Parameters
    ----------
    X_array: numpy.ndarray
        numpy array (n, p, q), a set of multivariate functions, rows
        correspond to new observations
    G_array: numpy.ndarray
        numpy array (m, p, q), with a set of multivariate function to move
        up to a mode (default is None, which makes it the same as X_array)
    sigma: float
        scale for guassian kernel pseudo-density
    eps: float
        positive error bound to declare progression towards the mode complete
    maxT: int or numpy.ndarray
        int or vector of integers (m, ), the maximum number of steps to take
        if eps doesn't stop the progression towards the mode. If this is a
        vector of integers, then this is relative to each row.
    parallel: int
        the number of cores to use to parallelize the process (across
        rows of G_array). If parallel = 1, then no paralleization is used. Note
        if this number is greater than the number of cores available it will
        revert to the maximum number of cores.
    verbose: boolean
        if true then progress is reported with progressbar / output

    Returns
    -------
    t: numpy.ndarray
        numpy vector (m, ) of number of interations actually taken for each
        multivariate function in G_array
    current_obs_inner: numpy.ndarray
        numpy array (m, p, q) final step taken by the each of the individual
        multivariate fucntions in G_array toward their mode
    """
    if G_array is None:
        G_array = X_array.copy()

    if type(maxT) is int:
        maxT = maxT*np.ones(G_array.shape[0], dtype = int)

    t_vec = np.zeros(G_array.shape[0])
    if parallel == 1:
        G_shift_out = G_array.copy()
        if verbose:
            bar = progressbar.ProgressBar()
            r_iter = bar(np.arange(G_array.shape[0]))
        else:
            r_iter = np.arange(G_array.shape[0])

        for r_idx in r_iter:
            t, step_inner = meanshift_multidim_funct_single(X_array,
                                                           G_array[r_idx],
                                                           sigma = sigma,
                                                           eps = eps,
                                                           maxT = maxT[r_idx])
            G_shift_out[r_idx] = step_inner
            t_vec[r_idx] = t

    else:
        n_cores = min(joblib.cpu_count(), parallel)
        if verbose:
            parallel_verbose = 10
        else:
            parallel_verbose = 0


        inner_func = lambda r_idx: meanshift_multidim_funct_single(X_array,
                                                                   G_array[r_idx],
                                                                   sigma = sigma,
                                                                   eps = eps,
                                                                   maxT = maxT[r_idx])

        out = joblib.Parallel(n_jobs = n_cores, verbose= parallel_verbose)(
                    joblib.delayed(inner_func)(r_idx)
                    for r_idx in np.arange(G_array.shape[0]))

        # process out information
        G_shift_out = G_array.copy()
        for r_idx in np.arange(len(out)):
            t, value = out[r_idx]
            G_shift_out[r_idx] = value.copy()
            t_vec[r_idx] = t

    return t_vec, G_shift_out


def mode_clustering(X_array, sigma, eps=1e-07, maxT = 50, diff_eps = 1e-07,
                    parallel = 1, verbose = True):
    """
    Find mode clusters of multivariate functional objects

    Parameters
    ----------
    X_array: numpy.ndarray
        numpy array (r, n, p). Each row has a single representation of a
        multivariate function in it.
    sigma: float
        scale value for the distances between observations
    eps: float
        if difference between steps is less than this - treat as
        if the point has converged
    maxT: int
        max number of iterations of the algorithm
    diff_eps: float
        if the final step of each of the points is within
        this distance from each-other they will be grouped together.
    parallel: boolean
        if we should parallelize the meanshift algorithm part
    verbose:  boolean
        if we should show progress

    Returns
    -------
    t : int
        integer of the number of groups there are
    df : pandas.core.frame.DataFrame
        DataFrame with indices of X_array's rows (in column "index") and a
        grouping index for each observation (in column "group")
    """

    # meanshift
    out_t, out_G_array = meanshift_multidim_funct(X_array = X_array,
                                                 sigma = sigma,
                                                 eps = eps,
                                                 maxT = maxT,
                                                 parallel = parallel,
                                                 verbose = verbose)

    # distance between converged-to functions
    dist_mat = l2_dist_matrix(out_G_array)

    # graph based analysis
    binary_mat = 1*(dist_mat < diff_eps) # thresholding graph connections
    G = nx.from_numpy_array(binary_mat)

    group_info = list(nx.connected_components(G))
    n_groups = len(group_info)

    # grouping DF creation
    grouping = []
    indices = []
    for g_idx, g_set in enumerate(group_info):
        grouping = grouping + [g_idx]*len(g_set)
        indices = indices + list(g_set)

    pd_out = pd.DataFrame(data = {"index": indices,
                                  "group": grouping},
                         columns = ["index", "group"])

    return n_groups, pd_out


def mode_clustering_check(X_array, sigma, eps_vec=np.array([1e-07]),
                          maxT = 50, diff_eps_vec = np.array([1e-07]),
                          parallel = 1, verbose = True):
    """
    Inner tuning across different values of eps and diff_eps to identify the
        number of mode clusters for multivariate functional objects

    Parameters
    ----------
    X_array: numpy.ndarray
        numpy array (r, n, p). Each row has a single representation of a
        multivariate function in it.
    sigma: float
         scale value for the distances between observations
    eps_vec: numpy.ndarray
        numpy array (e, ) The range of eps values to look across. Where a
        single eps value determines when to stop stepping up the psuedo-density
        if the step before is less than eps away from the current step
        (suggests convergence)
    maxT: int
        max number of iterations of the meanshift algorithm
    diff_eps_vec: numpy.ndarray
        numpy array (d, ) The range of diff_eps values to look across.
        Where a single diff_eps value determines if the distance between a pair
        of converged points is small enough to suggest they should be in the
        same group
    parallel: boolean
        if we should parallelize the meanshift algorithm part
    verbose:  boolean
        if we should show progress

    Returns
    -------
    numpy.ndarray
        a matrix (e, d) of integers of the number of mode clusters observed with
        paramters eps_vec[e] and diff_eps_vec[d]
    """

    info_mat = -1 * np.ones((eps_vec.shape[0], diff_eps_vec.shape[0]))


    eps_vec.sort()
    diff_eps_vec.sort()

    eps_list = list(eps_vec.copy())
    diff_eps_list = list(diff_eps_vec.copy()[::-1])

    out_G_array = X_array.copy()
    maxT_vec = np.array([maxT]*X_array.shape[0])

    eps_index = 0
    while np.min(maxT_vec) > 0 and len(eps_list) > 0:

        # EPS part
        eps = eps_list.pop() # will grab largest value (which is at the end...)
        #
        out_t, out_G_array = meanshift_multidim_funct(X_array = X_array,
                                                      G_array = out_G_array,
                                                     sigma = sigma,
                                                     eps = eps,
                                                     maxT = maxT_vec,
                                                     parallel = parallel,
                                                     verbose = verbose)


        maxT_vec = maxT_vec - (out_t + 1) # since out_t is 0 indexed (but maxT_vec isn't)

        # DIFF_EPS part
        for diff_eps_index, diff_eps in enumerate(diff_eps_list):

            dist_mat = l2_dist_matrix(out_G_array)

            # graph based analysis
            binary_mat = 1*(dist_mat < diff_eps) # thresholding graph connections
            G = nx.from_numpy_array(binary_mat)

            group_info = list(nx.connected_components(G))
            info_mat[eps_index, diff_eps_index] = len(group_info)

        eps_index += 1

    return info_mat
