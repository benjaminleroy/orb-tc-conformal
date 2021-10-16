import numpy as np
from .distances import l2_dist_matrix
from .utils import check_character_percent

def psuedo_density_multidim_func(X_array, Y_array = None,
                  sigma_string = "45%"):
    """
    calculate guassian kernel pseudo-density values for multidimensional
    functions

    Parameters
    ----------
    X_array : numpy.ndarray
        numpy array (r, n, p), each row is a representation of a
        multidimensional function. These will be used to define the
        kernel pseudo-density
    Y_array : numpy.ndarray
        numpy array (s, n, p),  each row is a representation of a
        multidimensional function. The returned psuedo-density values will
        be for these functions. If Y_array is None, then will use X_array as
        Y_array.
    sigma_string : str
        string of the precentage for the sigma value

    Returns
    -------
    pd_vec : numpy.ndarray
        numpy vector (s, ) of psuedo-density values of Y_array
    """

    sigma_proportion = check_character_percent(sigma_string,
                                               name = "sigma_string")

    # calculating true sigma value:
    X_dmat = l2_dist_matrix(X_array)
    sigma = np.quantile(X_dmat.ravel(), sigma_proportion)


    if Y_array is None:
        Y_dmat = X_dmat.copy()
    else:
        Y_dmat = l2_dist_matrix(X_array = X_array,
                                Y_array = Y_array)

    Y_kmat = np.exp(-Y_dmat**2/sigma**2)
    pd_vec = Y_kmat.mean(axis = 0)
    # ^given oc.l2_dist_matrix output means Y_values will be associated with columns

    return pd_vec
