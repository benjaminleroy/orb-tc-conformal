import numpy as np

def l2_dist(mat1, mat2):
    """
    calculate l2 distance between two matrices

    Parameters
    ----------
    mat1: numpy.ndarray
        numpy array (n, p)
    mat2: numpy.ndarray
        numpy array (n, p)

    Returns
    -------
    float
        l2 distance between matrices
    """
    assert len(mat1.shape) == 2, \
        "mat1/mat2 should be 2d matrices"
    assert mat1.shape == mat2.shape, \
        "matrix shapes must be the same"
    return np.sqrt(np.sum((mat1-mat2)**2))

def l2_dist_lots2one(mat1, data_array):
    """
    calculate l2 distance between a matrix and an array of matrices

    Parameters
    ----------
    mat1: numpy.ndarray
        numpy array (n, p)
    data_array: numpy.ndarray
        numpy array (r,n,p) treat each row (relative to first index)
        as a new matrix

    Returns
    --------
    numpy.ndarray
        numpy vector (r,) of l2 distances between mat1 and data_array[r]


    """
    assert len(mat1.shape) == 2 and len(data_array.shape) == 3, \
        "mat1 should be 2d and data_array should be 3d"
    assert mat1.shape == data_array.shape[1:], \
        "matrix and data_array columns must be the same shape"

    mat1_array = np.tile(mat1.reshape(1,mat1.shape[0], mat1.shape[1]),
                         (data_array.shape[0],1,1))

    return np.sqrt(np.sum((mat1_array-data_array)**2, axis = (1,2)))

def l2_dist_matrix(X_array, Y_array = None):
    """
    calculates a distance matrix between two areas of multivariate functions

    Parameters
    ----------
    X_array: numpy.ndarray
        numpy array (r, n, p), array of multivariate functions (each row
        is one funciton)
    Y_array: numpy.ndarray
        numpy array (t, n, p), array of multivariate functions (each row
        is one funciton). Default is None (if so, then X_array is used)

    Returns
    -------
    numpy.ndarray
        numpy array (r, t) of distances between X_array[r] and Y_array[t]
    """
    # checks / definitions

    if Y_array is None:
        Y_array = X_array.copy()

    assert X_array.shape[1:] == Y_array.shape[1:], \
        "X_array and Y_array's matrix structure should be the same dimensions"

    n = X_array.shape[0]
    m = Y_array.shape[0]
    dmat = np.zeros((n, m))

    for r_idx in np.arange(n):
        dmat[r_idx,:] = l2_dist_lots2one(X_array[r_idx], Y_array)

    return dmat
