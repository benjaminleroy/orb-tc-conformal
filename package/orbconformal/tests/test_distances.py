import numpy as np
import orbconformal as oc

def test_l2_dist():
    """
    test l2_dist function (basic)
    """
    # same matrix
    mat1 = np.random.normal(size = 20).reshape((4,5))
    assert oc.l2_dist(mat1, mat1) == 0, \
        "distance between self should be 0"

    # random
    mat2 = np.random.normal(size = 20).reshape((4,5))
    assert oc.l2_dist(mat1, mat2) >= 0, \
        "distance between two random matrices should be greater than"+\
        "or equal to 0"

    # static
    mat1 = np.arange(20).reshape((4,5))
    mat2 = mat1.copy()
    mat2[0,0] = 20
    assert oc.l2_dist(mat1, mat2) == 20, \
        "static distance test didn't correctly work"


def test_l2_dist_lots2one():
    """
    test l2_dist_lots2one function
    """

    data_array = np.random.normal(size = 5*20).reshape(5,4,5)
    mat1 = data_array[0]
    assert oc.l2_dist_lots2one(mat1, data_array)[0] == 0, \
        "l2_dist_lots2one: distance between self should be 0"
    assert np.all(oc.l2_dist_lots2one(mat1, data_array) >= 0), \
        "l2_dist_lots2one: " +\
        "distance between two random matrices should be greater than"+\
        "or equal to 0"

    mat2 = mat1.copy()
    mat2[0,0] += 20
    assert oc.l2_dist_lots2one(mat2, data_array)[0] == 20, \
        "l2_dist_lots2one: static distance test didn't correctly work"


def test_l2_dist_matrix():
    """
    test l2_dist_matrix (basic)
    """
    np.random.seed(2)
    X_array = np.random.uniform(size = 5*3*4).reshape((5,3,4))
    X_array2 = X_array[1:3]

    square = oc.l2_dist_matrix(X_array)

    assert np.all(square == square.T) and np.all(np.diag(square) == 0) and \
        np.all(square >=0), \
        "expected square matrix is symmetric, and diags are 0, as non-neg"

    partial = oc.l2_dist_matrix(X_array2, X_array)

    assert np.all(square[1:3] == partial), \
        "slicing of data into the l2_dist_matrix works correctly"
