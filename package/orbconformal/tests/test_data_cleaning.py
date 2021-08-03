import numpy as np
import orbconformal as oc

def test_linear_interp():
    """
    static test for linear_interp
    """

    mat = np.arange(24*5).reshape(30,4)
    na_vec = np.array([0,0,0,0,.1,.1,.1,0,0,0,
                       0,0,0,1,0,0,0,0,1,1,
                       0,0,0,0,0,0,0,0,0,0])
    assert np.all(oc.linear_interp(mat, na_vec) == mat), \
        "expect similar linear progression to be correctly filled in"

    mat2 = np.array(mat.copy(), dtype = float)
    mat2[3:8,:] = mat2[3:8,:]*np.random.uniform(size = 1)

    assert np.all(np.abs(oc.linear_interp(mat2, na_vec) - mat2)<1e-14), \
        "expect similar linear progression to be correctly filled in"+\
        "(non unit jump check)"
