import numpy as np
import orbconformal as oc

def test_psuedo_density_multidim_func():
    """
    test psuedo_density_multidim_func (super basic)
    """
    z1 = np.array([
        [8.83,8.89,8.81,8.87,8.9,8.87],
        [8.89,8.94,8.85,8.94,8.96,8.92],
        [8.84,8.9,8.82,8.92,8.93,8.91],
        [8.79,8.85,8.79,8.9,8.94,8.92],
        [8.79,8.88,8.81,8.9,8.95,8.92],
        [8.8,8.82,8.78,8.91,8.94,8.92],
        [8.75,8.78,8.77,8.91,8.95,8.92],
        [8.8,8.8,8.77,8.91,8.95,8.94],
        [8.74,8.81,8.76,8.93,8.98,8.99],
        [8.89,8.99,8.92,9.1,9.13,9.11],
        [8.97,8.97,8.91,9.09,9.11,9.11],
        [9.04,9.08,9.05,9.25,9.28,9.27],
        [9,9.01,9,9.2,9.23,9.2],
        [8.99,8.99,8.98,9.18,9.2,9.19],
        [8.93,8.97,8.97,9.18,9.2,9.18]
    ])

    np.random.seed(2022)

    shifts_vec2 = np.random.choice(np.random.uniform(size = 30)+ \
                                       np.array([0]*15+[-2]*15),
                                  replace = False, size = 30)

    z_multi2 = np.concatenate([(z1+shifts_vec2[i]).reshape(1,z1.shape[0],
                                                           z1.shape[1])
                             for i in np.arange(shifts_vec2.shape[0])])


    X_array = z_multi2
    Y_array = z_multi2[[1,5,7,10]]

    pd_all = oc.psuedo_density_multidim_func(X_array = X_array)
    pd_sub = oc.psuedo_density_multidim_func(X_array = X_array,
                                             Y_array = Y_array)

    assert np.all(pd_all[[1,5,7,10]] == pd_sub), \
        "expect psuedo_densities based on the same X_array to return the "+\
        "same regardless if Y_array is non or not."

    assert np.all(pd_all >= 0), \
        "psuedo-density value should be geq 0"
