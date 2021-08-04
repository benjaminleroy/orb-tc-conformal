import numpy as np
import orbconformal as oc

def test_calc_small_ball_rad_multidim_func():
    """
    test calc_small_ball_rad_multidim_func
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

    # basic check size of output and non-negativity

    pd_vec = np.arange(30)[::-1]/10

    r_vec = oc.calc_small_ball_rad_multidim_func(z_multi2, pd_vec)
    assert np.all(r_vec >= 0), \
        "expect all radius distances to be non-negative"

    # check ordering (and make sure it correctly orders observations)

    new_order = np.random.choice(30, size = 30, replace = False)
    pd_vec2 = pd_vec[new_order]

    r_vec2 = oc.calc_small_ball_rad_multidim_func(z_multi2, pd_vec2)
    r_vec2_2 = oc.calc_small_ball_rad_multidim_func(z_multi2[np.argsort(pd_vec2)[::-1]], pd_vec)

    assert np.all(r_vec2 == r_vec2_2), \
        "expected ordering to be correctly applied for radius calculation"


    # static example with just a single point that effects everything
    almost_zero = np.zeros(z1.shape)
    almost_zero[5,3] = 1

    static_almost_zero_change = np.concatenate([
        (z1 + i*almost_zero).reshape((1, z1.shape[0],z1.shape[1]))
             for i in np.array([0,.5, 1, 2, 5])])

    pd_vec_s = np.arange(5)[::-1]
    r_vec_s = oc.calc_small_ball_rad_multidim_func(static_almost_zero_change,
                                                pd_vec_s)
    assert np.all(r_vec_s == np.array([0,.5,.5,1,3])), \
        "static test 1 failed."


    pd_vec_s2 = np.array([5,1,2,3,4])

    r_vec_s2 = oc.calc_small_ball_rad_multidim_func(static_almost_zero_change,
                                                 pd_vec_s2)
    assert np.all(r_vec_s2 == np.array([0,5,3,3,3])), \
        "static test 2 failed."
