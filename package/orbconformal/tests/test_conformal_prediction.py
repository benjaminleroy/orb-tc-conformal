import numpy as np
import pandas as pd
import orbconformal as oc


def test_simulation_based_conformal():
    """
    test for simulation_based_conformal
    """
    eps = 10**(-14-1)
    diff_eps = 10**(-8-1)
    maxT = 500

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



    cs_z_50 = oc.simulation_based_conformal(true_z = z1,
                               simulated_z = z_multi2,
                               _sigma_string = "20%",
                               _small_ball_radius_string = "50%",
                               _maxT = maxT,
                               _eps = eps,
                               _diff_eps = diff_eps,
                               _parallel = 1,
                               verbose = False)

    cs_z_20 = oc.simulation_based_conformal(true_z = z1,
                               simulated_z = z_multi2,
                               _sigma_string = "20%",
                               _small_ball_radius_string = "20%",
                               _maxT = maxT,
                               _eps = eps,
                               _diff_eps = diff_eps,
                               _parallel = 1,
                               verbose = False)

    assert cs_z_50 > cs_z_20, \
        "we expect on average that a large small-ball-radius percentage," +\
        "so conformal score should be larger"

    # extremely static
    almost_zero = np.zeros(z1.shape)
    almost_zero[5,3] = 1

    static_almost_zero_change = np.concatenate([
        (z1 + i*almost_zero).reshape((1, z1.shape[0],z1.shape[1]))
             for i in np.array([.5, 1, 2, 5])])


    cs_z_static = oc.simulation_based_conformal(true_z = z1,
                               simulated_z = static_almost_zero_change,
                               _sigma_string = "50%",
                               _small_ball_radius_string = "70%",
                               _maxT = maxT,
                               _eps = eps,
                               _diff_eps = diff_eps,
                               _parallel = 1,
                               verbose = False)

    assert cs_z_static < 4, \
        "observation should be contained in some of prediction "+\
        "region if not too far away"

    static_almost_zero_change2 = np.concatenate([
        (z1 + i*almost_zero).reshape((1, z1.shape[0],z1.shape[1]))
             for i in np.array([.5,.75, 1, 2, 5])+10])


    cs_z_static3 = oc.simulation_based_conformal(true_z = z1,
                               simulated_z = static_almost_zero_change2,
                               _sigma_string = "20%",
                               _small_ball_radius_string = "70%",
                               _maxT = maxT,
                               _eps = eps,
                               _diff_eps = diff_eps,
                               _parallel = 1,
                               verbose = False)

    assert cs_z_static3 == 5, \
        "static observation shouldn't be covered by any level sets"
