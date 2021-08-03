import numpy as np
import plotly.graph_objects as go # for part of test not run...
import orbconformal as oc
import joblib

def test_meanshift_multidim_funct_single():
    """
    test meanshift_multidim_funct_single
    """
    # currently just checking if errors (not works correctly)

    X_array = np.random.uniform(size = 5*3*4).reshape((5,3,4))
    current_obs = X_array[0]
    sigma = 1
    eps = 1e-10
    maxT = 200
    out_t, out_step = oc.meanshift_multidim_funct_single(X_array = X_array,
                                    current_obs = current_obs,
                                    sigma = sigma, eps = eps,
                                    maxT = maxT)

    assert np.all(out_step <= X_array.max(axis = 0)) and \
        np.all(out_step >= X_array.min(axis = 0)), \
        "shouldn't converge outside the band of observations"


    # more complicated example (for visual purposes only):

    if False:

        np.random.seed(2021)

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

        shifts_vec = np.random.choice(np.random.uniform(size = 30)+ np.array([0]*15+[-2]*15),
                                      replace = False, size = 30)

        z_multi = np.concatenate([(z1+shifts_vec[i]).reshape(1,z1.shape[0], z1.shape[1])
                                 for i in np.arange(shifts_vec.shape[0])])



        my_current_obs = z_multi[0]

        complete_list_steps = [my_current_obs.copy()]

        for steps in np.arange(15):
            t, out =  oc.meanshift_multidim_funct_single(X_array = z_multi,
                                            current_obs = my_current_obs,
                                            sigma = 5, eps = 1e-10, maxT = 1)

            complete_list_steps.append(out.copy())
            my_current_obs = out.copy()

        # see if this seems to make sense to the visual eye:

        # all points:
        oc.vis_surfaces(z_multi).show()

        # a single convergence (for point 0):
        oc.vis_sequence_surface(np.array(complete_list_steps))


def test_meanshift_multidim_funct():
    """
    not really completed (checks if doesn't error)
    """
    parallel = 2
    n_cores = min(joblib.cpu_count(), parallel)

    X_array = np.random.uniform(size = 5*3*4).reshape((5,3,4))
    G_array = X_array.copy()
    sigma = 1
    eps = 1e-10
    maxT = 200

    out_t, out_step = oc.meanshift_multidim_funct(X_array = X_array,
                                    sigma = sigma, eps = eps,
                                    maxT = maxT, parallel = 1, verbose = False)

    out_t, out_step = oc.meanshift_multidim_funct(X_array = X_array,
                                    sigma = sigma, eps = eps,
                                    maxT = maxT, parallel = 2, verbose = False)

def test_mode_clustering():
    """
    test mode_clustering function
    """
    # data creation:
    np.random.seed(1)

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

    # single grouping
    shifts_vec_p = np.random.choice(np.random.normal(size = 100, scale = .25),
                                    replace = False, size = 100)

    z_p = np.concatenate([(z1+shifts_vec_p[i]).reshape(1,z1.shape[0], z1.shape[1])
                             for i in np.arange(shifts_vec_p.shape[0])])

    dist_mat = oc.l2_dist_matrix(z_p)
    sigma = np.quantile(dist_mat.ravel(), .45)

    n_groups, pd_out = oc.mode_clustering(X_array = z_p, sigma = sigma,
                                      eps = 1e-15, maxT = 150, diff_eps = 1e-05,
                                      parallel = 1,
                                      verbose = False)

    assert n_groups == 1 and np.all(pd_out.group == 0), \
        "for simple example, expected the number of clusters to be 1"

    idx_values = np.array(list(set(pd_out.index)))
    idx_values.sort()
    assert np.all(idx_values == np.arange(100, dtype = int)), \
        "index structure in returned pd_out isn't correct"

    # 2 groups
    np.random.seed(2022)

    shifts_vec2 = np.random.choice(np.random.uniform(size = 30)+ np.array([0]*15+[-2]*15),
                                  replace = False, size = 30)

    z_multi2 = np.concatenate([(z1+shifts_vec2[i]).reshape(1,z1.shape[0], z1.shape[1])
                             for i in np.arange(shifts_vec2.shape[0])])

    dist_mat2 = oc.l2_dist_matrix(z_multi2)
    sigma2 = np.quantile(dist_mat2.ravel(), .45)

    n_groups2, pd_out2 = oc.mode_clustering(X_array = z_multi2, sigma = sigma2,
                                      eps = 1e-15, maxT = 100, diff_eps = 1e-05,
                                      parallel = 1,
                                      verbose = False)

    assert n_groups2 == 2 and np.all([x in [0,1] for x in pd_out.group]), \
        "for simple example (2 groups), expected the number of clusters to be 2"

    idx_values2 = np.array(list(set(pd_out2.index)))
    idx_values2.sort()
    assert np.all(idx_values2 == np.arange(idx_values2.shape[0], dtype = int)), \
        "index structure in returned pd_out2 isn't correct"

def test_mode_clustering_check():
    # data creation:

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

    # single groups -------------
    np.random.seed(1)

    shifts_vec_p = np.random.choice(np.random.normal(size = 100, scale = .25),
                                    replace = False, size = 100)

    z_p = np.concatenate([(z1+shifts_vec_p[i]).reshape(1,z1.shape[0], z1.shape[1])
                             for i in np.arange(shifts_vec_p.shape[0])])

    dist_mat = oc.l2_dist_matrix(z_p)
    sigma = np.quantile(dist_mat.ravel(), .6)


    n_groups = oc.mode_clustering_check(X_array = z_p, sigma = sigma,
                                      eps_vec = np.array([1e-01,1e-05, 1e-10, 1e-15]),
                                      maxT = 30,
                                      diff_eps_vec = np.array([1e-03,1e-05,1e-08]),
                                      parallel = 1,
                                      verbose = False)

    n_groups2 = oc.mode_clustering_check(X_array = z_p, sigma = sigma,
                                      eps_vec = np.array([1e-01,1e-05, 1e-10, 1e-15]),
                                      maxT = 150,
                                      diff_eps_vec = np.array([1e-03,1e-05,1e-08]),
                                      parallel = 1,
                                      verbose = False)


    non_negative_rows_small = np.sum((n_groups == -1).sum(axis = 1) == 0)

    assert np.all(n_groups[:(non_negative_rows_small-1)] == \
        n_groups2[:(non_negative_rows_small-1)]), \
        "if maxT is large enough, eps outcomes should match given rest same"

    number_n1 =  (n_groups == -1).sum(axis = 1)
    number_n1_sorted = number_n1.copy()
    number_n1_sorted.sort()

    assert np.all(number_n1 == number_n1_sorted), \
        "number of negative values in matrix is "+\
        "monotonically increasing as eps decrease"

    # row values leq as we go down
    for r_idx in np.arange(n_groups.shape[0]-1):
        assert np.all(n_groups[r_idx] >= n_groups[r_idx+1]), \
            "row i's values should be less than row (i+1)'s' [smaller]"
        assert np.all(n_groups2[r_idx] >= n_groups2[r_idx+1]), \
            "row i's values should be less than row (i+1)'s' [larger]"

    # column values geq as we go right
    for c_idx in np.arange(n_groups.shape[1]-1):
        assert np.all(n_groups[:,c_idx] <= n_groups[:,c_idx+1]), \
            "column j's values should be less than row (j+1)'s' [smaller]"
        assert np.all(n_groups2[:,c_idx] <= n_groups2[:,c_idx+1]), \
            "row i's values should be less than row (i+1)'s' [larger]"

    assert np.all(n_groups2[-1] == 1), \
        "should find only a single group with highest eps"

    # 2 groups -------------------

    shifts_vec_p2 = np.random.choice(
                        np.concatenate(
                            [np.random.normal(size = 50,
                                              scale = .25, loc = -2),
                             np.random.normal(size = 50,
                                              scale = .25, loc = 2),
                            ]),
                                    replace = False, size = 100)

    z_p2 = np.concatenate([(z1+shifts_vec_p2[i]).reshape(1,z1.shape[0], z1.shape[1])
                             for i in np.arange(shifts_vec_p2.shape[0])])

    dist_mat2 = oc.l2_dist_matrix(z_p2)
    sigma2 = np.quantile(dist_mat2.ravel(), .6)


    n_groups_2 = oc.mode_clustering_check(X_array = z_p2, sigma = sigma2,
                                      eps_vec = np.array([1e-01,1e-05, 1e-10, 1e-15]),
                                      maxT = 30,
                                      diff_eps_vec = np.array([1e-03,1e-05,1e-08]),
                                      parallel = 1,
                                      verbose = False)

    n_groups2_2 = oc.mode_clustering_check(X_array = z_p2, sigma = sigma2,
                                      eps_vec = np.array([1e-01,1e-05, 1e-10, 1e-15]),
                                      maxT = 150,
                                      diff_eps_vec = np.array([1e-03,1e-05,1e-08]),
                                      parallel = 1,
                                      verbose = False)


    non_negative_rows_small2 = np.sum((n_groups == -1).sum(axis = 1) == 0)

    assert np.all(n_groups_2[:(non_negative_rows_small2-1)] == \
        n_groups2_2[:(non_negative_rows_small2-1)]), \
        "if maxT is large enough, eps outcomes should match given rest same"

    number_n1_2 =  (n_groups_2 == -1).sum(axis = 1)
    number_n1_sorted_2 = number_n1_2.copy()
    number_n1_sorted_2.sort()

    assert np.all(number_n1_2 == number_n1_sorted_2), \
        "number of negative values in matrix is "+\
        "monotonically increasing as eps decrease [2 groups case]"

    # row values leq as we go down
    for r_idx in np.arange(n_groups_2.shape[0]-1):
        assert np.all(n_groups_2[r_idx] >= n_groups_2[r_idx+1]), \
            "row i's values should be less than row (i+1)'s' "+\
            "[smaller, 2 groups]"
        assert np.all(n_groups2_2[r_idx] >= n_groups2_2[r_idx+1]), \
            "row i's values should be less than row (i+1)'s' "+\
            "[larger, 2 groups]"

    # column values geq as we go right
    for c_idx in np.arange(n_groups_2.shape[1]-1):
        assert np.all(n_groups_2[:,c_idx] <= n_groups_2[:,c_idx+1]), \
            "column j's values should be less than row (j+1)'s' "+\
            "[smaller, 2 groups]"
        assert np.all(n_groups2_2[:,c_idx] <= n_groups2_2[:,c_idx+1]), \
            "row i's values should be less than row (i+1)'s' "+\
            "[larger, 2 groups]"

    assert np.all(n_groups2_2[-1] == 2), \
        "should find only 2 groups with highest eps [2 groups]"
