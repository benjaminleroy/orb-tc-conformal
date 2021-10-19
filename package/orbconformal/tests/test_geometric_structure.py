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

def test_calc_small_ball_rad_multidim_func2():
    """
    test calc_small_ball_rad_multidim_func, per_time=True
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

    r_mat = oc.calc_small_ball_rad_multidim_func(z_multi2, pd_vec,
                                                 per_time=True)

    assert np.all(r_mat >= 0), \
        "expect all radius distances to be non-negative"

    # check ordering (and make sure it correctly orders observations)

    new_order = np.random.choice(30, size = 30, replace = False)
    pd_vec2 = pd_vec[new_order]

    r_mat2 = oc.calc_small_ball_rad_multidim_func(z_multi2, pd_vec2,
                                                  per_time=True)
    r_mat2_2 = oc.calc_small_ball_rad_multidim_func(z_multi2[np.argsort(pd_vec2)[::-1]],
                                                    pd_vec, per_time=True)

    assert np.all(r_mat2 == r_mat2_2), \
        "expected ordering to be correctly applied for radius calculation"


    r_vec_global = oc.calc_small_ball_rad_multidim_func(z_multi2, pd_vec2,
                                                        per_time=False)

    assert np.all(r_vec_global == r_mat2.max(axis = 1)), \
        "expect per_time=False to just be the .max(axis=1) of the per_time=True"


    # static example with just a single point that effects everything
    almost_zero = np.zeros(z1.shape)
    almost_zero[5,3] = 1

    static_almost_zero_change = np.concatenate([
        (z1 + i*almost_zero).reshape((1, z1.shape[0],z1.shape[1]))
             for i in np.array([0,.5, 1, 2, 5])])

    pd_vec_s = np.arange(5)[::-1]
    r_mat_s = oc.calc_small_ball_rad_multidim_func(static_almost_zero_change,
                                                pd_vec_s, per_time=True)
    assert np.all(r_mat_s[:,5] == np.array([0,.5,.5,1,3])), \
        "static test 1 failed (per_time=True), should be same as static " +\
        "test (per_time=False), with 6th row of (per_time=True) the same as " +\
        "output of (per_time=False)."
    assert np.all(r_mat_s[:,:5] == 0) and \
        np.all(r_mat_s[:,6:] == 0), \
        "static test 1 failed (per_time=True) - except for 6th row, " +\
        "(per_time=True) values should be 0"


    pd_vec_s2 = np.array([5,1,2,3,4])

    r_mat_s2 = oc.calc_small_ball_rad_multidim_func(static_almost_zero_change,
                                                 pd_vec_s2, per_time=True)

    assert np.all(r_mat_s2[:,5] == np.array([0,5,3,3,3])), \
        "static test 2 failed (per_time=True), should be same as static " +\
        "test (per_time=False), with 6th row of (per_time=True) the same as " +\
        "output of (per_time=False)."

    assert np.all(r_mat_s2[:,:5] == 0) and \
        np.all(r_mat_s[:,6:] == 0), \
        "static test 2 failed (per_time=True) - except for 6th row, " +\
        "(per_time=True) values should be 0"


def test_when_contained():
    """
    test when_contained
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

    true_z = z1 + np.random.uniform(size=1)

    pd_vec = np.arange(z_multi2.shape[0])/z_multi2.shape[0]

    sbr = .3


    assert oc.geometric_structure.when_contained(true_z,
                           z_multi2,
                           pd_vec,
                           sbr,
                           verbose=False) == 0, \
        "true_z completely contained right away if within sims and very large sbr"

    true_z2 = z1 - .2

    assert oc.geometric_structure.when_contained(true_z2,
                   z_multi2,
                   pd_vec,
                   sbr,
                   verbose=False) > 0,\
        "true_z can be quickly contained is sbr is large (but not inside the sims)"

    true_z3 = z1 - .3

    assert oc.geometric_structure.when_contained(true_z3,
                   z_multi2,
                   pd_vec,
                   sbr,
                   verbose=False) is np.inf,\
        "true_z cannot be contained if outside sbr for all simulations"



    # static test
    almost_zero = np.zeros(z1.shape)
    almost_zero[5,3] = 1

    static_almost_zero_change = np.concatenate([
        (z1 + i*almost_zero).reshape((1, z1.shape[0],z1.shape[1]))
             for i in np.array([0,.5, 1, 2, 5])])

    pd_vec_s = np.arange(5)

    sbr_v = [.4,.8,1.1,2.1,2]
    expected_when_contained = [4,3,2,1,1]

    when_contained_list = list()
    for sbr in sbr_v:
        when_contained_list.append(oc.geometric_structure.when_contained(z1,
                   static_almost_zero_change,
                   pd_vec_s,
                   sbr,
                   verbose=False))

    assert np.all(expected_when_contained == when_contained_list), \
        "static containment incorrect"

def test_when_contained2():
    """
    test when_contained, per_time=True
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


    # although random - this test's simulations are just a single value plus
    # z1, so the outcomes should be the same as test_when_contained when
    # per_time = False

    np.random.seed(2022)

    shifts_vec2 = np.random.choice(np.random.uniform(size = 30)+ \
                                       np.array([0]*15+[-2]*15),
                                  replace = False, size = 30)

    z_multi2 = np.concatenate([(z1+shifts_vec2[i]).reshape(1,z1.shape[0],
                                                           z1.shape[1])
                             for i in np.arange(shifts_vec2.shape[0])])

    true_z = z1 + np.random.uniform(size=1)

    pd_vec = np.arange(z_multi2.shape[0])/z_multi2.shape[0]

    sbr = np.array([.3]*z1.shape[0])


    assert oc.geometric_structure.when_contained(true_z,
                           z_multi2,
                           pd_vec,
                           sbr,
                           per_time=True,
                           verbose=False) == 0, \
        "true_z completely contained right away if within sims and very large sbr"

    true_z2 = z1 - .2

    assert oc.geometric_structure.when_contained(true_z2,
                   z_multi2,
                   pd_vec,
                   sbr,
                   per_time=True,
                   verbose=False) > 0,\
        "true_z can be quickly contained is sbr is large (but not inside the sims)"

    true_z3 = z1 - .3

    assert oc.geometric_structure.when_contained(true_z3,
                   z_multi2,
                   pd_vec,
                   sbr,
                   per_time=True,
                   verbose=False) is np.inf,\
        "true_z cannot be contained if outside sbr for all simulations"


    # specific structures from per_time (expect assertion errors)
    correct_error0 = False

    try:
        oc.geometric_structure.when_contained(true_z,
                           z_multi2,
                           pd_vec,
                           sbr,
                           per_time=False,
                           verbose=False)
    except AssertionError:
        correct_error0 = True

    assert correct_error0, \
        "expected error if per_time=False, but sbr is vector"

    correct_error1 = False
    try:
        oc.geometric_structure.when_contained(true_z,
                           z_multi2,
                           pd_vec,
                           sbr = .3,
                           per_time=True,
                           verbose=False)
    except AssertionError:
        correct_error1 = True

    assert correct_error1, \
        "expected error if per_time=True, but sbr is float"


    # static test
    almost_zero = np.zeros(z1.shape)
    almost_zero[5,3] = 1

    static_almost_zero_change = np.concatenate([
        (z1 + i*almost_zero).reshape((1, z1.shape[0],z1.shape[1]))
             for i in np.array([0,.5, 1, 2, 5])])

    pd_vec_s = np.arange(5)

    sbr_v = [.4,.8,1.1,2.1,2]
    expected_when_contained = [4,3,2,1,1]

    # old set of tests
    when_contained_list = list()
    for sbr in sbr_v:
        sbr_vector = np.array([sbr]*z1.shape[0])
        when_contained_list.append(oc.geometric_structure.when_contained(z1,
                   static_almost_zero_change,
                   pd_vec_s,
                   sbr_vector,
                   per_time=True,
                   verbose=False))

    assert np.all(expected_when_contained == when_contained_list), \
        "static containment incorrect"


    # same results, but different radius values for outside

    when_contained_list_0 = list()
    when_contained_list_inf = list()

    for sbr in sbr_v:
        sbr_vector_0 = np.zeros(z1.shape[0])
        sbr_vector_0[5] = sbr
        sbr_vector_inf = np.inf*np.ones(z1.shape[0])
        sbr_vector_inf[5] = sbr
        when_contained_list_0.append(oc.geometric_structure.when_contained(z1,
                   static_almost_zero_change,
                   pd_vec_s,
                   sbr_vector_0,
                   per_time=True,
                   verbose=False))
        when_contained_list_inf.append(oc.geometric_structure.when_contained(z1,
                   static_almost_zero_change,
                   pd_vec_s,
                   sbr_vector_inf,
                   per_time=True,
                   verbose=False))

    assert np.all(when_contained_list_0 == expected_when_contained), \
        "static containment incorrect (per_time=True, rest = 0)"
    assert np.all(when_contained_list_inf == expected_when_contained), \
        "static containment incorrect (per_time=True, rest = inf)"




