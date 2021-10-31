import numpy as np
import pandas as pd
import progressbar

from  .geometric_structure import calc_small_ball_rad_multidim_func, \
    when_contained
from .meanshift import mode_clustering
from .utils import check_character_percent
from .distances import l2_dist_matrix
from .psuedo_density import psuedo_density_multidim_func


def simulation_based_conformal(true_z,
                               simulated_z,
                               _sigma_string = "42.5%",
                               _small_ball_radius_string = "80%",
                               _maxT = 500,
                               _eps = 1e-10,
                               _diff_eps = 1e-15,
                               _parallel = 1,
                               verbose = False
                               ):
    """
    Global function for conformal prediction for multidimensional functions
    (assumed 2d surfaces repesented as matrices) with mode clustering

    Arguments
    ---------
    true_z : numpy array
        numpy array (n,m) of true function
    simulated_z : list
        list of numpy matrices (n,m)
    _sigma_string : string
        string of percentage for the sigma-quantile to define the sigma
        relative to the pairwise distances between simulations
    _small_ball_radius_string : string
        string of percentage for the small ball radius defined as the minimum
        spanning distance for the top X% of the simulations
    _maxT : int
        maximum steps for mode clustering alogrithm
    _eps : float
        absolute error between steps of the mode clustering alogirthm to decide
        if it has converted for a given step
    _diff_eps : float
        absolution difference between converged observation to be grouped
        together.
    _parallel : int
        if we should parallelize the mode clustering meanshift algorithm part
        (number of cores)
    verbose : boolean
        logic to be verbose about the progress of the function (TODO: needed?)
    """
    # convert percentages to proportions ----------
    _sigma_prop = check_character_percent(_sigma_string)
    _sbr_prop = check_character_percent(_small_ball_radius_string)

    # convert z_list to z_array -------------
    simulated_Z_array = np.array(simulated_z)
    B = simulated_Z_array.shape[0]

    # calculate sigma_val ------------------
    dmat = l2_dist_matrix(simulated_z)
    sigma_val = np.quantile(dmat, _sigma_prop)


    # overall psuedo-density values --------------
    pd_vec = psuedo_density_multidim_func(X_array=simulated_Z_array,
                                          sigma_string=_sigma_string)


    # sbr_vec = calc_small_ball_rad_multidim_func(simulated_Z_array,
    #                                             pd_vec,
    #                                             per_time=_per_time)
    # if not _per_time:
    #     sbr_val = sbr_vec[int(np.ceil(sbr_vec.shape[0]*_sbr_prop)-1)]
    # else:
    #     sbr_val = sbr_vec[int(np.ceil(sbr_vec.shape[0]*_sbr_prop)-1),:]


    # mode clustering ---------------

    num_groups, group_df = mode_clustering(simulated_Z_array,
                                            sigma = sigma_val,
                                            eps = _eps,
                                            maxT = _maxT,
                                            diff_eps = _diff_eps,
                                            parallel = _parallel,
                                            verbose = verbose)

    # defining indices per mode grouping -----------
    simulated_Z_idx_group_list = [
        np.arange(group_df.shape[0])[group_df.group == g_idx]
        for g_idx in np.arange(num_groups)]

    # small-ball-radius value --------------
    mode_radius_info_global = []
    mode_radius_info_per_time = []

    for m_idx in np.arange(num_groups):
        sbr_mat = calc_small_ball_rad_multidim_func(simulated_Z_array[simulated_Z_idx_group_list[m_idx],:,:],
                                                       pd_vec[simulated_Z_idx_group_list[m_idx]],
                                                       per_time=True)
        sbr_vec_per_time = sbr_mat[int(np.ceil(sbr_mat.shape[0]*_sbr_prop)-1),:]

        mode_radius_info_per_time.append(sbr_vec_per_time)


        sbr_start_vec = calc_small_ball_rad_multidim_func(simulated_Z_array[simulated_Z_idx_group_list[m_idx],:,:],
                                                       pd_vec[simulated_Z_idx_group_list[m_idx]],
                                                       per_time=False)

        sbr_vec_global = sbr_start_vec[int(np.ceil(sbr_start_vec.shape[0]*_sbr_prop)-1)]

        mode_radius_info_global.append(sbr_vec_global)




    # defining df with conformal score values (across modes) --------
    def inner_rank(df, name_string = "local_rank"):
        """rank pd vector"""
        order_of_ranking = np.argsort(df["pd"].values)
        ranking = np.zeros(df.shape[0], dtype = int)
        ranking[order_of_ranking] = np.arange(df.shape[0], dtype = int)

        df[name_string] = ranking
        return df

    full_ranking_from_mode_list_df = group_df.copy()
    full_ranking_from_mode_list_df["pd"] = pd_vec

    full_ranking_from_mode_list_df = inner_rank(full_ranking_from_mode_list_df,
                                                "global_rank")

    full_ranking_from_mode_list_df = full_ranking_from_mode_list_df.groupby(
        "group").apply(lambda df: inner_rank(df, "local_rank"))

    n_obs = full_ranking_from_mode_list_df.shape[0]
    end_df = pd.DataFrame(data = dict(index = np.array([np.inf]*num_groups),
                                  group = np.arange(num_groups, dtype = int),
                                  pd = np.array([-np.inf]*num_groups),
                                  global_rank = np.array([n_obs]*num_groups, dtype =int),
                                  local_rank = np.array([np.inf]*num_groups)))

    full_ranking_from_mode_list_df = pd.concat((full_ranking_from_mode_list_df,
                                                end_df))


    # defining pd values per mode grouping --------
    pd_vec_group_list = [pd_vec[idx_vec]
                            for idx_vec in simulated_Z_idx_group_list]

    mode_iter = np.arange(num_groups)
    if verbose:
        #print("containment per mode cluster:")
        bar = progressbar.ProgressBar()
        mode_iter = bar(mode_iter)

    true_cs_per_mode_per_time = pd.DataFrame()
    true_cs_per_mode_global = pd.DataFrame()

    for mode_idx in mode_iter:
        sim_Z_sub = simulated_Z_array[simulated_Z_idx_group_list[mode_idx]]
        pd_vec_sub = pd_vec_group_list[mode_idx]

        sbr_per_time = mode_radius_info_per_time[mode_idx]
        sbr_global = mode_radius_info_global[mode_idx]

        # per time -------------
        contained_idx_per_time = when_contained(true_z = true_z,
                                       simulated_Z_array = sim_Z_sub,
                                       pd_vec = pd_vec_sub,
                                       sbr = sbr_per_time,
                                       per_time = True)
        inner_true_cs_df_per_time = pd.DataFrame(data = dict(group = [mode_idx],
                                                    local_rank = 1.0*np.array([contained_idx_per_time])))

        true_cs_per_mode_per_time = pd.concat((true_cs_per_mode_per_time, inner_true_cs_df_per_time))

        # global ---------------
        contained_idx_global = when_contained(true_z = true_z,
                                       simulated_Z_array = sim_Z_sub,
                                       pd_vec = pd_vec_sub,
                                       sbr = sbr_global,
                                       per_time = False)
        inner_true_cs_df_global = pd.DataFrame(data = dict(group = [mode_idx],
                                                    local_rank = 1.0*np.array([contained_idx_global])))


        true_cs_per_mode_global = pd.concat((true_cs_per_mode_global, inner_true_cs_df_global))

    individual_info_per_time = pd.merge(inner_true_cs_df_per_time, full_ranking_from_mode_list_df,
             how = "left", on = ["group", "local_rank"])

    cs_score_per_time = B - individual_info_per_time["global_rank"].min()

    individual_info_global = pd.merge(inner_true_cs_df_global, full_ranking_from_mode_list_df,
         how = "left", on = ["group", "local_rank"])

    cs_score_global = B - individual_info_global["global_rank"].min()

    return cs_score_per_time, cs_score_global
