import numpy as np
import pandas as pd
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



def test_remove_init_and_final_missing_rows_for_interp():
    """
    test remove_init_and_final_missing_rows_for_interp
    """

    size_cols = ["s1","s2","s3"]
    rad_cols = ["r1","r2","r3"]

    # single NA s
    info_3_top = np.random.uniform(size = (10,3))
    info_3_top[0,:] = np.nan

    info_3_bottom = np.random.uniform(size = (10,3))
    info_3_bottom[-1,:] = np.nan



    df_s_NA_top = pd.DataFrame(data = {"s1": info_3_top[:,0],
                             "s2": info_3_top[:,1],
                             "s3": info_3_top[:,2],
                             "r1": np.random.uniform(size=10),
                             "r2": np.random.uniform(size=10),
                             "r3": np.random.uniform(size=10),
                             "random_info": np.array(["A","B"]*5)},
                     columns = ["random_info","s1","s2","s3",
                               "r1","r2","r3"])

    df_s_NA_bottom = pd.DataFrame(data = {"s1": info_3_bottom[:,0],
                             "s2": info_3_bottom[:,1],
                             "s3": info_3_bottom[:,2],
                             "r1": np.random.uniform(size=10),
                             "r2": np.random.uniform(size=10),
                             "r3": np.random.uniform(size=10),
                             "random_info": np.array(["A","B"]*5)},
                     columns = ["random_info","s1","s2","s3",
                               "r1","r2","r3"])

    df_s_NA_top_rm = oc.remove_init_and_final_missing_rows_for_interp(df_s_NA_top,
                                                                  size_cols,
                                                                  rad_cols)


    assert np.all(df_s_NA_top_rm.reset_index(drop = True) == \
                  df_s_NA_top[1:].reset_index(drop = True)), \
        "expected top row to be dropped (s)"
    df_s_NA_bottom_rm = oc.remove_init_and_final_missing_rows_for_interp(
                                                              df_s_NA_bottom,
                                                              size_cols,
                                                              rad_cols)

    assert np.all(df_s_NA_bottom_rm.reset_index(drop = True) == \
                  df_s_NA_bottom[:-1].reset_index(drop = True)), \
        "expected bottom row to be dropped (s)"

    # single NA r
    df_r_NA_top = pd.DataFrame(data = {"s1": np.random.uniform(size=10),
                             "s2": np.random.uniform(size=10),
                             "s3": np.random.uniform(size=10),
                             "r1": info_3_top[:,0],
                             "r2": info_3_top[:,1],
                             "r3": info_3_top[:,2],
                             "random_info": np.array(["A","B"]*5)},
                     columns = ["random_info","s1","s2","s3",
                               "r1","r2","r3"])

    df_r_NA_bottom = pd.DataFrame(data = {"s1": np.random.uniform(size=10),
                             "s2": np.random.uniform(size=10),
                             "s3": np.random.uniform(size=10),
                             "r1": info_3_bottom[:,0],
                             "r2": info_3_bottom[:,1],
                             "r3": info_3_bottom[:,2],
                             "random_info": np.array(["A","B"]*5)},
                     columns = ["random_info","s1","s2","s3",
                               "r1","r2","r3"])

    df_r_NA_top_rm = oc.remove_init_and_final_missing_rows_for_interp(df_r_NA_top,
                                                                  size_cols,
                                                                  rad_cols)


    df_r_NA_bottom_rm = oc.remove_init_and_final_missing_rows_for_interp(
                                                              df_r_NA_bottom,
                                                              size_cols,
                                                              rad_cols)

    assert np.all(df_r_NA_top_rm.reset_index(drop = True) == \
                  df_r_NA_top[1:].reset_index(drop = True)), \
        "expected top row to be dropped (r)"

    assert np.all(df_r_NA_bottom_rm.reset_index(drop = True) ==  \
                  df_r_NA_bottom[:-1].reset_index(drop = True)), \
        "expected bottom row to be dropped (r)"



    # top and bottom s
    info_3_both = np.random.uniform(size = (10,3))
    info_3_both[0:2,:] = np.nan
    info_3_both[-1,:] = np.nan

    df_s_NA_both = pd.DataFrame(data = {"s1": info_3_both[:,0],
                             "s2": info_3_both[:,1],
                             "s3": info_3_both[:,2],
                             "r1": np.random.uniform(size=10),
                             "r2": np.random.uniform(size=10),
                             "r3": np.random.uniform(size=10),
                             "random_info": np.array(["A","B"]*5)},
                     columns = ["random_info","s1","s2","s3",
                               "r1","r2","r3"])
    df_s_NA_both_rm = oc.remove_init_and_final_missing_rows_for_interp(
                                                          df_s_NA_both,
                                                          size_cols,
                                                          rad_cols)

    assert np.all(df_s_NA_both_rm.reset_index(drop = True) == \
                  df_s_NA_both[2:-1].reset_index(drop = True)), \
        "expected top 2 and bottom 1 row dropped (s top2, s bottom1)"


    # top and bottom r
    df_r_NA_both = pd.DataFrame(data = {"s1": np.random.uniform(size=10),
                             "s2": np.random.uniform(size=10),
                             "s3": np.random.uniform(size=10),
                             "r1": info_3_both[:,0],
                             "r2": info_3_both[:,1],
                             "r3": info_3_both[:,2],
                             "random_info": np.array(["A","B"]*5)},
                     columns = ["random_info","s1","s2","s3",
                               "r1","r2","r3"])

    df_r_NA_both_rm = oc.remove_init_and_final_missing_rows_for_interp(
                                                          df_r_NA_both,
                                                          size_cols,
                                                          rad_cols)

    assert np.all(df_r_NA_both_rm.reset_index(drop = True) ==  \
                  df_r_NA_both[2:-1].reset_index(drop = True)), \
        "expected top 2 and bottom 1 row dropped (r)"



    # s and r (1)
    info_3_top2 = np.random.uniform(size = (10,3))
    info_3_top2[0:2,:] = np.nan

    df_sr_NA = pd.DataFrame(data = {"s1": info_3_top2[:,0],
                         "s2": info_3_top2[:,1],
                         "s3": info_3_top2[:,2],
                         "r1": info_3_bottom[:,0],
                         "r2": info_3_bottom[:,1],
                         "r3": info_3_bottom[:,2],
                         "random_info": np.array(["A","B"]*5)},
                 columns = ["random_info","s1","s2","s3",
                           "r1","r2","r3"])


    df_sr_NA_both_rm = oc.remove_init_and_final_missing_rows_for_interp(
                                                          df_sr_NA,
                                                          size_cols,
                                                          rad_cols)

    assert np.all(df_sr_NA_both_rm.reset_index(drop = True) == \
                  df_sr_NA[2:-1].reset_index(drop = True)), \
        "expected top 2 and bottom 1 row dropped (s top2, r bottom1)"

    # s and r (2)
    info_3_top2 = np.random.uniform(size = (10,3))
    info_3_top2[0:2,:] = np.nan

    df_sr_NA = pd.DataFrame(data = {"s1": info_3_bottom[:,0],
                         "s2": info_3_bottom[:,1],
                         "s3": info_3_bottom[:,2],
                         "r1": info_3_top2[:,0],
                         "r2": info_3_top2[:,1],
                         "r3": info_3_top2[:,2],
                         "random_info": np.array(["A","B"]*5)},
                 columns = ["random_info","s1","s2","s3",
                           "r1","r2","r3"])


    df_sr_NA_both_rm = oc.remove_init_and_final_missing_rows_for_interp(
                                                          df_sr_NA,
                                                          size_cols,
                                                          rad_cols)

    assert np.all(df_sr_NA_both_rm.reset_index(drop = True) == \
                    df_sr_NA[2:-1].reset_index(drop = True)), \
        "expected top 2 and bottom 1 row dropped (s bottom1, r top2)"

    # s and r (3)
    info_3_top2 = np.random.uniform(size = (10,3))
    info_3_top2[0:2,:] = np.nan

    df_sr_NA = pd.DataFrame(data = {"s1": info_3_top2[:,0], # not completely na as well
                         "s2": info_3_bottom[:,1],
                         "s3": info_3_bottom[:,2],
                         "r1": info_3_bottom[:,0],
                         "r2": info_3_bottom[:,1],
                         "r3": info_3_bottom[:,2],
                         "random_info": np.array(["A","B"]*5)},
                 columns = ["random_info","s1","s2","s3",
                           "r1","r2","r3"])


    df_sr_NA_both_rm = oc.remove_init_and_final_missing_rows_for_interp(
                                                          df_sr_NA,
                                                          size_cols,
                                                          rad_cols)
    assert  np.all(
        np.logical_or(df_sr_NA_both_rm == df_sr_NA[:-1],
                      np.logical_and(df_sr_NA_both_rm.isna(),
                                     df_sr_NA[:-1].isna()))), \
        "expected bottom 1 row dropped (s bottom0, r bottom1), "+\
        "not completely na top"

    # s and r (4)
    info_3_top2 = np.random.uniform(size = (10,3))
    info_3_top2[0:2,:] = np.nan

    df_sr_NA = pd.DataFrame(data = {"s1": info_3_top2[:,0],
                         "s2": info_3_top2[:,1],
                         "s3": info_3_top2[:,2],
                         "r1": info_3_top[:,0],
                         "r2": info_3_top[:,1],
                         "r3": info_3_top[:,2],
                         "random_info": np.array(["A","B"]*5)},
                 columns = ["random_info","s1","s2","s3",
                           "r1","r2","r3"])


    df_sr_NA_both_rm = oc.remove_init_and_final_missing_rows_for_interp(
                                                          df_sr_NA,
                                                          size_cols,
                                                          rad_cols)

    assert np.all(df_sr_NA_both_rm.reset_index(drop = True) == \
                  df_sr_NA[2:].reset_index(drop = True)), \
        "expected top 2 row dropped (s top2, r top1)"

def test_remove_init_and_final_missing_rows_for_interp_group():
    """
    test for group application and random NA values with
    remove_init_and_final_missing_rows_for_interp
    """


    df_group = pd.DataFrame(data = {"ID": ["A"]*5+["B"]*6,
                                    "s1": [np.nan]+list(np.random.uniform(size=9))+[np.nan],
                                    "s2": [np.nan]*2+list(np.random.uniform(size=9)),
                                    "r1": [np.nan]+list(np.random.uniform(size=9))+[np.nan],
                                    "r2": list(np.random.uniform(size=9)) + 2*[np.nan],
                                    "junk1": np.random.uniform(size=11)},
                            columns = ["ID", "s1","s2","r1", "r2", "junk1"])

    size_col_gtest = ["s1", "s2"]
    rad_col_gtest = ["r1",  "r2"]


    df1 = df_group[df_group.ID == "A"]

    df1_rm = oc.remove_init_and_final_missing_rows_for_interp(df1, size_col_gtest,
                                                        rad_col_gtest)

    assert np.all(
            np.logical_or(df1[1:].reset_index(drop = True) ==\
                            df1_rm.reset_index(drop =True),
                          np.logical_and(df1[1:].reset_index(drop = True).isna(),
                                         df1_rm.reset_index(drop = True).isna()))), \
        "df1 should have first row removed."


    df1_1_rm = oc.remove_init_and_final_missing_rows_for_interp(df1, size_col_gtest,
                                                        rad_col_gtest,
                                                          .3) # aka anything above 0 in this case

    assert np.all(
            np.logical_or(df1[2:].reset_index(drop = True) ==\
                            df1_1_rm.reset_index(drop =True),
                          np.logical_and(df1[2:].reset_index(drop = True).isna(),
                                         df1_1_rm.reset_index(drop = True).isna()))), \
        "df1 should have first two row removed. (cutoff != 1)"


    df_smaller = df_group.groupby("ID").apply(
        lambda df: oc.remove_init_and_final_missing_rows_for_interp(df, size_col_gtest,
                                                        rad_col_gtest)).reset_index(
        drop = True) #should error with a single NA value...


    assert np.all(
            np.logical_or(df_group[1:-1].reset_index(drop = True) ==\
                            df_smaller.reset_index(drop =True),
                          np.logical_and(df_group[1:-1].reset_index(drop = True).isna(),
                                         df_smaller.reset_index(drop = True).isna()))), \
        "df_group should have first and last row removed. (cutoff == 1)"

def test_remove_init_and_final_missing_rows_for_interp_full_NA():
    """
    test remove_init_and_final_missing_rows_for_interp with only NAs
    """

    df_s_NA = pd.DataFrame(data = {"s1": [np.nan]*11,
                                    "s2": [np.nan]*11,
                                    "r1": [np.nan]*2+list(np.random.uniform(size=9)),
                                    "r2": list(np.random.uniform(size=9)) + 2*[np.nan],
                                    "junk1": np.random.uniform(size=11)},
                            columns = ["ID", "s1","s2","r1", "r2", "junk1"])

    size_col_gtest = ["s1", "s2"]
    rad_col_gtest = ["r1",  "r2"]

    assert oc.remove_init_and_final_missing_rows_for_interp(df_s_NA,
                                                         size_col_gtest,
                                                         rad_col_gtest).shape[0] == 0,\
        "if size or rad have all NAs we expect 0 rows to return (s)"

    df_r_NA = pd.DataFrame(data = {"r1": [np.nan]*11,
                                    "r2": [np.nan]*11,
                                    "s1": [np.nan]*2+list(np.random.uniform(size=9)),
                                    "s2": list(np.random.uniform(size=9)) + 2*[np.nan],
                                    "junk1": np.random.uniform(size=11)},
                            columns = ["ID", "s1","s2","r1", "r2", "junk1"])

    size_col_gtest = ["s1", "s2"]
    rad_col_gtest = ["r1",  "r2"]

    assert oc.remove_init_and_final_missing_rows_for_interp(df_r_NA,
                                                         size_col_gtest,
                                                         rad_col_gtest).shape[0] == 0,\
        "if size or rad have all NAs we expect 0 rows to return (r)"
