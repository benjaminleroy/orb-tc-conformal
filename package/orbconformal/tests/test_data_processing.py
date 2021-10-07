import numpy as np
import pandas as pd
import orbconformal as oc

def test_randomize_range():
    """
    test randomize_range function
    """
    for _ in range(20):
        l = np.random.choice(50) + 50
        wl = np.random.choice(20) + 20
        idx = np.array([oc.randomize_range(length_range = l, window_length = wl)
                           for _ in range(100)])
        assert np.all(np.logical_and(idx >= 0, idx < l - wl)), \
            "error with randimize_range for l=%i, wl=%i" % (l, wl)



def test_update_from_previous_starting_point():
    """
    test update_from_previous_starting_point (single test)
    """
    df = pd.DataFrame(data = {"ID":["A"]*3,
                              "length":[852, 441, 214],
                              "starting_index":[303, 119,
                                               123]})
    df2 = oc.update_from_previous_starting_point(df, col_length="length",
                                              col_starting = "starting_index")

    df2_expected = pd.DataFrame(data = {"ID":["A"]*3,
                              "length":[852, 441, 214],
                              "starting_index":[303,
                                                852-441+119,
                                               852-214+123]})
    assert np.all(df2.reset_index() == df2_expected.reset_index()), \
        "incorrect correct update of starting point relative to previous obs"



def test_df_subset():
    """
    test df_subset, static
    """
    df = pd.DataFrame(data ={"ID": ["A"]*20,
                             "val": np.random.uniform(size=20)})

    df2 = df.copy()
    df2.index = np.arange(5,5+20, dtype = int)


    length_tc_df = pd.DataFrame(data = {"ID": np.array(["A"]),
                                       "starting_index":np.array([3])})

    small_df = oc.df_subset(df, length_tc_df, window = 5)

    small_df2 = oc.df_subset(df2, length_tc_df, window = 5)

    assert np.all(small_df.drop(columns = "creation_index") ==                      df[3:8].reset_index(drop = True)),         "expected slice 3-8 of the rows errored"

    assert np.all(small_df2.drop(columns = "creation_index") ==                  df2[3:8].reset_index(drop = True)),         "expected slice 3-8 of the rows errored (different index)"

    # multiple cuts
    length_tc_df_m = pd.DataFrame(data = {"ID": np.array(["A","A"]),
                                       "starting_index":np.array([3, 10])})

    small_df_m = oc.df_subset(df,length_tc_df_m,window = 5)

    small_df_m_expected = pd.concat([df[3:8].reset_index(drop =True),
                                     df[10:15].reset_index(drop =True)])
    small_df_m_expected["creation_index"] = [0]*5+[1]*5
    assert np.all(small_df_m.reset_index() ==                      small_df_m_expected.reset_index()),         "expected 2 slices for sample ID: 3-8 and 10-15 errored"


    # multiple through a lambda function
    df2_2 = df2.copy()
    df2_2.ID = "B"
    length_tc_df3 = pd.DataFrame(data = {"ID": np.array(["A", "B"]),
                                       "starting_index":np.array([3,2])})
    df3 = pd.concat([df,df2_2])

    small_combo = df3.groupby("ID").apply(
        lambda df: oc.df_subset(df, length_tc_df3, window = 5)).reset_index(drop = True)

    #small_combo.reset_index(drop = True)
    assert np.all(small_combo[small_combo.ID == "A"] == small_df) and          np.all(small_combo[small_combo.ID == "B"].drop(
                    columns = "creation_index").reset_index(drop=True) == \
                    df2_2[2:7].reset_index(drop=True)), \
         "error in multiple attempt (apply(lambda...))"

    # multiple through a lambda function with multiple cuts
    df2_2 = df2.copy()
    df2_2.ID = "B"
    length_tc_df3 = pd.DataFrame(data = {"ID": np.array(["A", "A", "B"]),
                                       "starting_index":np.array([3,10,2])})
    df3 = pd.concat([df,df2_2])

    small_combo = df3.groupby("ID").apply(
        lambda df: oc.df_subset(df, length_tc_df3, window = 5)).reset_index(drop = True)

    assert np.all(small_combo[small_combo.ID == "A"].reset_index(drop = True) ==                  small_df_m_expected.reset_index(drop=True)) and          np.all(small_combo[small_combo.ID == "B"].drop(
                    columns = "creation_index").reset_index(drop=True) == \
                    df2_2[2:7].reset_index(drop=True)), \
         "error in multiple attempt (apply(lambda...))"
