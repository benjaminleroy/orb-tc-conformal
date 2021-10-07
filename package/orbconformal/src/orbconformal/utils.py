import re
import numpy as np
import pandas as pd

def check_character_percent(x, name = "x"):
    """
    check if string is a percentage (and return that as proportion if so)

    Parameters
    ----------
    x : string
        element to examine
    name : string
        string of name of x (assumingly this is used inside a function, that
        may not call it "x")

    Returns
    -------
    float
        the proportion version of the percentage (if string was a percentage
        and meets other expectations)
    """
    assert len(re.findall("%$", x)) == 1, \
        f"if {name!r} is a character it must be '__%'."

    percentage = float(re.sub("%","", x))/100

    assert percentage <= 1 and percentage > 0, \
        f"if {name!r} is entered as a percent, " +\
        "it must be a percentage <= 100% and " +\
        "greater than 0%"

    return percentage


def _check_increasing(pd_vec):
    """
    check if pandas series vector is increasing

    Arguments
    --------
    pd_vec : pandas Series
        pandas series of float values

    Returns
    -------
    Errors if vector is not increasing, else returns None

    """
    diff_vals = pd_vec.diff()
    if np.sum(np.logical_not(np.isnan(diff_vals))) > 0:
        assert np.all(diff_vals[np.logical_not(np.isnan(diff_vals))] >= 0),                 "all values should be monotonically increasing"



def check_define_sample_cuts(df_all, length_tc_df):
    """
    Internal function to check output of define_sample_cut meets criteria

    Arguments
    ---------
    df_all : pd.DataFrame
        return from define_sample_cut
    length_tc_df : pd.DataFrame
        data frame inputed into define_sample_cut

    Details
    -------
    internal check to make sure function is working correctly... not sure
    best approach with this...

    criteria:
    1) only IDs with a single row can have NA in the starting index
    2) all ids in length_tc_df are seen at least once in df_all
    3) starting_index values are monotonically increasing when sorted by
        creation index (per ID)
    """

    check_df = df_all.groupby('ID').apply(
        lambda df: pd.DataFrame(data={"length": np.array([df.shape[0]]),
                                      "sum_is_na":np.sum(np.isnan(df.starting_index))})
    ).reset_index()

    check_df = check_df.drop(columns = "level_1")

    assert np.all(check_df.length[check_df.sum_is_na == 1] == 1),     "expect all rows with starting_index of NA to only have 1 value"

    assert np.all([x in np.array(check_df.ID) for x in np.array(length_tc_df.ID)]),     "all IDs in the overall data are seen at least once in the starting_idx data"


    df_all.sort_values(["ID","creation_index"]).groupby("ID").apply(
        lambda df: _check_increasing(df["starting_index"]))

