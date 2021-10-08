import numpy as np
import pandas as pd
import random

def randomize_range(length_range, window_length = 36):
    """
    select start date uniformly while still giving correct number of units

    Arguments
    ---------
    length_range: int
        number of intervals in the full length of data
    window_length: int
        desired number of intervals in window length (default is 36)

    Returns
    -------
    int of the starting position that still allows for a sample of the
        desired window length to be obtained
    """
    max_start_value = length_range - window_length
    if max_start_value < 0:
        return None
    elif max_start_value == 0:
        return 0
    else:
        return int(np.random.choice(max_start_value))



def update_from_previous_starting_point(df, col_length, col_starting):
    """
    update data frame with starting points relative to length left to true
    global time

    Arguments
    ---------
    df : pd.DataFrame
        data frame with length able to selected from (col_length) and the actual
        point selected. This data frame should be for a single TC and be ordered
    col_length : str
        string of column name for length available to select from
    col_starting : str
        string of column name of starting index relative to length available

    Returns
    -------
    updated df where col_starting is now not a localized starting index but a
    global one
    """
    df = df.copy()
    df["diff_length"] = df[[col_length]].diff(periods=1)
    df["diff_length"] = df["diff_length"].astype(pd.Int64Dtype())
    df["logic"] = np.logical_not(np.isnan(df.diff_length))
    # converting from difference to culumative sum of past lengths
    df["diff_length"][df["logic"]] = np.cumsum(df["diff_length"][df["logic"]])

    df.loc[df["logic"], col_starting] = df[col_starting][df.logic] -1*        df.diff_length[df.logic]
    df = df.drop(columns = ["diff_length", "logic"])

    return(df)




def define_sample_cuts(length_tc_df, seed, window=60, dependent_gap=48):
    """
    defines samples cuts relative to length_tc_df

    Parameters
    ----------
    length_tc_df : pd.DataFrame
        columns "ID" and "length" (where length is an integer)
    seed : int
        seed for random.seed and np.random.seed setting
    window : int
        size of required window

    Returns
    -------
    df_large : pd.DataFrame
    """
    random.seed(seed)
    np.random.seed(seed)

    # initial calculation
    df_all = length_tc_df.copy()
    df_all["starting_index"] = [randomize_range(x, window)
                                    for x in df_all.length]
    df_all.starting_index = df_all.starting_index.convert_dtypes(int)
    df_all["creation_index"] = 0

    info_df_list = [df_all]
    last_df = df_all

    done = False
    index = 1

    while not done:
        left_over = last_df.length - window - dependent_gap - last_df.starting_index
        left_over[left_over < 0] = np.nan

        # creating a new df with info
        new_df = pd.DataFrame(data = {"ID":last_df.ID,
                             "length":left_over},
                     columns = ["ID", "length"])

        new_starting_index = np.nan * np.ones(new_df.shape[0])

        new_starting_index[np.logical_not(np.isnan(new_df.length))] =         [randomize_range(x, window)
         for x in new_df.length[np.logical_not(np.isnan(new_df.length))]]

        new_df["starting_index"] = new_starting_index
        new_df.starting_index = new_df.starting_index.convert_dtypes(int)

        new_df = new_df[np.logical_not(np.isnan(new_df.starting_index))]
        new_df["creation_index"] = index


        info_df_list.append(new_df)
        last_df = new_df

        index += 1
        if np.all(np.isnan(left_over)):
            done = True

    df_large = pd.concat(info_df_list)

    df_large = df_large.sort_values(["ID", "creation_index"])

    df_out = df_large.groupby("ID").apply(
        lambda df: update_from_previous_starting_point(df, "length",
                                                       "starting_index"))

    # final logic check
    check_define_sample_cuts(df_out, length_tc_df)

    return df_out


def df_subset(df, length_tc_df, window):
    """
    subset of dataframe (using it's 'ID' index) to identify size using another
    df

    Arguments
    ---------
    df : pd.DataFrame
        data frame to subset (has a single value in an 'ID' column)
    length_tc_df : pd.DataFrame
        data frame that contains columns 'ID', 'starting_index' for subsets,
        this can include mutliple 'starting_index' values for a single 'ID'
    window : int
        length of window after starting point (assumes that df has that much
        length - else will error)

    Returns
    -------
    subset of df starting at 'starting_index' and of length window for every
    row of length_tc_df. Includes a 'creation_index' associated with the
    relative row of length_tc_df given "ID" (0 - (max length of subset- 1))

    Details
    -------
    Doesn't check if associated window allows for cuts for subsets same ID to
        overlap
    """
    assert df['ID'].unique().shape[0] == 1, \
        "expected df to have a single unique value in the 'ID' column"

    info = length_tc_df.loc[length_tc_df["ID"] == df['ID'].unique()[0],:]

    if pd.isna(np.array(info.starting_index)[0]):
        warnings.warn("no starting index value for %s" % np.array(info.ID)[0],
                     stacklevel=2)
        return pd.DataFrame(columns = df.columns)

    df_list = []
    for idx in np.arange(info.shape[0], dtype = int):
        starting_index = np.array(info.starting_index)[idx]
        subset_df = df.reset_index(drop = True).loc[list(np.arange(starting_index,
                                               starting_index+window)),:].reset_index(drop=True)
        subset_df["creation_index"] = idx

        df_list.append(subset_df)

    df_out = pd.concat(df_list)

    return df_out
