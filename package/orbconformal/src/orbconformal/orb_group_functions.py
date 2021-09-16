import numpy as np
from functools import lru_cache
import os
import pandas as pd

#
# these functions came from code written by Pavel Khnokhlov and Trey McNeely
#
# they have been rewritten for some cleaner comments

#---------------------------
# Feature creation
#---------------------------


def distance_from_coord(lat1, lat2, lon1, lon2):
    """
    Return distance in KM given a pair of lat-lon coordinates.

    Parameters
    ----------
    lat1: float
        Starting latitude
    lat2: float
        Ending latitude
    lon1: float
        Starting Longitude
    lon2: float
        Ending longitude

    Returns
    -------
    Arc length of great circle
    """
    # https://www.movable-type.co.uk/scripts/latlong.html
    R = 6371
    latDel = lat1 - lat2
    lonDel = lon1 - lon2
    a = (
            np.sin(latDel / 2 * np.pi / 180) ** 2 +
            np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) *
            np.sin(lonDel / 2 * np.pi / 180) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def coord_from_distance(lat, lon, distance):
    """
    Find the bounding box for the stamp based on center coord and distance

    Parameters
    ----------
    lat: float
        Latitude of stamp center
    lon: float
        Longitude of stamp center
    distance: float
        Stamp radius

    Returns
    -------
    Dictionary of bounding latitudes and longitudes for stamp
    """
    # https://www.movable-type.co.uk/scripts/latlong.html
    R = 6371
    latEnd = [float()] * 360
    lonEnd = [float()] * 360
    bearings = range(0, 360)
    distance = float(distance)
    ii = 0
    for bearing in bearings:
        latEnd[ii] = np.arcsin(np.sin(lat * np.pi / 180) * np.cos(distance / R) +
                               np.cos(lat * np.pi / 180) * np.sin(distance / R) *
                               np.cos(bearing * np.pi / 180)
                               ) * 180 / np.pi
        lonEnd[ii] = lon + np.arctan2(
            np.sin(bearing * np.pi / 180) * np.sin(distance / R) * np.cos(lat * np.pi / 180),
            np.cos(distance / R) - np.sin(lat * np.pi / 180) * np.sin(latEnd[ii] * np.pi / 180)
        ) * 180 / np.pi
        ii += 1
    return ({
        'latHi': np.max(latEnd),
        'latLo': np.min(latEnd),
        'lonHi': np.max(lonEnd),
        'lonLo': np.min(lonEnd)
    })

@lru_cache
def stamp_area(lat, distance=400, resolution=0.04):
    """
    Find the number of pixels in a stamp at a given latitude (memoized)

    Parameters
    ----------
    lat: float
        Latitude of stamp center
    distance: float
        Stamp radius

    Returns
    -------
    integer number of pixels in image
    """
    bbox = coord_from_distance(lat, 0, distance)
    # Build the grid
    latpix = int(np.ceil((bbox['latHi'] - bbox['latLo']) / resolution))
    if latpix % 2 == 0:
        latpix += 1
    lonpix = int(np.ceil((bbox['lonHi'] - bbox['lonLo']) / resolution))
    if lonpix % 2 == 0:
        lonpix += 1
    lats = [resolution*(ii - np.floor(latpix/2)) for ii in range(latpix)]
    lons = [resolution*(ii - np.floor(lonpix/2)) for ii in range(lonpix)]
    grid = np.array(np.meshgrid(lats, lons)).reshape(2, latpix*lonpix).T
    # Compute radii over grid
    rads = distance_from_coord(lat * np.ones((grid.shape[0])),
                               lat * np.ones((grid.shape[0])) + grid[:, 0],
                               np.zeros((grid.shape[0])),
                               grid[:, 1])
    val = np.sum(rads < distance)
    return val


def size_normalization(df_size_rad_tc, size_col):
    """
    Normalize the size function relative to the actual area of the image used

    Parameters
    ----------
    df_size_rad_tc : pd.DataFrame
        data frame with size function information and a column 'LAT' that
        defines the size of the image area
    size_col : list of strings
        list of strings of column names associated with the size function

    Returns
    -------
    an updated df_size_rad_tc with normalized size function

    Details
    -------
    see also oc.stamp_area

    """
    df_size_rad_tc['area'] = df_size_rad_tc['LAT'].apply(stamp_area)

    for size_col in size_cols:
        df_size_rad_tc[size_col] = df_size_rad_tc[size_col] /\
                                        df_size_rad_tc['area']
    return df_size_rad_tc


#---------------------------
# Data Merging
#---------------------------

def data_merge(df_size, df_rad, df_tc):
    """
    Merge size, rad and tc dataframe into 1 dataframe and collect unique TC ids

    Arguments
    ---------
    df_size: pd.DataFrame
        data frame with size function information for each TC at given times
    df_rad: pd.DataFrame
        data frame with rad function information for each TC at given times
    df_tc: pd.DataFrame
        data frame with location function information for each TC at given
        times

    Returns
    -------
    df_size_rad_tc: pd.DataFrame
        combination of df_size, df_rad, df_tc
    unique_storm_ids: list
        unique IDs for TCs in the df_size_rad_tc data frame

    Details
    -------
    Expected to be done before other transforms of the data (but not really
    needed...)
    """
    df_size_rad = df_size.merge(df_rad, how='inner',
                                on=['timestamp', 'ID'],
                                suffixes=('_size', '_rad'))
    df_size_rad_tc = df_size_rad.merge(df_tc, how='inner',
                                       on=['timestamp', 'ID'],
                                       suffixes=('', '_tc'))

    unique_storm_ids = list(df_size_rad_tc['ID'].unique()) # training data?

    return df_size_rad_tc, unique_storm_ids


#---------------------------
# PCA feature creation
#---------------------------

def update_pca_size(pca_size, size_cols_means, size_cols, df_size_rad_tc):
    """
    Update size dataframe with first 3 PCA compression vectors

    Parameters
    ----------
    pca_size : sklearn PCA model
        fit on training data
    size_cols_means : numpy vector
        means of columns of training data (pre PCA compression)
    size_cols : list
        names of columns in df_size_rad_tc that are associated with the size function
    df_size_rad_tc : pd.DataFrame
        data frame with size function for test data

    Returns
    -------
    updated df_size_rad_tc function with 3 new columns; 'size_pca{1,2,3}'

    """
    size_pca_trans = pca_size.transform(df_size_rad_tc.loc[:, size_cols] - size_cols_means)
    df_size_rad_tc['size_pca1'] = size_pca_trans[:, 0]
    df_size_rad_tc['size_pca2'] = size_pca_trans[:, 1]
    df_size_rad_tc['size_pca3'] = size_pca_trans[:, 2]

    return df_size_rad_tc


def update_pca_rad(pca_rad, rad_cols_means, rad_cols, df_size_rad_tc):
    """
    Update rad dataframe with first 2 PCA compression vectors

    Parameters
    ----------
    pca_rad : sklearn PCA model
        fit on training data
    rad_cols_means : numpy vector
        means of columns of training data (pre PCA compression)
    rad_cols : list
        names of columns in df_size_rad_tc that are associated with the rad function
    df_size_rad_tc : pd.DataFrame
        data frame with rad function for test data

    Returns
    -------
    updated df_size_rad_tc function with 2 new columns; 'rad_pca{1,2}'

    Details
    -------
    If any row has any NAs in the rad rows it is not compressed (naturally)
    """

    no_null_rows = np.array(df_size_rad_tc.loc[:, rad_cols].isna().sum(1) == 0)

    rad_pca_trans = pca_rad.transform(df_size_rad_tc.loc[no_null_rows, rad_cols] - rad_cols_means)

    df_size_rad_tc['rad_pca1'] = np.nan
    df_size_rad_tc['rad_pca2'] = np.nan

    df_size_rad_tc.loc[no_null_rows, 'rad_pca1'] = rad_pca_trans[:, 0]
    df_size_rad_tc.loc[no_null_rows, 'rad_pca2'] = rad_pca_trans[:, 1]

    return df_size_rad_tc


#---------------------------
# Data Loading & processing
#---------------------------


def get_orb_files(file_dir, location_prefix, orb_suffix, min_year=2000, max_year=2005):
    """
    Load a certain type of ORB functions for tropical cyclones from a window of years

    Parameters
    ----------
    file_dir : string
        string of file directory where the files exist (e.g. )
    location_prefix : string
        location that the tropical cyclone is from (e.g. AL = Alantic Ocean)
    orb-suffix : string
       suffix at the end of the file that captures the type of data information
       (e.g. RAD = "_rad.csv", SIZE = "-size.csv", PATH = "_TCdata.csv")
    min_year : int
        minimum year to grab tropical cylones from (naturally inclusive)
    max_year : int
        maximum year to grab tropical cylones from (naturally inclusive)

    Returns
    -------
    list of file names

    Details
    -------
    File names look something like "AL082005_rad.csv" (for location_prefix = "AL", orb-suffix = "_rad.csv",
    min_year <= 2005, max_year >=2005)
    """
    years = [str(i) for i in range(min_year, max_year + 1)]
    files = os.listdir(file_dir)
    rad_files = []
    for f in files:
        is_orb = orb_suffix in f
        year_valid = False
        location_valid = f.startswith(location_prefix)
        for year in years:
            year_valid = year_valid or (year in f)
        if is_orb and year_valid and location_valid:
            rad_files.append(f)
    return rad_files

def collect_size_df(size_directory, size_files):
    """
    create dataframe for size functions of all tropical cyclones in selection

    Parameters
    ----------
    size_directory : string
        location of directory for size files
    size_files : list of strings
        list of csv file names for the size information

    Returns
    -------
    df_size : pd.DataFrame
        DataFrame of size functions concatenated across all tropical cylones
    size_cols : list
        list of strings of columns of df_size that are associated with the actual function
    """
    dfs = []
    for f in size_files:
        path = f'{size_directory}/{f}'
        storm_id = f.split('-')[0]
        df = pd.read_csv(path, header=0, skiprows=[0, 2])
        del df['Unnamed: 1']
        del df['time']
        df = df.transpose()
        df.reset_index(inplace=True)
        df.rename(columns=dict([(i, str(int(i) - 100)) for i in range(100)]), inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)
        df['ID'] = storm_id
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('min')
        dfs.append(df)

    df_size = pd.concat(dfs)

    size_cols = [str(i) for i in range(-1, -100 - 1, -1)]

    return df_size, size_cols

def collect_rad_df(rad_directory, rad_files):
    """
    create dataframe for rad functions of all tropical cyclones in selection

    Parameters
    ----------
    rad_directory : string
        location of directory for rad files
    rad_files : list of strings
        list of csv file names for the rad information

    Returns
    -------
    df_rad : pd.DataFrame
        DataFrame of size functions concatenated across all tropical cylones
    rad_cols : list
        list of strings of columns of df_rad that are associated with the actual function
    """
    dfs = []
    for f in rad_files:
        path = f'{rad_directory}/{f}'
        storm_id = f.split('_')[0]
        df = pd.read_csv(path, header=0, skiprows=[0, 2])
        df.rename(columns={'radius': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('min')
        df['ID'] = storm_id
        df.rename(columns=dict([(str(float(i)), str(i)) for i in range(5, 600 + 5, 5)]), inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        dfs.append(df)

    df_rad = pd.concat(dfs)
    rad_cols = [str(i) for i in range(5, 600 + 5, 5)]

    return df_rad, rad_cols

def collect_tc_df(tc_directory, tc_files):
    """
    create dataframe for lon/lat paths of all tropical cyclones in selection

    Parameters
    ----------
    tc_directory : string
        location of directory for tc (lon/lat path) files
    rad_files : list of strings
        list of csv file names for the tc (lon/lat path) information

    Returns
    -------
    df_tc : pd.DataFrame
        DataFrame of lat/lon paths concatenated across all tropical cylones
    tc_cols : list
        list of strings of columns of df_tc that are associated with the paths (LAT only)
    """

    dfs = []
    for f in tc_files:
        path = f'{tc_directory}/{f}'
        storm_id = f.split('_')[0]
        df = pd.read_csv(path, header=0)
        df.rename(columns={'TIMESTAMP': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('min')
        #df = df.resample('0.5H', on='timestamp').mean().reset_index().interpolate()
        ## ^ interpolation wasn't working due to timestamp columns...
        df = df.resample('0.5H', on='timestamp').mean().reset_index()
        df.loc[:,df.columns != "timestamp"] = df.loc[:,df.columns != "timestamp"].interpolate()
        df['LAT'] = df['LAT'].round(1)
        df['ID'] = storm_id
        dfs.append(df[['ID', 'timestamp', 'LAT', 'WIND']])

    df_tc = pd.concat(dfs)
    tc_cols = ["LAT"]
    return df_tc, tc_cols

