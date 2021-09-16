import numpy as np
from functools import lru_cache

#
# these functions came from code written by Pavel Khnokhlov and Trey McNeely
#
# they have been rewritten for some cleaner comments


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

