# importlib is included in the standard library for Python >= 3.8
# here __version__ is obtained from installed version of package
from importlib.metadata import version
import pandas as _pd
import os as _os

__version__ = version(__name__)

from .utils import check_character_percent
from .data_cleaning import linear_interp, linear_interp_df, \
    remove_init_and_final_missing_rows_for_interp
from .data_processing import randomize_range, \
    update_from_previous_starting_point, define_sample_cuts, \
    df_subset

from .vis_tools import vis_surfaces, vis_sequence_surface, \
    vis_slice_x, vis_slice_y


from .distances import l2_dist, l2_dist_lots2one, l2_dist_matrix, \
    l2_dist_lots2one_pointwise
from .meanshift import meanshift_multidim_funct_single, \
    meanshift_multidim_funct, mode_clustering, mode_clustering_check
from .psuedo_density import psuedo_density_multidim_func
from .geometric_structure import calc_small_ball_rad_multidim_func



# feature creation
from .orb_group_functions import distance_from_coord, coord_from_distance, \
    stamp_area, size_normalization
# data merging
from .orb_group_functions import data_merge
# pca feature creation
from .orb_group_functions import update_pca_size, update_pca_rad
# data loading / processing
from .orb_group_functions import get_orb_files, collect_size_df, collect_rad_df, collect_tc_df



from .simulate import simulate



# loading in data --------
_location = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.realpath(__file__))))
_my_file = _os.path.join(_location, 'data/', 'AL122005_rad.csv')

tc_rad = _pd.read_csv(_my_file, skiprows = 1)
