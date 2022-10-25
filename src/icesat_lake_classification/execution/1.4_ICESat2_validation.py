import os

import numpy as np
import pandas as pd

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth


if __name__ == "__main__":

    s2_date = '20190617_L1C'
    s2_band_list = ['NDWI_10m', 'B03', "B04", "B08", "B11", "B12"]
    s2_data_dir = "F:/onderzoeken/thesis_msc/data/Sentinel/{}".format(s2_date)

    base_dir = 'F:/onderzoeken/thesis_msc/'
    overwrite_empirical = False

    # Parameters
    plot_starter_index = 150
    ph_per_image = 50000
    NDWI_threshold = 0.21


    figures_dir = os.path.join(base_dir, 'figures', s2_date)
    data_dir_in = os.path.join(base_dir, 'data')
    data_dir = os.path.join(base_dir, 'data', s2_date)

    # Process
    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    cluster_fn_in_list = pth.get_files_from_folder(os.path.join(data_dir_in, 'cluster'), '*1222*gt*l*.csv')

    depth_data_green_list, depth_data_red_list = [], []
    green_reflectance_list, red_reflectance_list = [], []
    # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, 5 = real bottom
    for fn in cluster_fn_in_list:
        utl.log("Loading classification track/beam: {}".format(os.path.basename(fn)[:-4]), log_level='INFO')
        with utl.codeTimer('loading data'):

            classification_df = pd.read_hdf(os.path.join(data_dir, 'classification', pth.get_filname_without_extension(fn) +'.h5' ))
            s2_df = pd.read_hdf(os.path.join(data_dir, 'Training', pth.get_filname_without_extension(fn) + '.h5'))

            classification_df[s2_band_list] = s2_df[s2_band_list].copy()
            del s2_df

            classification_df['NDWI_class'] = 0  # nodata value
            classification_df['NDWI_class'] = np.where((classification_df['NDWI_10m'] > NDWI_threshold), 1, 2)  # 1 for lakes, 2 for no lake

            classification_df['valid_point_index'] = np.where((classification_df['NDWI_10m'] > 0.2)& (classification_df['lake_rolling'] == 1) & (
                                                                          classification_df['SurfBottR'] > 1) & (classification_df['SurfNoiseR'] > 2.5) & (
                                                                          classification_df['SurfBottR'] < 10) & (classification_df['dem_diff'] < 400) & (
                                                                          classification_df['range'] < 400) & (classification_df['slope_mean'] < 0.1), 1, 0)

        if not pth.check_existence(os.path.join(data_dir, 'empirical', pth.get_filname_without_extension(fn)[0:44] + '_green.csv'), overwrite_empirical):
            #make the empirical relation
            utl.log('Retrieving data for the empirical relations', log_level='INFO')
            with utl.codeTimer('empirical'):
                empirical_index_green = np.where((classification_df['B03'] > 0) & (classification_df['valid_point_index'] > 0))
                green_depth = -1 * classification_df['lake_diff'].iloc[empirical_index_green].values
                depth_index_green = np.where((~np.isnan(green_depth)) & (green_depth < 25) & (green_depth > 0))

                empirical_index_red = np.where((classification_df['B04'] > 0) & (classification_df['valid_point_index'] > 0))
                red_depth = -1 * classification_df['lake_diff'].iloc[empirical_index_red].values
                depth_index_red = np.where((~np.isnan(red_depth)) & (red_depth < 25) & (red_depth > 0))

                depth_data_green_list += list(green_depth[depth_index_green])
                depth_data_red_list += list(red_depth[depth_index_red])

                green_reflectance_list += list(classification_df['B04'].iloc[empirical_index_green].values[depth_index_green])
                red_reflectance_list += list(classification_df['B03'].iloc[empirical_index_red].values[depth_index_red])

        utl.log('Saving data for the empirical relation', log_level='INFO')
        pd.DataFrame({'depth':depth_data_green_list, 'reflectance': green_reflectance_list}).to_csv(os.path.join(data_dir, 'empirical', pth.get_filname_without_extension(fn)[0:44] + '_green.csv'))
        pd.DataFrame({'depth': depth_data_red_list, 'reflectance': red_reflectance_list}).to_csv(os.path.join(data_dir, 'empirical', pth.get_filname_without_extension(fn)[0:44] + '_red.csv'))
