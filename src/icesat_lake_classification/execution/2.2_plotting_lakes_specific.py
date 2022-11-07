import os
import math

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import icesat_lake_classification.validation as validation

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.ICESat2_visualization import get_confusion_matrix, get_confusion_matrix2


if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    empirical = True
    NDWI_threshold = 0.21

    s2_band_list = ['NDWI_10m', 'B03', "B04", "B08", "B11", "B12"]
    s2_date = '20190617_L1C'
    base_dir = 'F:/onderzoeken/thesis_msc/'
    figures_dir = os.path.join(base_dir, 'figures', s2_date)
    data_dir = os.path.join(base_dir, 'data', s2_date)

    if not pth.check_existence(os.path.join(figures_dir, 'final')):
        os.mkdir(os.path.join(figures_dir, 'final'))

    classification_df_fn_list = pth.get_files_from_folder(os.path.join(data_dir, 'classification'), '*1222*gt1l*.h5')

    if empirical:
        utl.log('Plotting the Empirical relations', log_level='INFO')
        empirical_df = pd.read_csv(pth.get_files_from_folder(os.path.join(data_dir, 'empirical'),'*1222*.csv')[0])

        parameters_green, parameters_red, parameters_green_physical, parameters_red_physical, model = validation.estimate_relations(empirical_df, ['NDWI_10m', 'B03', "B04", "B08", "B11", "B12"])

    slices_list_122 = [[slice(11460000 - 10000, 11500000), slice(13425000, 13460000), slice(13225000 + 15000,13225000+26000),
                        slice(11100000 +15000, 11100000 + 35000), slice(11025000 +10000, 11025000 + 35000), slice(15325000-5000, 15325000 +15000),
                        slice(14150000-5000, 14150000+27000), slice(14100000 -25000, 14150000), slice(13525000 + 5000, 13525000+40000), slice(12025000, 12025000 +20000),
                        slice(14000000+10000, 14000000+25000)] ,

                       [slice(8800000 -5000,8825000 -15000), slice(10550000, 10550000+26000), slice(9900000 -15000, 9900000 + 10000),
                        slice(9200000+5000, 9200000 + 20000), slice(8500000, 8500000+ 35000), slice(11125000 -15000, 11125000+20000)],

                       [slice(13465000-15000, 13485000), slice(11150000-10000, 11150000 +15000), slice(13450000 +10000, 13450000+30000), slice(12500000, 12500000+20000),
                       slice(13025000 -10000, 13025000 +15000)],

                       [slice(1800000-5000, 1800000 + 8000), slice(1825000+8000, 1825000+6000+30000-10000), slice(1775000+18000, 1775000+40000)]]


    for i_fn, fn in enumerate(classification_df_fn_list):

        slices = slices_list_122[0]
        if not pth.check_existence(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn))):
            os.mkdir(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn)))

        utl.log(fn, log_level='INFO')
        classification_df = pd.read_hdf(fn)
        n_ph = len(classification_df)
        s2_df = pd.read_hdf(os.path.join(data_dir, 'Training', pth.get_filname_without_extension(fn) + '.h5'))

        classification_df[s2_band_list] = s2_df[s2_band_list].copy()
        del s2_df

        classification_df['NDWI_class'] = 0  # nodata value
        classification_df['NDWI_class'] = np.where((classification_df['NDWI_10m'] > NDWI_threshold), 1,2)  # 1 for lakes, 2 for no lake

        if empirical:
            utl.log('Calculating empircal/physical depth lines for the plot', log_level='INFO')

            classification_df = validation.calculate_depth_from_relations(classification_df,parameters_green, parameters_red,
                                                                          parameters_green_physical, parameters_red_physical, model, ['NDWI_10m','B03', "B04", "B08", "B11", "B12"])

        if not pth.check_existence(os.path.join(figures_dir, 'final')):
            os.mkdir(os.path.join(figures_dir, 'final'))

        if not pth.check_existence(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn))):
            os.mkdir(os.path.join(figures_dir, 'final', pth.get_filname_without_extension(fn)))

        utl.log('making confusion_matrix', log_level='INFO')
        # make confusion matrix
        # CM_tuple = get_confusion_matrix(classification_df)
        # CM_tuple2 = get_confusion_matrix2(classification_df)

        n_ph = len(classification_df)
        for i, index_slice in enumerate(slices):

            utl.log('Making color map', log_level='INFO')
            # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, bottom sure = 5
            cluster_map = {0: 'Tan', 1: 'darkgrey', 2: 'darkgrey', 3: 'cornflowerblue', 4: 'skyblue',
                           5: 'cornflowerblue'}
            c_cluster = [cluster_map[x] if not math.isnan(x) else cluster_map[0] for x in
                                              classification_df['clusters'].iloc[index_slice]]

            # clusters - 1 Lake, 0 no lake, 2, deep enough but to steep, 3 not deep enough but flat
            result_map_bottom = {1: 'Indigo', 0: 'dimgrey', 2: 'darkviolet', 3: 'violet', np.nan: 'dimgrey', 10: 'red'}
            c_bottom = [result_map_bottom[x] if not math.isnan(x) else result_map_bottom[10] for
                                             x in classification_df['lake_rolling'].iloc[index_slice]]

            # clusters - 1 lake in NDWI, 0 is nodata, 2 no lake
            result_map_surface = {1: 'Indigo', 0: 'dimgrey', 2: 'lightgrey', np.nan: 'dimgrey'}
            c_surface = [result_map_surface[x] if not math.isnan(x) else result_map_surface[0]
                                              for x in classification_df['NDWI_class'].iloc[index_slice]]


            utl.log("Conditions apply - Plotting this graph- {}".format(
                classification_df['NDWI_class'].iloc[index_slice].unique()), log_level='INFO')

            f1, ax1 = plt.subplots(figsize=(10, 10))
            ax1.scatter(classification_df['distance'].iloc[index_slice] - np.min(classification_df['distance'].iloc[index_slice].values),
                        classification_df['height'].iloc[index_slice],
                        c=c_cluster, marker=',', s=0.3)

            # plot bottom_line
            points = np.array([classification_df['bottom_distance'].iloc[index_slice] - np.min(classification_df['bottom_distance'].iloc[index_slice].values),
                               classification_df['bottom_height'].iloc[index_slice]]).T.reshape(-1, 1, 2)
            lines = np.concatenate([points[:-1], points[1:]], axis=1)
            colored_lines = LineCollection(lines, colors=c_bottom,
                                           linewidths=(2,))
            ax1.add_collection(colored_lines)

            # plot surface line
            points_surface = np.array([classification_df['surface_distance'].iloc[index_slice] - np.min(classification_df['surface_distance'].iloc[index_slice].values),
                                       classification_df['surface_height'].iloc[index_slice]]).T.reshape(-1, 1, 2)
            lines_surface = np.concatenate([points_surface[:-1], points_surface[1:]], axis=1)
            colored_lines_surface = LineCollection(lines_surface,
                                                   colors=c_surface,
                                                   linewidths=(2,))
            ax1.add_collection(colored_lines_surface)

            if empirical:
                # ax1.plot(classification_df['surface_distance'].iloc[index_slice] - np.min(classification_df['surface_distance'].iloc[index_slice].values),
                #          classification_df['green'].iloc[index_slice], 'darkgreen', label='Empirical - B03')
                # ax1.plot(classification_df['surface_distance'].iloc[index_slice] - np.min(classification_df['surface_distance'].iloc[index_slice].values),
                #          classification_df['green_phys'].iloc[index_slice], 'mediumseagreen', label='Physical - B03')
                #
                # ax1.plot(classification_df['surface_distance'].iloc[index_slice] - np.min(classification_df['surface_distance'].iloc[index_slice].values),
                #          classification_df['red'].iloc[index_slice], 'maroon' , label='Empirical - B04')
                # ax1.plot(classification_df['surface_distance'].iloc[index_slice] - np.min(classification_df['surface_distance'].iloc[index_slice].values),
                #          classification_df['red_phys'].iloc[index_slice], 'orangered' , label='Physical - B04')

                # #machine learning
                ax1.plot(classification_df['surface_distance'].iloc[index_slice] - np.min(classification_df['surface_distance'].iloc[index_slice].values),
                         -classification_df['ML'].iloc[index_slice] + classification_df['surface_height'].iloc[index_slice], 'mediumvioletred', label='Machine Learning')


            ax1.set_ylim(np.median(classification_df['height'].iloc[index_slice].values) - 30, np.median(classification_df['height'].iloc[index_slice].values) + 30)
            ax1.set_xlim(np.min(classification_df['distance'].iloc[index_slice].values - np.min(classification_df['distance'].iloc[index_slice].values)),
                     np.max(classification_df['distance'].iloc[index_slice].values - np.min(classification_df['distance'].iloc[index_slice].values)))

            ax1.set_title('Longitude: {} -- Latitude: {}'.format(np.round(classification_df['lon'].iloc[index_slice].iloc[0], 2),
                                                    np.round(classification_df['lat'].iloc[index_slice].iloc[0], 2)))
            ax1.get_xaxis().set_tick_params(which='both', direction='in', labelsize=16)
            ax1.get_yaxis().set_tick_params(which='both', direction='in', labelsize=16)
            ax1.set_xlabel('Along-track distance [m]', fontsize=22)
            ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]', fontsize=22)
            # ax1.legend(loc='lower right')

            outpath = os.path.join(os.path.join(figures_dir, 'final',
                                                'lake_classification_s2_lon_{}_lat_{}_ph_distance_median_{}___{}___{}_ML_3.png'.format(
                                                    np.round(classification_df['lon'].iloc[index_slice].iloc[0], 2),
                                                    np.round(classification_df['lat'].iloc[index_slice].iloc[0], 2),
                                                    index_slice.start, index_slice.stop, i)))
            plt.savefig(outpath)
            plt.close('all')





