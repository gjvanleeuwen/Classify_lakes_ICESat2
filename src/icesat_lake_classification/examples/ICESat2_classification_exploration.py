import os
import math

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.classification_optimization import find_optimal_eps
from icesat_lake_classification.ICESat2_visualization import plot_classified_photons

if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    plt.ioff()
    pd.options.mode.chained_assignment = None  # default='warn'

    base_dir = 'F:/onderzoeken/thesis_msc/'
    figures_dir = os.path.join(base_dir, 'Exploration/figures')
    lake_dir = os.path.join(figures_dir, 'lake')

    example_lakes = {'ID': [0, 1, 2, 3, 4, 5, 6],
                     'beam': ['gt2l', 'gt2l', 'gt2l', 'gt1r', 'gt1r', 'gt1l', 'gt3l'],
                     'lon': [(-49.05 ,-49.02), (-48.87, -48.84), (-48.38, -48.35), (-48.53, -48.50), (-48.45, -48.42), (-49.01, -48.98), (-48.79, -48.76)],
                     'lat': [(68.95, 69.04), (68.41, 68.50), (66.81, 66.90), (67.07, 67.17), (66.81, 66.90), (68.59, 68.68), (68.42, 68.51)]}


    complete_track = True
    iteration_starter = 10
    min_pts = 6
    eps_method = 'max'
    classification_method = 'DBSCAN'
    cluster_method = 'BOOL'
    classification_dir = 'F:/onderzoeken/thesis_msc/Exploration/data'
    classification_df_fn_list = pth.get_files_from_folder(classification_dir, '*1222*gt2l*')

    if complete_track:
        for fn in classification_df_fn_list:
            utl.log(fn, log_level='INFO')
            classification_df = pd.read_csv(fn)
            n_ph = len(classification_df)
            ph_per_image = 50000

            start_index_array = np.arange(0, n_ph, ph_per_image)
            end_index_array = np.append(np.arange(ph_per_image, n_ph, ph_per_image), n_ph - 1)
            class_list = []

            for i, (ph_start, ph_end) in enumerate(zip(start_index_array, end_index_array)):

                if ph_start < 10950000:
                    continue

                index_slice = slice(ph_start,ph_end)
                utl.log(index_slice, log_level='INFO')

                # create dataframe with just 2 variables
                data_df = classification_df[['distance', 'height']].iloc[index_slice]
                data_df['distance'] = data_df['distance'] - min(data_df['distance'])

                if not pth.check_existence(
                        os.path.join(base_dir, 'Exploration/figures/', os.path.basename(fn)[:-4])):
                    os.mkdir(os.path.join(base_dir, 'Exploration/figures/', os.path.basename(fn)[:-4]))

                eps1 = find_optimal_eps(data_df, min_pts=min_pts, method=eps_method,
                                        outpath=os.path.join(figures_dir,
                                                             os.path.basename(fn)[:-4],
                                                             'histogram_EPS_ph_{}_method_{}_minpts_{}.png'.format(
                                                                 ph_start, eps_method, min_pts)), strict=1.5)

                clustering = DBSCAN(eps=eps1, min_samples=min_pts).fit(data_df)
                clusters = np.where(np.array(clustering.labels_) >= 0, 1, 0)

                outpath = os.path.join(figures_dir, os.path.basename(fn)[:-4],
                                       'photon_classification_ph_{}_{}_{}_{}.png'.format(ph_start,
                                                                                         classification_method,
                                                                                         eps_method, min_pts))
                plot_classified_photons(data_df, clusters, ph_start, eps1, outpath)

                ### step 2 --> eliminate surface
                data_df2 = data_df.loc[clusters > 0]
                noise_df = data_df.loc[clusters == 0]

                min_pts_step2 = 3
                buffer_factor_step2_line1 = 1
                strict_step2 = 6
                window_surface_line1 = 100
                window_surface_line2 = 400
                min_periods_line2 = 50

                eps2 = find_optimal_eps(data_df2, min_pts=min_pts_step2, method=eps_method,
                                        outpath=os.path.join(figures_dir, os.path.basename(fn)[:-4],
                                                             'histogram_EPS_step2_ph_{}_method_{}_minpts_{}.png'.format(
                                                                 ph_start, eps_method, min_pts_step2,
                                                                 strict=strict_step2)))

                clustering = DBSCAN(eps=eps2, min_samples=min_pts_step2).fit(data_df2)
                data_df2['clusters'] = np.where(np.array(clustering.labels_) >= 0, 1, 0)

                # make rolling average of surface and calculate difference with normal data
                data_df2_window = data_df2.loc[data_df2['clusters'] > 0].rolling(
                    window=window_surface_line1).median()  # .iloc[N - 1:].values
                data_df2_window['diff'] = np.abs(
                    data_df2_window['height'] - data_df2['height'])  # .loc[data_df2['clusters'] > 0])
                data_df2_window['surface_data_buffer_height'] = data_df2_window['height'].loc[
                    data_df2_window['diff'] < (buffer_factor_step2_line1 * eps2)]

                # Create a 2nd line even more smooth
                data_df2_window['surface_data_buffer_height_window'] = data_df2_window[
                    'surface_data_buffer_height'].copy().rolling(window=window_surface_line2,
                                                                 min_periods=min_periods_line2).mean()
                data_df2_window['diff2'] = np.abs(
                    data_df2_window['surface_data_buffer_height_window'] - data_df2['height'])

                # add detailed clusters to original dataframe for plot
                data_df2['clusters2'] = data_df2['clusters'].copy()
                data_df2['diff'] = data_df2_window['diff'].copy()
                data_df2['clusters2'].loc[data_df2['diff'] < (buffer_factor_step2_line1 * eps2)] = 2

                # plot a scatter with both surface lines and the photon classification
                f1, ax1 = plt.subplots(figsize=(20, 20))
                ax1.scatter(data_df2['distance'], data_df2['height'], c=data_df2['clusters2'], cmap='Set2', marker=',',
                            s=0.5)

                ax1.plot(data_df2_window.loc[:, 'distance'], data_df2_window.loc[:, 'height'])
                ax1.plot(data_df2_window.loc[:, 'distance'],
                         data_df2_window.loc[:, 'surface_data_buffer_height_window'])

                ax1.set_title('classification gt1l' + "- for photons {} and EPS {}".format(ph_start, eps2))
                ax1.get_xaxis().set_tick_params(which='both', direction='in')
                ax1.get_yaxis().set_tick_params(which='both', direction='in')
                ax1.set_xlabel('distance')
                ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')
                outpath = os.path.join(figures_dir, os.path.basename(fn)[:-4],
                                       'photon_classification_step2_ph_{}_{}_{}_{}.png'.format(ph_start,
                                                                                               classification_method,
                                                                                               eps_method,
                                                                                               min_pts_step2))
                plt.savefig(outpath)

                ### step 3 - Extracting the bottom
                min_pts_step3 = 6
                buffer_factor_step3 = 1
                window_bottom_line = 50
                min_periods_bottom_line = 15
                method3 = 'max'
                buffer_bottom_line = - 0.25

                df_interp = utl.interpolate_df_to_new_index(data_df2,
                                                            data_df2_window.loc[:, ['distance', 'height']].copy(),
                                                            'distance')
                # df_interp.rename(columns={'surface_data_buffer_height_window': "height"},inplace=True)
                df_interp['diff'] = data_df2['height'] - df_interp['height']
                bottom_df = data_df2.loc[(data_df2['clusters2'] == 0) & (df_interp['diff'] < buffer_bottom_line)]
                # bottom_df = data_df2.loc[(data_df2['clusters2'] > 0)]

                # eps3 = find_optimal_eps(bottom_df[['distance', 'height']], min_pts=min_pts_step3, method=method3,
                #                         outpath=os.path.join(figures_dir,
                #                                              os.path.basename(fn)[:-4],
                #                                              'histogram_EPS_step3_ph_{}_method_{}_minpts_{}.png'.format(
                #                                                  ph_start, method3, min_pts_step3)), strict=1.5)
                #
                # clustering = DBSCAN(eps=eps3, min_samples=min_pts_step3).fit(bottom_df[['distance', 'height']])
                bottom_df['clusters'] = np.ones(len(bottom_df)) #np.where(np.array(clustering.labels_) >= 0, 1, 0)
                bottom_df_window = bottom_df.loc[bottom_df['clusters'] > 0].rolling(window=window_bottom_line,
                                                                                    min_periods=min_periods_bottom_line).mean()

                # # plot a scatter with both surface lines and the photon classification
                # f1, ax1 = plt.subplots(figsize=(20, 20))
                # ax1.scatter(bottom_df['distance'], bottom_df['height'], c=bottom_df['clusters'], cmap='Set2',
                #             marker=',',
                #             s=0.5)
                #
                # ax1.plot(bottom_df_window.loc[:, 'distance'], bottom_df_window.loc[:, 'height'])
                #
                # ax1.set_title('classification gt1l' + "- for photons {} and EPS {}".format(ph_start, eps3))
                # ax1.get_xaxis().set_tick_params(which='both', direction='in')
                # ax1.get_yaxis().set_tick_params(which='both', direction='in')
                # ax1.set_xlabel('distance')
                # ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')
                # outpath = os.path.join(figures_dir, os.path.basename(fn)[:-4],
                #                        'photon_classification_step3_ph_{}_{}_{}_{}.png'.format(ph_start,
                #                                                                                classification_method,
                #                                                                                eps_method,
                #                                                                                min_pts_step2))
                # plt.savefig(outpath)

                ## save clustering of all individual photons
                data_df['clusters'] = clusters.copy()  # noise = 0 , data = 1 #surface = 2, bottom =3, bottom_noise=5
                clusters2_fill = np.where(np.array(data_df2['clusters']) > 0, 2,
                                          0)  # everywhere not noise fill with labels cluster 2
                data_df['clusters'][clusters > 0] = clusters2_fill
                clusters3_fill = np.where(np.array(bottom_df['clusters']) > 0, 3, 5)
                clusters_non_noise = data_df['clusters'][
                    clusters > 0].copy()  # get copy of clusters without noise photons
                clusters_non_noise[(data_df2['clusters2'] == 0) & (df_interp[
                                                                       'diff'] < buffer_bottom_line)] = clusters3_fill  # fill everywhere bottom photons are with result of cluster 3
                # # & (df_interp['diff'] < (-buffer_factor_step3 * eps2))]
                data_df['clusters'][clusters > 0] = clusters_non_noise

                plt.ioff()
                f1, ax1 = plt.subplots(figsize=(20, 20))
                ax1.scatter(data_df['distance'], data_df['height'], c=data_df['clusters'], cmap='Set2', marker=',',
                            s=0.5)

                ax1.plot(data_df2_window.loc[:, 'distance'], data_df2_window.loc[:, 'height'])
                ax1.plot(bottom_df_window.loc[:, 'distance'], bottom_df_window.loc[:, 'height'])

                ax1.set_title('classification gt1l' + "- for photons {}".format(ph_start))
                ax1.get_xaxis().set_tick_params(which='both', direction='in')
                ax1.get_yaxis().set_tick_params(which='both', direction='in')
                ax1.set_xlabel('distance')
                ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')

                outpath = os.path.join(figures_dir, os.path.basename(fn)[:-4],
                                       'FINAL_classification_MEAN_ph_{}.png'.format(ph_start))
                plt.savefig(outpath)

                # class_list.extend(data_df['clusters'].values.astype('int').tolist())

            # classification_df['class'] = class_list
            # classification_df.to_csv(os.path.join(fn[:-4],'_class.csv'))

    elif not complete_track:
        for i, lake_ID in enumerate(example_lakes['ID']):

            classification_df_fn_list = pth.get_files_from_folder(classification_dir, '*1222*{}*'.format(example_lakes['beam'][lake_ID]))

            for fn in classification_df_fn_list:
                utl.log(fn, log_level='INFO')
                classification_df = pd.read_csv(fn)
                print(classification_df.columns)
                longitude_data = classification_df['lon']

                # get lake data index
                min_lon, max_lon = example_lakes['lon'][lake_ID][1], example_lakes['lon'][lake_ID][0]
                min_lon_index, max_lon_index = utl.find_nearest(longitude_data, min_lon), utl.find_nearest(longitude_data, max_lon)
                index_slice = slice(min_lon_index, max_lon_index)

                utl.log(index_slice, log_level='INFO')

                # create dataframe with just 2 variables
                data_df = classification_df[['distance', 'height']].iloc[index_slice]
                data_df['distance'] = data_df['distance'] - min(data_df['distance'])

                if not pth.check_existence(os.path.join(base_dir, 'Exploration/figures/lake', os.path.basename(fn)[:-4])):
                    os.mkdir(os.path.join(base_dir, 'Exploration/figures/lake', os.path.basename(fn)[:-4]))

                eps1 = find_optimal_eps(data_df, min_pts=min_pts, method=eps_method,
                                       outpath=os.path.join(lake_dir,
                                                            os.path.basename(fn)[:-4],
                                                            'histogram_EPS_ph_{}_method_{}_minpts_{}.png'.format(min_lon_index, eps_method, min_pts)), strict=1.5)

                clustering = DBSCAN(eps=eps1, min_samples=min_pts).fit(data_df)
                clusters = np.where(np.array(clustering.labels_) >= 0, 1, 0)

                outpath = os.path.join(lake_dir, os.path.basename(fn)[:-4],
                             'photon_classification_ph_{}_{}_{}_{}.png'.format(min_lon_index, classification_method, eps_method, min_pts))
                plot_classified_photons(data_df, clusters, min_lon_index, eps1, outpath)



                ### step 2 --> eliminate surface
                data_df2 = data_df.loc[clusters > 0]
                noise_df = data_df.loc[clusters == 0]

                min_pts_step2 = 3
                buffer_factor_step2_line1 = 1
                strict_step2 = 6
                window_surface_line1 = 100
                window_surface_line2 = 400
                min_periods_line2 = 50

                eps2 = find_optimal_eps(data_df2, min_pts=min_pts_step2, method=eps_method,
                                           outpath=os.path.join(lake_dir, os.path.basename(fn)[:-4],
                                                                'histogram_EPS_step2_ph_{}_method_{}_minpts_{}.png'.format(
                                                                    min_lon_index, eps_method, min_pts_step2, strict=strict_step2)))

                clustering = DBSCAN(eps=eps2, min_samples=min_pts_step2).fit(data_df2)
                data_df2['clusters'] = np.where(np.array(clustering.labels_) >= 0, 1, 0)

                # make rolling average of surface and calculate difference with normal data
                data_df2_window = data_df2.loc[data_df2['clusters'] > 0].rolling(window=window_surface_line1).median() # .iloc[N - 1:].values
                data_df2_window['diff'] = np.abs(data_df2_window['height'] - data_df2['height']) #.loc[data_df2['clusters'] > 0])
                data_df2_window['surface_data_buffer_height'] = data_df2_window['height'].loc[data_df2_window['diff'] < (buffer_factor_step2_line1*eps2)]

                # Create a 2nd line even more smooth
                data_df2_window['surface_data_buffer_height_window'] = data_df2_window['surface_data_buffer_height'].copy().rolling(window=window_surface_line2, min_periods=min_periods_line2).mean()
                data_df2_window['diff2'] = np.abs(data_df2_window['surface_data_buffer_height_window'] - data_df2['height'])

                # add detailed clusters to original dataframe for plot
                data_df2['clusters2'] = data_df2['clusters'].copy()
                data_df2['diff'] = data_df2_window['diff'].copy()
                data_df2['clusters2'].loc[data_df2['diff'] < (buffer_factor_step2_line1*eps2)] = 2

                # plot a scatter with both surface lines and the photon classification
                f1, ax1 = plt.subplots(figsize=(20, 20))
                ax1.scatter(data_df2['distance'], data_df2['height'], c=data_df2['clusters2'], cmap='Set2', marker=',',
                            s=0.5)

                ax1.plot(data_df2_window.loc[:, 'distance'], data_df2_window.loc[:, 'height'])
                ax1.plot(data_df2_window.loc[:, 'distance'], data_df2_window.loc[:, 'surface_data_buffer_height_window'])

                ax1.set_title('classification gt1l' + "- for photons {} and EPS {}".format(min_lon_index, eps2))
                ax1.get_xaxis().set_tick_params(which='both', direction='in')
                ax1.get_yaxis().set_tick_params(which='both', direction='in')
                ax1.set_xlabel('distance')
                ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')
                outpath = os.path.join(lake_dir, os.path.basename(fn)[:-4], 'photon_classification_step2_ph_{}_{}_{}_{}.png'.format(min_lon_index,
                                                                                               classification_method,eps_method, min_pts_step2))
                plt.savefig(outpath)


                ### step 3 - Extracting the bottom
                min_pts_step3 = 6
                buffer_factor_step3 = 1
                window_bottom_line = 50
                min_periods_bottom_line = 15
                method3 = 'max'
                buffer_bottom_line = - 0.25

                df_interp = utl.interpolate_df_to_new_index(data_df2,data_df2_window.loc[:, ['distance', 'height']].copy(),'distance')
                # df_interp.rename(columns={'surface_data_buffer_height_window': "height"},inplace=True)
                df_interp['diff'] = data_df2['height'] - df_interp['height']
                bottom_df = data_df2.loc[(data_df2['clusters2'] == 0) & (df_interp['diff'] < buffer_bottom_line)]
                # bottom_df = data_df2.loc[(data_df2['clusters2'] > 0)]

                eps3 = find_optimal_eps(bottom_df[['distance', 'height']], min_pts=min_pts_step3, method=method3,
                                       outpath=os.path.join(lake_dir,
                                                            os.path.basename(fn)[:-4],
                                                            'histogram_EPS_step3_ph_{}_method_{}_minpts_{}.png'.format(
                                                                min_lon_index, method3, min_pts_step3)), strict=1.5)

                clustering = DBSCAN(eps=eps3, min_samples=min_pts_step3).fit(bottom_df[['distance', 'height']])
                bottom_df['clusters'] = np.where(np.array(clustering.labels_) >= 0, 1, 0)
                bottom_df_window = bottom_df.loc[bottom_df['clusters'] > 0].rolling(window=window_bottom_line, min_periods=min_periods_bottom_line).mean()

                # plot a scatter with both surface lines and the photon classification
                f1, ax1 = plt.subplots(figsize=(20, 20))
                ax1.scatter(bottom_df['distance'], bottom_df['height'], c=bottom_df['clusters'], cmap='Set2', marker=',',
                            s=0.5)

                ax1.plot(bottom_df_window.loc[:, 'distance'], bottom_df_window.loc[:, 'height'])

                ax1.set_title('classification gt1l' + "- for photons {} and EPS {}".format(min_lon_index, eps3))
                ax1.get_xaxis().set_tick_params(which='both', direction='in')
                ax1.get_yaxis().set_tick_params(which='both', direction='in')
                ax1.set_xlabel('distance')
                ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')
                outpath = os.path.join(lake_dir, os.path.basename(fn)[:-4],
                                       'photon_classification_step3_ph_{}_{}_{}_{}.png'.format(min_lon_index,
                                                                classification_method, eps_method, min_pts_step2))
                plt.savefig(outpath)

                print('i')


                ## save clustering of all individual photons
                data_df['clusters'] = clusters.copy() #noise = 0 , data = 1 #surface = 2, bottom =3, bottom_noise=5
                clusters2_fill = np.where(np.array(data_df2['clusters']) > 0, 2, 0)  # everywhere not noise fill with labels cluster 2
                data_df['clusters'][clusters > 0] = clusters2_fill
                clusters3_fill = np.where(np.array(bottom_df['clusters']) > 0, 3, 5)
                clusters_non_noise = data_df['clusters'][clusters > 0].copy()  # get copy of clusters without noise photons
                clusters_non_noise[(data_df2['clusters2'] == 0) & (df_interp['diff'] < buffer_bottom_line)] = clusters3_fill  # fill everywhere bottom photons are with result of cluster 3
                # # & (df_interp['diff'] < (-buffer_factor_step3 * eps2))]
                data_df['clusters'][clusters > 0] = clusters_non_noise

                plt.ioff()
                f1, ax1 = plt.subplots(figsize=(20, 20))
                ax1.scatter(data_df['distance'], data_df['height'], c=data_df['clusters'], cmap='Set2', marker=',',
                            s=0.5)

                ax1.plot(data_df2_window.loc[:, 'distance'], data_df2_window.loc[:, 'height'])
                ax1.plot(bottom_df_window.loc[:, 'distance'], bottom_df_window.loc[:, 'height'])

                ax1.set_title('classification gt1l' + "- for photons {}".format(min_lon_index))
                ax1.get_xaxis().set_tick_params(which='both', direction='in')
                ax1.get_yaxis().set_tick_params(which='both', direction='in')
                ax1.set_xlabel('distance')
                ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')

                outpath = os.path.join(lake_dir, os.path.basename(fn)[:-4],
                                       'FINAL_classification_MEAN_ph_{}.png'.format(min_lon_index))
                plt.savefig(outpath)




