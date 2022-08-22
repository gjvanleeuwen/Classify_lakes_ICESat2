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

    base_dir = 'F:/onderzoeken/thesis_msc/'

    example_lakes = {'ID': [0, 1, 2, 3, 4, 5, 6],
                     'beam': ['gt2l', 'gt2l', 'gt2l', 'gt1r', 'gt1r', 'gt1l', 'gt3l'],
                     'lon': [(-49.05 ,-49.02), (-48.87, -48.84), (-48.38, -48.35), (-48.53, -48.50), (-48.45, -48.42), (-49.01, -48.98), (-48.79, -48.76)],
                     'lat': [(68.95, 69.04), (68.41, 68.50), (66.81, 66.90), (67.07, 67.17), (66.81, 66.90), (68.59, 68.68), (68.42, 68.51)]}

    plt.ioff()

    iteration_starter = 10
    classification_method = 'DBSCAN'
    cluster_method = 'BOOL'
    classification_dir = 'F:/onderzoeken/thesis_msc/Exploration/data'
    classification_df_fn_list = pth.get_files_from_folder(classification_dir, '*1222*gt2l*')

    for fn in classification_df_fn_list:
        utl.log(fn, log_level='INFO')
        classification_df = pd.read_csv(fn)
        n_ph = len(classification_df)
        ph_per_image = 50000

        start_index_array = np.arange(0, n_ph, ph_per_image)
        end_index_array = np.append(np.arange(ph_per_image, n_ph, ph_per_image), n_ph - 1)
        class_list = []

        for i, (ph_start, ph_end) in enumerate(zip(start_index_array, end_index_array)):
            if i < iteration_starter:
                continue

            index_slice = slice(ph_start,ph_end)
            utl.log(index_slice, log_level='INFO')

            data_df = classification_df[['distance', 'height']].iloc[index_slice]
            data_df['distance'] = data_df['distance'] - min(data_df['distance'])

            min_pts = 6
            # generate new min_pts
            if not pth.check_existence(os.path.join(base_dir, 'Exploration/figures/', os.path.basename(fn)[:-4])):
                os.mkdir(os.path.join(base_dir, 'Exploration/figures/', os.path.basename(fn)[:-4]))

            if classification_method == "DBSCAN":
                eps = find_optimal_eps(data_df, min_pts=min_pts, method='total',
                                       outpath=os.path.join(base_dir, 'Exploration/figures/', os.path.basename(fn)[:-4],
                                                            'histogram_EPS_ph_{}.png'.format(ph_start)))

                clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(data_df)

            elif classification_method == "OPTICS":
                eps = "NAN"
                clustering = OPTICS(min_samples=3).fit(data_df)
            else:
                break

            class_list.extend(list(clustering.labels_))

            if cluster_method == 'BOOL':
                clusters = np.where(np.array(clustering.labels_) >= 0,1,0)
            elif cluster_method == 'CONTINUOUS':
                clusters = np.array(clustering.labels_)
            else:
                break

            plot_classified_photons(data_df, clusters, base_dir, fn, ph_start, eps, classification_method)

        classification_df['class'] = class_list
        classification_df.to_csv(os.path.join(fn[:-4],'_class.csv'))


    for lake_ID in example_lakes['ID']:
        if lake_ID == 6:
            classification_df = pd.read_csv(os.path.join(base_dir, 'Exploration', 'testing_data_lake_{}.csv'.format(lake_ID)))
            classification_df.drop(classification_df.columns[0], axis=1, inplace=True)

            classification_df['distance'] = classification_df['distance'] - min(classification_df['distance'])
            data_df = classification_df.drop('beam', axis=1)
            data_df['norm_height'] = data_df['dem'] - data_df['height']
            data_df.drop(['norm_height', 'dem', 'lon', 'lat'], axis=1, inplace=True)
            data_df['height'] = np.array(data_df['height'] * 100).astype(np.int32)
            data_df['distance'] = np.array(data_df['distance'] * 100).astype(np.int32)

            scaler = StandardScaler()
            data_df_scaled = pd.DataFrame(scaler.fit_transform(data_df.values))
            data_df_normalized = pd.DataFrame(normalize(data_df_scaled.values))
            data_df_normalized.drop(data_df_normalized.columns[1], axis=1, inplace=True)

            for eps in np.arange(0.05,0.5,0.05):
                for min_samples in np.arange(5,155,10):
                    labels = np.zeros(len(classification_df))
                    for start in np.arange(0,len(classification_df), 1000):
                        end = start + 1000
                        if start+1000 > len(classification_df):
                            end = len(classification_df)-1
                        model = DBSCAN(eps=0.05, min_samples=50).fit(data_df_normalized.iloc[start:end])
                        labels[start:end] = model.labels_

                    fig, ax = plt.subplots(figsize=(20, 20))
                    # ax.scatter(data_dict['distance'] - min(data_dict['distance']), data_dict['height'], c=pred, cmap='viridis', marker='.',s=0.01)
                    ax.scatter(classification_df['distance'], classification_df['height'], c=labels, cmap='viridis', marker='.',
                               s=0.01)
                    # ax.scatter(data_dict['distance'] - min(data_dict['distance']), data_dict['dem'], c='blue', marker='.',s=0.01)

                    ax.set_title('Lake {} for beam {} - dbscan'.format(lake_ID, example_lakes['beam'][lake_ID]))
                    ax.get_xaxis().set_tick_params(which='both', direction='in')
                    ax.get_yaxis().set_tick_params(which='both', direction='in')
                    ax.set_xlabel('distance')
                    ax.set_ylabel('Elevation above WGS84 Ellipsoid [m]')

                    plt.savefig('F:/onderzoeken/thesis_msc/Exploration/figures/classification_exploration/test_dbscan_separate_{}_EPS{}_samples{}.png'.format(lake_ID, eps, min_samples))
            break


