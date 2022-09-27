import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth


if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    plt.ioff()
    pd.options.mode.chained_assignment = None  # default='warn'

    if plot_lakes:
        ### make some graphs
        utl.log('Make graphs of the calculated mode', log_level='INFO')
        plt.ioff()
        ph_per_image = 100000
        n_ph = len(classification_df)

        if not pth.check_existence(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10])):
            os.mkdir(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10]))

        start_index_array = np.arange(0, n_ph, ph_per_image)
        end_index_array = np.append(np.arange(ph_per_image, n_ph, ph_per_image), n_ph - 1)

        for i, (ph_start, ph_end) in enumerate(zip(start_index_array, end_index_array)):

            if i < 75:
                continue

            index_slice = slice(ph_start, ph_end)
            utl.log("Plotting slice {}/{} - Photon count {}".format(int(ph_start / ph_per_image), len(start_index_array),
                                                                    ph_start), log_level='INFO')

            # create dataframe with just 2 variables
            data_df = classification_df[
                ['lon', 'lat', 'distance', 'height', 'clusters', 'bottom_distance', 'bottom_height', 'surface_distance',
                 'surface_height', 'lake', 'lake_rolling']].iloc[index_slice]

            # clusters - 0 noise, - 1 signal, 2= surface, - 3 bottom, #photons close to surface =4, bottom sure = 5
            cluster_map = {1: 'darkgrey', 0: 'bisque', 2: 'dodgerblue', 3: 'wheat', 4: 'red', 5: 'mediumseagreen'}
            c_cluster = [cluster_map[x] for x in data_df['clusters']]

            f1, ax1 = plt.subplots(figsize=(20, 20))
            ax1.scatter(data_df['distance'], data_df['height'], c=c_cluster, marker=',',
                        s=0.5)

            result_map = {1: 'red', 0: 'green', 2: 'orange', 3: 'yellow'}
            c_bottom = [result_map[x] for x in data_df['lake_rolling']]

            # plot bottom_line
            points = np.array([data_df.loc[:, 'bottom_distance'], data_df.loc[:, 'bottom_height']]).T.reshape(-1, 1, 2)
            lines = np.concatenate([points[:-1], points[1:]], axis=1)
            colored_lines = LineCollection(lines, colors=c_bottom, linewidths=(2,))
            ax1.add_collection(colored_lines)

            # plot surface line
            ax1.plot(data_df.loc[:, 'surface_distance'], data_df.loc[:, 'surface_height'])

            ax1.set_title('classification gt1l' + "- for photons {}".format(ph_start))
            ax1.get_xaxis().set_tick_params(which='both', direction='in')
            ax1.get_yaxis().set_tick_params(which='both', direction='in')
            ax1.set_xlabel('distance')
            ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')

            outpath = os.path.join(os.path.join(base_dir, 'figures/', os.path.basename(fn)[:-10],
                                                'FINAL_classification_lon_{}_lat_{}_ph_{}_distance_median.png'.format(
                                                    np.round(data_df['lon'].iloc[0], 2),
                                                    np.round(data_df['lat'].iloc[0], 2),
                                                    ph_start, np.round(data_df['distance'].iloc[0], 2))))
            plt.savefig(outpath)
            plt.close('all')