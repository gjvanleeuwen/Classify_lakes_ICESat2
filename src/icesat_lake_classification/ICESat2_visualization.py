import numpy as np
import matplotlib.pyplot as plt

import icesat_lake_classification.utils as utl

def plot_classified_photons(data_df, clusters, ph_start, eps, outpath, add_line=False, line_data=None):
    plt.ioff()
    f1, ax1 = plt.subplots(figsize=(20, 20))
    ax1.scatter(data_df['distance'], data_df['height'], c=clusters, cmap='Set2', marker=',',
                s=0.5)
    if add_line:
        ax1.plot(line_data.iloc[:,0], line_data.iloc[:,1])

    ax1.set_title('classification gt1l' + "- for photons {} and EPS {}".format(ph_start, eps))
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.set_xlabel('distance')
    ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')
    plt.savefig(outpath)


def get_confusion_matrix (classification_df):
    classification_df = classification_df.iloc[np.where((classification_df['SurfNoiseR'] > 2)
                                                        & (classification_df['dem_diff'] < 25)
                                                        & (classification_df['range'] < 200)
                                                        & (classification_df['slope_mean'] < 0.1))]

    classification_df['CM'] = np.where(
        (classification_df['lake_rolling'] == 1) & (classification_df['NDWI_class'] == 1), 1, 0)
    classification_df['CM'][classification_df['CM'] == 0] = np.where(
        (classification_df['lake_rolling'][classification_df['CM'] == 0] == 1) & (
                classification_df['NDWI_class'][classification_df['CM'] == 0] != 1), 2, 0)
    classification_df['CM'][classification_df['CM'] == 0] = np.where(
        (classification_df['lake_rolling'][classification_df['CM'] == 0] != 1) & (
                classification_df['NDWI_class'][classification_df['CM'] == 0] != 1) & (
                classification_df['CM'] == 0), 3, 0)
    classification_df['CM'][classification_df['CM'] == 0] = np.where(
        (classification_df['lake_rolling'][classification_df['CM'] == 0] != 1) & (
                classification_df['NDWI_class'][classification_df['CM'] == 0] == 1) & (
                classification_df['CM'] == 0), 4, 0)

    utl.log('True positives {}, False positives {}, True negatives {}, False Negatives {}'.format(
        classification_df['CM'].value_counts()[1], classification_df['CM'].value_counts()[2],
        classification_df['CM'].value_counts()[3], classification_df['CM'].value_counts()[4]), log_level='INFO')

    return classification_df['CM'].value_counts()[1], classification_df['CM'].value_counts()[2],classification_df['CM'].value_counts()[3], classification_df['CM'].value_counts()[4]




