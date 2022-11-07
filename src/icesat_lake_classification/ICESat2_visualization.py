import numpy as np
import matplotlib.pyplot as plt
import math

import icesat_lake_classification.utils as utl

def plot_classified_photons(data_df, clusters, ph_start, eps, outpath, add_line=False, line_data=None):
    plt.ioff()
    f1, ax1 = plt.subplots(figsize=(10, 10))

    result_map_surface = {0: 'Tan', 1: 'darkgrey', 2: 'darkgrey', 3: 'cornflowerblue'}
    colors = [result_map_surface[x] if not math.isnan(x) else result_map_surface[0]
                                      for x in clusters.values]

    ax1.scatter(data_df['distance'], data_df['height'], c=colors, marker=',',
                s=0.3)
    if add_line:
        ax1.plot(line_data.iloc[:,0], line_data.iloc[:,1])

    ax1.set_title('DBSCAN - EPS {}'.format(np.round(eps,3)), fontsize=16)
    ax1.get_xaxis().set_tick_params(which='both', direction='in', labelsize=16)
    ax1.get_yaxis().set_tick_params(which='both', direction='in', labelsize=16)

    ax1.set_ylim(np.median(data_df['height'].values) - 20, np.median(data_df['height'].values) + 20)
    ax1.set_xlim(np.min(data_df['distance'].values), np.max(data_df['distance'].values))

    ax1.set_xlabel('Along-Track Distance [m]', fontsize=22)
    ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]', fontsize=22)
    plt.savefig(outpath)


def get_confusion_matrix (classification_df):

    classification_df = classification_df.iloc[np.where((classification_df['SurfNoiseR'] > 0.5)
                                                        & (classification_df['SurfBottR'] > 1)
                                                        & (classification_df['SurfBottR'] < 10)
                                                        & (classification_df['dem_diff'] < 400)
                                                        & (classification_df['range'] < 400)
                                                        & (classification_df['slope_mean'] < 0.1)
                                                        & (classification_df['NDWI_10m'] > 0))]

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

    utl.log('CM UNO True positives {}, False positives {}, True negatives {}, False Negatives {}'.format(
        classification_df['CM'].value_counts()[1], classification_df['CM'].value_counts()[2],
        classification_df['CM'].value_counts()[3], classification_df['CM'].value_counts()[4]), log_level='INFO')

    total = classification_df['CM'].value_counts()[1] + classification_df['CM'].value_counts()[2] + \
            classification_df['CM'].value_counts()[3] + classification_df['CM'].value_counts()[4]
    total = total / 100

    accuracy =  (classification_df['CM'].value_counts()[1] + classification_df['CM'].value_counts()[3])/total
    precision = classification_df['CM'].value_counts()[1] / (classification_df['CM'].value_counts()[1] + classification_df['CM'].value_counts()[2] )
    sensitivity = classification_df['CM'].value_counts()[1] / (
                classification_df['CM'].value_counts()[1] + classification_df['CM'].value_counts()[4])
    utl.log('accuracy {} --- Precision {} --- Sensitivity {}'.format(accuracy, precision, sensitivity), log_level="INFO")

    utl.log('True positives {}, False positives {}, True negatives {}, False Negatives {}'.format(
        classification_df['CM'].value_counts()[1] / total, classification_df['CM'].value_counts()[2] / total,
        classification_df['CM'].value_counts()[3] / total, classification_df['CM'].value_counts()[4] / total),
        log_level='INFO')

    return classification_df['CM'].value_counts()[1], classification_df['CM'].value_counts()[2],classification_df['CM'].value_counts()[3], classification_df['CM'].value_counts()[4]


def get_confusion_matrix2(classification_df):

    classification_df = classification_df.iloc[np.where((classification_df['SurfNoiseR'] > 0.5)
                                                        & (classification_df['SurfBottR'] > 1)
                                                        & (classification_df['SurfBottR'] < 10)
                                                        & (classification_df['dem_diff'] < 400)
                                                        & (classification_df['range'] < 400)
                                                        & (classification_df['slope_mean'] < 0.1)
                                                        & (classification_df['NDWI_10m'] > 0))]

    classification_df['CM'] = np.where(
        (classification_df['lake_rolling'] == 1) | (classification_df['lake_rolling'] == 2) & (classification_df['NDWI_class'] == 1), 1, 0)
    classification_df['CM'][classification_df['CM'] == 0] = np.where(
        (classification_df['lake_rolling'][classification_df['CM'] == 0] == 1) | (classification_df['lake_rolling'][classification_df['CM'] == 0] == 2) & (
                classification_df['NDWI_class'][classification_df['CM'] == 0] != 1), 2, 0)
    classification_df['CM'][classification_df['CM'] == 0] = np.where(
        (classification_df['lake_rolling'][classification_df['CM'] == 0] == 0) | (classification_df['lake_rolling'][classification_df['CM'] == 0] ==3) & (
                classification_df['NDWI_class'][classification_df['CM'] == 0] != 1) & (
                classification_df['CM'] == 0), 3, 0)
    classification_df['CM'][classification_df['CM'] == 0] = np.where(
        (classification_df['lake_rolling'][classification_df['CM'] == 0] == 0) | (classification_df['lake_rolling'][classification_df['CM'] == 0] ==3) & (
                classification_df['NDWI_class'][classification_df['CM'] == 0] == 1) & (
                classification_df['CM'] == 0), 4, 0)

    utl.log('CM2 True positives {}, False positives {}, True negatives {}, False Negatives {}'.format(
        classification_df['CM'].value_counts()[1], classification_df['CM'].value_counts()[2],
        classification_df['CM'].value_counts()[3], classification_df['CM'].value_counts()[4]), log_level='INFO')

    total = classification_df['CM'].value_counts()[1] + classification_df['CM'].value_counts()[2] + \
            classification_df['CM'].value_counts()[3] + classification_df['CM'].value_counts()[4]
    total = total / 100

    accuracy =  (classification_df['CM'].value_counts()[1] + classification_df['CM'].value_counts()[3])/total
    precision = classification_df['CM'].value_counts()[1] / (classification_df['CM'].value_counts()[1] + classification_df['CM'].value_counts()[2] )
    sensitivity = classification_df['CM'].value_counts()[1] / (
                classification_df['CM'].value_counts()[1] + classification_df['CM'].value_counts()[4])
    utl.log('accuracy {} --- Precision {} --- Sensitivity {}'.format(accuracy, precision, sensitivity), log_level="INFO")

    utl.log('True positives {}, False positives {}, True negatives {}, False Negatives {}'.format(
        classification_df['CM'].value_counts()[1] / total, classification_df['CM'].value_counts()[2] / total,
        classification_df['CM'].value_counts()[3] / total, classification_df['CM'].value_counts()[4] / total),
        log_level='INFO')

    return classification_df['CM'].value_counts()[1], classification_df['CM'].value_counts()[2],classification_df['CM'].value_counts()[3], classification_df['CM'].value_counts()[4]
