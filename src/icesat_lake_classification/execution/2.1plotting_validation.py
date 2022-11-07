import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import icesat_lake_classification.validation as validation
import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth


if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None  # default='warn'

    NDWI_threshold = 0.21
    s2_date = '20190617_L1C'
    base_dir = 'F:/onderzoeken/thesis_msc/'
    s2_band_list = ['NDWI_10m', 'B03', "B04", "B08", "B11", "B12"]

    figures_dir = os.path.join(base_dir, 'figures', s2_date)
    data_dir = os.path.join(base_dir, 'data', s2_date)

    classification_df_fn_list = pth.get_files_from_folder(os.path.join(data_dir, 'classification'), '*gt*l.h5')

    ## Processing
    if not pth.check_existence(os.path.join(figures_dir, 'final')):
        os.mkdir(os.path.join(figures_dir, 'final'))

    utl.log(classification_df_fn_list, log_level='INFO')
    utl.log('Plotting the Empirical relations', log_level='INFO')

    empirical_df = pd.read_csv(pth.get_files_from_folder(os.path.join(data_dir, 'empirical'),'*.csv')[0])
    validation.estimate_relations2(empirical_df, figures_dir, s2_date)

    empirical_df = empirical_df.groupby('NDWI_10m', as_index=False).median()
    validation.estimate_relations_CV(empirical_df, ['NDWI_10m', 'B03', "B04", "B08", "B11", "B12"], figures_dir, s2_date, folds=25,
                                     test_size=0.15, path_addition='CV')



    # empirical_df = pd.read_csv(pth.get_files_from_folder(os.path.join(data_dir, 'empirical'), '*.csv')[0])
    # empirical_df = empirical_df.groupby('NDWI_10m', as_index=False).median()
    #
    # parameters_green, parameters_red, parameters_green_physical, parameters_red_physical, model = validation.estimate_relations(
    #     empirical_df, ['B03', "B04", "B08", "B11", "B12"])
    # utl.log('Calculating empircal/physical depth lines for the plot', log_level='INFO')
    # # classification_df = validation.calculate_depth_from_relations (classification_df,
    # #                             parameters_green, parameters_red, parameters_green_physical,
    # #                             parameters_red_physical, model, ['B03', "B04", "B08", "B11", "B12"])









