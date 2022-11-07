import icesat_lake_classification.utils as utl
import icesat_lake_classification.path_utils as pth
from icesat_lake_classification.raster_utils import gdal_calc, geotiff_roi_cut, gdal_extract_data

import tempfile
import pandas as pd
from osgeo import gdal, ogr, osr
import os
from icesat_lake_classification.raster_band import RasterBand
import numpy as np

if __name__ == "__main__":

    utl.set_log_level(log_level='INFO')
    pd.options.mode.chained_assignment = None     # default='warn'

    base_dir = 'F:/onderzoeken/thesis_msc/'
    figures_dir = os.path.join(base_dir, 'figures')
    data_dir = os.path.join(base_dir, 'data')

    s2_data_dir = "F:/onderzoeken/thesis_msc/data/Sentinel/20190617_L1C"

    calculate_NDWI = True
    calculate_NDWI_mask = False
    resample_SWIR = False

    overwrite_NDWI = True
    overwrite_NDWI_mask = False
    overwrite_SWIR = False

    s2_band_list = ['B03', 'B04', 'B08', 'B02'] # Green, Red, NIR, blue
    NDWI_calc = '(A - B) / (A + B)'
    mask_calc = "A*logical_and(A>=0.21)"

    s2_fn_list = pth.get_files_from_folder(s2_data_dir, '*.SAFE')

    for subdir in s2_fn_list:

        s2_files = pth.get_sorted_s2_filelist(subdir, band_list=s2_band_list, recursive=True)

        NDWI_raster_dict = {'A': s2_files[3],
                            'B': s2_files[1]}
        NDWI_fn = s2_files[0][:-11] + "NDWI_10m.tif"

        if calculate_NDWI:

            if not pth.check_existence(NDWI_fn, overwrite=overwrite_NDWI):
                utl.log("Make NDWI " + NDWI_fn, log_level='INFO')
                gdal_calc(NDWI_raster_dict, NDWI_fn, NDWI_calc, dtype='Float32', no_data=-999,
                          create_options=("COMPRESS=LZW", "TILED=YES"),
                          file_format='GTiff', debug=True, overwrite=overwrite_NDWI)


        if calculate_NDWI_mask:

            NDWI_mask_fn = s2_files[0][:-11] + "NDWI_10m_mask.tif"
            NDWI_raster_dict2 = {'A': NDWI_fn, 'B': s2_files[0], 'C': s2_files[1]}

            if not pth.check_existence(NDWI_mask_fn, overwrite=overwrite_NDWI_mask):
                utl.log("Make mask " + NDWI_mask_fn, log_level='INFO')
                gdal_calc(NDWI_raster_dict2, NDWI_mask_fn, mask_calc, dtype='Float32', no_data=-999,
                          create_options=("COMPRESS=LZW", "TILED=YES"),
                          file_format='GTiff', debug=True, overwrite=overwrite_NDWI_mask)


    if resample_SWIR:
        res = 10

        s2_fn_list = pth.get_files_from_folder(s2_data_dir, '*.SAFE')

        for subdir in s2_fn_list:

            s2_files = pth.get_sorted_s2_filelist(subdir, band_list=['B03', 'B11', 'B12'], recursive=True)
            resample_fn = s2_files[0][:-11] + "B11_10m.tif"
            resample_fn2 = s2_files[0][:-11] + "B12_10m.tif"

            if not pth.check_existence(resample_fn, overwrite=overwrite_SWIR):
                utl.log("resampling the SWIR band - B11 - for : {}".format(s2_files[1]), log_level='INFO')

                ds = gdal.Warp(resample_fn, s2_files[1], xRes=res, yRes=-res, format='GTIFF',
                               creationOptions=['COMPRESS=LZW', 'TILED=YES'])
                ds = None

            if not pth.check_existence(resample_fn2, overwrite=overwrite_SWIR):
                utl.log("resampling the SWIR band - B12 -  for : {}".format(s2_files[2]), log_level='INFO')
                ds = gdal.Warp(resample_fn2, s2_files[2], xRes=res, yRes=-res, format='GTIFF',
                               creationOptions=['COMPRESS=LZW', 'TILED=YES'])
                ds = None












