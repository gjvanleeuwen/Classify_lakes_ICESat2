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
    data_dir = "F:/onderzoeken/thesis_msc/data/Sentinel/20190620"

    calculate_NDWI = False
    calculate_NDWI_mask = False
    resample_SWIR = False
    get_ROI = True

    overwrite_NDWI = False
    overwrite_NDWI_mask = False
    overwrite_SWIR = False

    s2_band_list = ['B03_10', 'B04_10', 'B08_10'] # Green, Red, NIR, SWIR
    NDWI_calc = '(A - B) / (A + B)'
    mask_calc = "A*logical_and(A<=1,A>=0.3)"

    s2_fn_list = pth.get_files_from_folder(data_dir, '*.SAFE')

    for subdir in s2_fn_list:

        s2_files = pth.get_sorted_s2_filelist(subdir, band_list=s2_band_list, recursive=True)

        NDWI_raster_dict = {'A': s2_files[0],
                            'B': s2_files[2]}
        NDWI_fn = s2_files[0][:-11] + "NDWI_10m.tif"

        if calculate_NDWI:

            if not pth.check_existence(NDWI_fn, overwrite=overwrite_NDWI):
                utl.log("Make NDWI " + NDWI_fn, log_level='INFO')
                gdal_calc(NDWI_raster_dict, NDWI_fn, NDWI_calc, dtype='Float32', no_data=-999,
                          create_options=("COMPRESS=LZW", "TILED=YES"),
                          file_format='GTiff', debug=True, overwrite=overwrite_NDWI)


        if calculate_NDWI_mask:

            NDWI_mask_fn = s2_files[0][:-11] + "NDWI_10m_mask.tif"
            NDWI_raster_dict2 = {'A': NDWI_fn}

            if not pth.check_existence(NDWI_mask_fn, overwrite=overwrite_NDWI_mask):
                utl.log("Make mask " + NDWI_mask_fn, log_level='INFO')
                gdal_calc(NDWI_raster_dict2, NDWI_mask_fn, mask_calc, dtype='Float32', no_data=-999,
                          create_options=("COMPRESS=LZW", "TILED=YES"),
                          file_format='GTiff', debug=True, overwrite=overwrite_NDWI_mask)


    if resample_SWIR:
        res = 10

        s2_fn_list = pth.get_files_from_folder(data_dir, '*.SAFE')

        for subdir in s2_fn_list:

            s2_files = pth.get_sorted_s2_filelist(subdir, band_list=['B03_10', 'B11_20'], recursive=True)
            resample_fn = s2_files[0][:-11] + "B11_10m.tif"

            if not pth.check_existence(resample_fn, overwrite=overwrite_SWIR):
                utl.log("resampling the SWIR band for : {}".format(s2_files[1]), log_level='INFO')

                ds = gdal.Warp(resample_fn, s2_files[1], xRes=res, yRes=-res, format='GTIFF',
                               creationOptions=['COMPRESS=LZW', 'TILED=YES'])
                ds = None


    if get_ROI:
        classification_dir = 'F:/onderzoeken/thesis_msc/Exploration/data/lake_class'
        classification_df_fn_list = pth.get_files_from_folder(classification_dir, '*1222*gt*l*_class.csv')
        s2_band_list = ['NDWI_10m', 'B03_10', "B04_10"]

        training_data_dir = "F:/onderzoeken/thesis_msc/data/Training/"

        s2_dir_list = pth.get_files_from_folder(data_dir, '*.SAFE')

        # loop through the different beam geometries
        for i, fn in enumerate(classification_df_fn_list):
            utl.log('Loading Beam file: {}'.format(fn), log_level="INFO")
            classification_df = pd.read_csv(fn, usecols=['lon', 'lat', 'lake_rolling', 'clusters'])
            classification_df = classification_df[classification_df['clusters'] == 5]

            # loop through the various S2 scenes for this date
            for i, subdir in enumerate(s2_dir_list):
                s2_files = pth.get_sorted_s2_filelist(subdir, band_list=s2_band_list, recursive=True,
                                                      extension='*')
                utl.log('Loading Sentinel image {}/{} -- name: {}'.format(i, len(s2_dir_list), subdir), log_level="INFO")
                # loop through the different S2 Bands
                for s2_fn, band in zip(s2_files, s2_band_list):
                    utl.log('Loading band {}'.format(band), log_level="INFO")

                    RB = RasterBand(s2_fn, check_file_existence=True)
                    srs = osr.SpatialReference()
                    srs.SetWellKnownGeogCS("WGS84")
                    proj = srs.ExportToWkt()
                    RB = RB.warp(projection=proj)
                    values, index = RB.get_values_at_coordinates(classification_df['lon'].values, classification_df['lat'].values)
                    index = index[0][np.where((values != RB.no_data_value) & (~np.isnan(values)))]
                    values = values[np.where((values != RB.no_data_value) & (~np.isnan(values)))]

                    if len(values) > 0:
                        utl.log('Icesat track overlays with Sentinel image - Adding data to dataframe', log_level='INFO') #, sentinel extent {}, Coordinates extent {}'.format(RB.extent.to_tuple(),
                                    #[classification_df['lon'].min(), classification_df['lon'].max(), classification_df['lat'].min(), classification_df['lat'].max()]), log_level='INFO')
                        if not band in classification_df.columns:
                            classification_df[band] = classification_df['lon'][classification_df['lon'] == 0]
                        classification_df[band].iloc[index] = values
                        # print(min(values), max(values))
                    else:
                        utl.log('NO match found - Sentinel image does not overlay ICESat Track',log_level='INFO') # sentinel extent {}, Coordinates extent {}'.format(RB.extent.to_tuple(),
                                    #[classification_df['lon'].min(), classification_df['lon'].max(), classification_df['lat'].min(), classification_df['lat'].max()]), log_level='INFO')

            outdir = os.path.join(training_data_dir, pth.get_filname_without_extension(fn))

            if not pth.check_existence(outdir):
                os.mkdir(outdir)

            classification_df.to_csv(os.path.join(outdir, "classification_with_s2_info.csv"))









