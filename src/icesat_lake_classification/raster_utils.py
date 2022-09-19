import os
import sys
import re
import shutil
import tempfile
import subprocess

from uuid import uuid4
from osgeo import gdal
import numpy as np
from retrying import retry
from functools import wraps

from osgeo import osr, ogr
from shapely.geometry import shape

import icesat_lake_classification.utils as utl

retry_args = dict(wait_exponential_multiplier=1000,
                  wait_exponential_max=5000,
                  stop_max_attempt_number=5)


@retry(**retry_args)
def move_file(local_fname,target):

    """
    Move a file another directory depending on the target string.

    Parameters
    ----------
    local_fname: string
        filename on local disk
    target: string
        target filename on local disk or object name
        if object name then the string has to start with /vsigs or gs://
    """
    return shutil.move(local_fname, target)



def _retry_on_tiff_read_fail(exception):
    """Return True if we should retry. This usually occurs when gdal tries to read
    from a google cloud bucket, but fails. In this case, the cache must be cleared
    """
    utl.log('RuntimeError TIFFReadEncodedTile', 'WARNING')
    gdal.VSICurlClearCache()
    return isinstance(exception, RuntimeError)


retry_on_tiff_read_fail_decorator = retry(retry_on_exception=_retry_on_tiff_read_fail,
                                          wait_fixed=2000,
                                          stop_max_attempt_number=5)




def get_gdal_datatype(numpy_dtype):
    """
    Get GDAL datatype from numpy.dtype.
    Float64 is transformed to float32
    Int64 is transformed to int32
    """
    from osgeo import gdal_array
    # scipy.misc.imread error for float64
    if numpy_dtype == np.float64:
        numpy_dtype = np.float32
    if numpy_dtype == np.int64:
        numpy_dtype = np.int32
    return gdal_array.NumericTypeCodeToGDALTypeCode(numpy_dtype)

def gdal_temporary_use_exceptions(func):
    """Temporary usage of gdal Exceptions, and
    revert to the old state once the function returns"""
    @wraps(func)
    def temp_use_exceptions(*args, **kwargs):
        gdal_use_exceptions_old = bool(gdal.GetUseExceptions())
        gdal.UseExceptions()
        value = func(*args, **kwargs)
        if not gdal_use_exceptions_old:
            gdal.DontUseExceptions()
        return value
    return temp_use_exceptions


@gdal_temporary_use_exceptions
def write_geotiff(file_name, raster, geo_transform, projection=None, no_data_value=None,
                  create_options=['COMPRESS=LZW', 'TILED=YES'],
                  scale_factor=1, offset=0, ct=None,
                  metadata=None,
                  metadata_domain=None):
    """
    Write a GeoTIFF file

    Parameters
    ----------
    file_name: string
        file name of the GeoTIFF file
        if it starts with /vsigs the file will first be stored in a temporary directory and
        then uploaded to google cloud
    raster: numpy.ndarray
        data to write
        Can be a list of numpy array to write in multibands in the order of the list provided
        If only a numpy array is provided, a singleband geotiff is written
    geo_transform: tuple of float
        geo transform tuple (lx, xres, xrot, uy, yrot, yres)
    projection: list, optional
        GDAL projection information
    no_data_value: int or float, optional
        Value to use as no-date in output file
    create_options: list, optional
        List of creation options. See http://www.gdal.org/frmt_gtiff.html
    scale_factor: float, optional
        Scale factor applied to the data to be stored in the GeoTIFF.
        This scale factor will be applied when reading the data.
    offset: float, optional
        Offset applied to the data to be stored in the GeoTIFF.
        Used together with scale_factor
    ct: gdal.ColorTable, optional
        if ct, forces legend of first band in the written geotiff to supplied gdal.ColorTable object
    metadata: dict, optional
        dictionary of metadata fields to add to the geotiff
    metadata_domain: string, optional
        If metadata should be written into another domain than the default.
        Metadata in the default domain will be visible by e.g. gdalinfo.
    storage_client: :py:class:`google.gcloud.storage.Client`
        Set to not create a client for each request.
        Will make the query more stable and faster
    """
    from osgeo import gdal, gdal_array, osr

    if not isinstance(raster, (list,)):
        raster=[raster]

    cols = raster[0].shape[1]
    rows = raster[0].shape[0]
    driver = gdal.GetDriverByName("GTiff")
    data_type = get_gdal_datatype(raster[0].dtype)

    ds = driver.Create(file_name, cols, rows, len(raster), data_type, create_options)
    # (top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution)
    ds.SetGeoTransform(geo_transform)
    if not projection:
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS("WGS84")
        projection = srs.ExportToWkt()
    ds.SetProjection(projection)
    if metadata is not None:
        ds.SetMetadata(metadata, metadata_domain)

    # write the raster
    for i_band in range(len(raster)):
        band = ds.GetRasterBand(i_band+1)
        band.SetScale(scale_factor)
        band.SetOffset(offset)
        # only set colortable of first band, if ct is supplied
        if i_band == 0:
            if ct:
                band.SetColorTable(ct)
        if not no_data_value is None:
            band.SetNoDataValue(np.double(no_data_value))

        band.WriteArray(raster[i_band])
    ds.FlushCache()
    ds = None


class BoundingBox(object):
    """
    Bounding Box object defining a rectangular bounding box given by top left and bottom right corner coordinates

    Parameters
    ----------
    topleft_x: float
        top left x value
    topleft_y: float
        top left y value
    bottomright_x: float
        bottom right x value
    bottomright_y: float
        upper right y value
    srid: int, optional
        SRID number of the coordinates
    """
    def __init__(self, topleft_x, topleft_y, bottomright_x, bottomright_y,
                 srid=4326):
        self.topleft_x = topleft_x
        self.topleft_y = topleft_y
        self.bottomright_x = bottomright_x
        self.bottomright_y = bottomright_y
        self.srid = srid

    def to_tuple(self):
        """
        Return as tuple (x_ul, y_ul, x_lr, y_lr)

        Returns
        -------
        bounding box: tuple
            bounding box in the following format (topleft x, topleft y, bottomright x, bottomright y)
        """
        return self.topleft_x, self.topleft_y, self.bottomright_x, self.bottomright_y

    def to_transwin_tuple(self):
        """
        Return as tuple formatted for `projWin` and `srcWin` options in gdal.TranslateOptions

        Returns
        -------
        bounding box: tuple
            bounding box in the following format (topleft x, topleft y, bottomright x, bottomright y)
        """
        return self.to_tuple()

    def to_warpbounds_tuple(self):
        """
        Return as tuple formatted for `outputBounds` options in gdal.WarpOptions

        for gdal.TranslateOptions use `to_tuple` method

        Returns
        -------
        bounding box: tuple
            bounding box in the following format (topleft x, bottomright y, bottomright x, topleft y)
        """
        return self.topleft_x, self.bottomright_y, self.bottomright_x, self.topleft_y

    def to_shape(self):
        """
        Convert boundingbox into Polygon geometry

        Returns
        -------
        polygon: shapely.geometry.polygon.Polygon
            Polygon shape over bbox
        """
        geojson = {'type': 'Polygon', 'coordinates': [
            [[self.topleft_x, self.bottomright_y],
             [self.topleft_x, self.topleft_y],
             [self.bottomright_x, self.topleft_y],
             [self.bottomright_x, self.bottomright_y],
             [self.topleft_x, self.bottomright_y]]]}
        return shape(geojson)

    @property
    def top(self):
        """
        top property

        Returns
        -------
        top: float
            top value of the rectangular bounding box
        """
        return self.topleft_y

    @property
    def bottom(self):
        """
        bottom property

        Returns
        -------
        bottom: float
            bottom value of the rectangular bounding box
        """
        return self.bottomright_y

    @property
    def left(self):
        """
        left property

        Returns
        -------
        left: float
            left value of the rectangular bounding box
        """
        return self.topleft_x

    @property
    def right(self):
        """
        right property

        Returns
        -------
        right: float
            right value of the rectangular bounding box
        """
        return self.bottomright_x

    def to_ogr_geom(self):
        """
        Convert the bounding box into a 2D ORG geometry

        Returns
        -------
        polygon: :py:class:`ogr.Geometry`
            type ogr.wkbPolygon
        """

        # Create ring
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(self.topleft_x, self.bottomright_y)
        ring.AddPoint(self.topleft_x, self.topleft_y)
        ring.AddPoint(self.bottomright_x, self.topleft_y)
        ring.AddPoint(self.bottomright_x, self.bottomright_y)
        ring.AddPoint(self.topleft_x, self.bottomright_y)

        # Create polygon
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(self.srid)
        poly.AssignSpatialReference(srs)
        poly.Set3D(False)
        return poly



def gdal_calc(input_raster_dict, output_raster, calc,
              band_dict=None, dtype='Byte', no_data=255,
              create_options=("COMPRESS=LZW", "TILED=YES"),
              file_format='GTiff', allbands=None,
              debug=True, overwrite=True, *args):
    """
    Use the Raster Calculator gdal_calc.py in your python code using this function
    The scipt is executed in a subprocess

    Parameters
    ----------
    input_raster_dict: dict
        Dictionary mapping keys [A-Z] to gdal compatible files e.g.
        {'A': 'test-1.tif', 'B': 'test-2.tif'}
    output_raster: str
        Location to write output to
    calc: str
         Expression to calculate, can include any numpy function e.g.
         'A/2 + log10(B)'
    band_dict: dict
        Mapping alias key [A-Z] to bandnr value (1-999)
    dtype: str
        output datatype, must be one of ['Int32', 'Int16', 'Float64',
        'UInt16', 'Byte', 'UInt32', 'Float32']
    no_data:
        Output no_data value
    create_options: tuple of str
    file_format: str
        gdal File Format
    allbands: str
        Use allbands for of given raster (A-Z)
    debug: bool
    overwrite: bool
    args
        Additional arguments are parsed directly to the gdal_calc cli
    """
    if sys.platform.startswith('win'):
        cmd_list = ['python', os.path.join(os.environ['CONDA_PREFIX'], 'Scripts', 'gdal_calc.py')]
    else:
        cmd_list = ['gdal_calc.py']
    for alias, fn in input_raster_dict.items():
        if not re.match(r'[A-Z]', alias):
            raise ValueError('Input product dict alias is not in [A-Z]')
        elif not gdal.Open(fn):
            raise RuntimeError('Input product dict filename {} does not exist'.format(fn))
        cmd_list.extend(['-{}'.format(alias), '{}'.format(fn)])
    if band_dict is not None:
        for alias, bandnr in band_dict.items():
            cmd_list.append('--{}_band={}'.format(alias, bandnr))
    cmd_list.append('--outfile={}'.format(output_raster))
    cmd_list.append('--calc={}'.format(calc))
    cmd_list.append('--type={}'.format(dtype))
    cmd_list.append('--NoDataValue={}'.format(no_data))
    cmd_list.append('--format={}'.format(file_format))
    if allbands is not None:
        cmd_list.append('--allBands={}'.format(allbands))
    for co in create_options:
        cmd_list.append('--co={}'.format(co))
    if overwrite:
        cmd_list.append('--overwrite')
    if debug:
        cmd_list.append('--debug')
    cmd_list.extend(list(args))
    subprocess.check_call(cmd_list)

def get_coordinate (geo_transform, row, col, correction=0.0):
    # + (row + correction) * geo_transform[2]
    lon = geo_transform[0] + (col + correction) * geo_transform[1]
    # + (col + correction) * geo_transform[4]
    lat = geo_transform[3] + (row + correction) * geo_transform[5]
    return lon, lat

def geotiff_roi_cut(gdal_ds_name, ogr_geom,
                    output_fname, output_format='GTIFF',
                    dstSRS=None, geomSRS=None, dstNodata=255,
                    crop_to_geom=False,
                    strict_SRS=False):
    """
    Get a geotiff which is subset from a given input dataset by
    both a lat, lon bounding box and a shapefile.

    Parameters
    ----------
    gdal_ds_name: string
        Dataset identifier which can be used by gdal.Open
    ogr_geom: ogr.Geometry
        OGR geometry object
    output_fname: string
        Filename of the output GeoTIFF to write
    output_format: string
        File type in which the raster will be written
    dstSRS: string, optional
        Destination SRS for gdal.Warp
    geomSRS: string, optional
        Geometry SRS for the creation of the layer
    dstNodata: float, int, optional
        Set this value as a no-data value in the output dataset.
        Also used as the no data value for regions that are cut
        by the shapefile.
    crop_to_geom: boolean, optional
        If set then the output file will be cropped to the shapefile extent
    strict_SRS: boolean, optional
        if a strict SRS check should be performed.
        when not set the default gdal behavior of assuming that the
        geometry has the same SRS as the dataset will be used
    """
    roi_subset_fname = '/vsimem/{}.vrt'.format(uuid4())
    drv = ogr.GetDriverByName('Esri Shapefile')
    roi_shp_fname = '/vsimem/{}.shp'.format(uuid4())
    roi_ds = drv.CreateDataSource(roi_shp_fname)

    if geomSRS is None:
        geomSRS = ogr_geom.GetSpatialReference()
        if geomSRS is None:
            if strict_SRS:
                raise ValueError('No geometry SRS is supplied and none can be '
                                 'retrieved from the geometry.')
            else:
                pass

    lyr = roi_ds.CreateLayer('roi', geomSRS, geom_type=ogr.wkbMultiPolygon)
    feat = ogr.Feature(lyr.GetLayerDefn())
    feat.SetGeometry(ogr_geom)
    lyr.CreateFeature(feat)
    (x1, x2, y2, y1) = lyr.GetExtent()
    feat = None
    roi_ds = None
    gdal.Warp(roi_subset_fname,
              gdal_ds_name,
              format='VRT',
              cropToCutline=crop_to_geom,
              cutlineDSName=roi_shp_fname,
              warpOptions=['CUTLINE_ALL_TOUCHED=TRUE'],
              dstNodata=dstNodata,
              dstSRS=dstSRS)

    gdal_options = dict()
    gdal_options['format'] = output_format
    # set gdal creation options only when outputformat is geotiff
    if output_format == 'GTIFF':
        gdal_options['creationOptions'] = ['COMPRESS=LZW', 'TILED=YES']

    gdal.Translate(output_fname, roi_subset_fname, **gdal_options)

    gdal.Unlink(roi_subset_fname)
    gdal.Unlink(roi_shp_fname)

def warp_raster_gcs_or_local (raster_path, out_path,
                              tmpdir, resolution=None, overwrite=False,
                              resampling_method='near', storage_client=None, warp_dict=None):
    """
    function that cals gdal warp on a s2 raster that is located on the google bucket or local and can place it in bucket or local.
    Parameters
    ----------
    raster_path : str
        full path to the raster on the bucket to be warped
    out_path : str
        full path and fn to bucket or local location for warped file
    tmpdir : str
        path to existing tmp directory, can be the same for multiple calls
    resolution : int, optional
        resolution for gdalwarp call, if none will keep resolution of original raster
    overwrite : bool, optional
        if True overwrites warped file if already exists, otherwise checks existence and skips processing if exists
    resampling_method : str, optional
        resampling method for gdal warp call, can be 'cubic' , 'bilinear' or 'near'
    out_path : str, optional
        given outpath if you want to hardcode it and deny s2 structure
    storage_client: :py:class:`google.gcloud.storage.Client`, optional
        Set to not create a client for each request.
        Will make the query more stable and faster
    Returns
    -------
    outpath of written warped raster
    """
    gdal.UseExceptions()
    out_path_local = os.path.join(tmpdir, os.path.basename(out_path))
    in_path = raster_path

    if utl.check_existence(out_path, overwrite=overwrite, storage_client=storage_client):
        return out_path
    utl.log(('warping image: ', os.path.basename(in_path), 'To image: ', os.path.basename(out_path)),
            log_level='INFO')

    warp_options = dict()
    warp_options['format'] = 'GTIFF'
    warp_options['creationOptions'] = ['TILED=YES', 'COMPRESS=LZW']
    warp_options['warpOptions'] = ['CUTLINE_ALL_TOUCHED=TRUE']
    if resolution:
        warp_options['xRes'] = resolution
        warp_options['yRes'] = resolution
    warp_options['resampleAlg'] = resampling_method
    if warp_dict:
        for kw in warp_dict:
            warp_options.update({kw: warp_dict[kw]})

    rb = RasterBand(in_path, check_file_existence=False)
    if not 'outputBounds' in list(warp_options.keys()):
        # need them as: output bounds as (minX, minY, maxX, maxY) in target SRS
        maxY, minY = rb.extent.top, rb.extent.bottom
        maxX, minX = rb.extent.right, rb.extent.left
        warp_options['outputBounds'] = (minX, minY, maxX, maxY)
    if not 'dstSRS' in list(warp_options.keys()):
        warp_options['dstSRS'] = rb.srs
    if not 'dstNodata' in list(warp_options.keys()):
        warp_options['dstNodata'] = rb.no_data_value

    warp_options['srcNodata'] = rb.no_data_value
    warp_options['srcSRS'] = rb.srs
    gdal.Warp(out_path_local, in_path, **warp_options)

    utl.log(('Moving warped file to bucket:', os.path.basename(out_path)), log_level='INFO')
    move_file(out_path_local, out_path, storage_client=storage_client)

def calc_index (path_1, path_2,
                nodata=65535, datatype='float'):
    """
    Function to calculate a standard normalized index from 2 paths and potentially scale it for integer datatype
    Parameters
    ----------
    path_1 : str
        full path to the index path that is on the left side of the equation in path_1 - path_2 / path_1 + path_2
    path_2 : str
        full path to the index path that is on the right side of the equation in path_1 - path_2 / path_1 + path_2
    datatype : str, optional
        datatype to force on output array, if startswith int or uint wil scale the float map with multiplier
        and addition to keep negative values and float precision
    nodata : int, optional
        nodata value of input raster and to force on output
    Returns
    -------
    result : np.array
        full 2d raster with index value for every pixel, scaled to integer if datatype == int/uint
        and always masked for nodata pixels
    """
    rb_1 = RasterBand(path_1, bandnr=1)[:]
    rb_2 = RasterBand(path_2, bandnr=1)[:]
    result = np.zeros(rb_1.shape, dtype=float)

    result[rb_1 != nodata] = (rb_1[rb_1 != nodata] - rb_2[rb_1 != nodata]) / (
            rb_1[rb_1 != nodata] + rb_2[rb_1 != nodata])

    if datatype.startswith('int') or datatype.startswith('uint'):
        result[rb_1 != nodata] = (result[rb_1 != nodata]) + 1
        result[(rb_1 != nodata) & (result < 0)] = 0
        result[rb_1 != nodata] = (result[rb_1 != nodata]) * 100

    result[rb_1 == nodata] = nodata
    result = result.astype(datatype)

    return result



