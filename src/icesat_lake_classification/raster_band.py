
import warnings
from uuid import uuid4

# external packages
from osgeo import gdal, gdalconst, osr, gdal_array
import numpy as np

from icesat_lake_classification.raster_utils import get_coordinate, BoundingBox, retry_on_tiff_read_fail_decorator, write_geotiff


class RasterBand(object):
    """
    RasterBand constructor

    Parameters
    ----------
    raster_file_name: str
        Filename optionally including path
    check_file_existence: bool
        Try to open the dataset as check if the file can be accessed
        set to False by default
    close_dataset: bool
        Don't close the dataset after reading
    bandnr: int
        Number of the band to read from the underlying GeoTIFF file
    need_rw: bool, optional
        By default the raster is opened in GA_Readonly mode (only option for cloud based data).
        If this flag is set, the file is opened in GA_Update mode (useful for __setitem__ only)
    read_masked_scaled: bool
        Automatically apply mask, scale and offset during reading of the data
    vmin: int or float or None
        Values lower than vmin are masked. Mind that the masking is applied
        after applying mask, scale and offset
    vmax: int or float or None
        Values higher than vmax are masked. Mind that the masking is applied
        after applying mask, scale and offset

    """
    def __init__(self,
                 raster_file_name,
                 bandnr=1,
                 check_file_existence=False,
                 close_dataset=True,
                 need_rw=False,
                 read_masked_scaled=False,
                 vmin=None,
                 vmax=None):

        self._raster_fn = raster_file_name
        self._need_rw = need_rw
        self._band_nr = bandnr
        self._ds = None
        self._band = None
        self.read_masked_scaled = read_masked_scaled
        self._unscaled_dtype = np.float64
        self.ndim = 2
        self.vmin = vmin
        self.vmax = vmax
        self._close_dataset = close_dataset
        self._data_type = None
        self._gt = None
        self._projection = None
        self._xsize = None
        self._ysize = None
        self._no_data_value = None
        self._scale = None
        self._offset = None
        self._data = None
        self._metadata = None
        if check_file_existence:
            self._get_raster_info()

    @classmethod
    def from_array_gt(cls, raster, geo_transform,
                      file_name=None,
                      projection=None,
                      no_data_value=None,
                      read_masked_scaled=False,
                      need_rw=False,
                      **kwargs):
        """
        Write a (virtual) GeoTIFF and open it as RasterBand

        Parameters
        ----------
        raster: numpy.ndarray
            data to write, a singleband geotiff is written
        geo_transform: tuple of float
            geo transform tuple (lx, xres, xrot, uy, yrot, yres)
        file_name: string, optional
            file name of the GeoTIFF file
            if not provided, an in-memory geotiff will be created (into /vsimem/rasterband
            if it starts with /vsigs the file will first be stored in a temporary directory and
            then uploaded to google cloud
        projection: str, optional
            GDAL projection information, set to EPSG:4326 by default
        no_data_value: int or float, optional
            Value to use as no-date in output file
        read_masked_scaled: bool
            Automatically apply mask, scale and offset during reading of the data
        need_rw: bool, optional
            By default the raster is opened in GA_Readonly mode (only option for cloud based data).
            If this flag is set, the file is opened in GA_Update mode (useful for __setitem__ only)
        kwargs:
            keyword arguments parsed to libsat.gdal_utils.write_geotiff
            - create_options: list, optional
            - scale_factor
            - offset
            - metadata
            - metadata_domain
            - storage_client
        """
        if file_name is None:
            file_name = '/vsimem/rasterband/{}.tif'.format(uuid4())
        write_geotiff(file_name, raster, geo_transform,
                      projection=projection,
                      no_data_value=no_data_value,
                      **kwargs)
        rb = cls(file_name, read_masked_scaled=read_masked_scaled, need_rw=need_rw)
        return rb

    def _apply_mask_scale_offset(self, data):
        data = data.astype(self.unscaled_dtype)
        data = np.where(data == self.no_data_value, np.nan, data)
        return data * self.scale + self.offset

    def _remove_scale_offset(self, data):
        data = ((data - self.offset) / self.scale)
        data[np.isnan(data)] = self.no_data_value
        if self.dtype in [np.int, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64]:
            data = np.round(data).astype(self.dtype)

        return data

    def unscale_data(self):
        """
        Data is stored scaled in the geotiff so unscaling means to apply scale and offset.
        Same as supplying `read_masked_scaled=True` at __init__
        """
        self.read_masked_scaled = True
        if self._data is not None:
            self._data = None

    def scale_data(self):
        """
        Data is stored scaled in the geotiff so scaling means to
        not apply scale and offset to the raw data.
        Same as supplying `read_masked_scaled=False` at __init__
        """
        self.read_masked_scaled = False
        if self._data is not None:
            self._data = None

    def __setitem__(self, slc, data):
        """
        Support for 2D item assignment without steps
        """
        ds = gdal.Open(self._raster_fn, gdalconst.GA_Update)
        band = ds.GetRasterBand(self._band_nr)
        y_slice, x_slice = slc
        band.WriteArray(data,
                        xoff=x_slice.start,
                        yoff=y_slice.start)
        del band
        del ds

    @retry_on_tiff_read_fail_decorator
    def __getitem__(self, slc):
        """
        Support for 2D slicing without steps
        """
        y_slice, x_slice = (slice(None),) * 2
        reducex, reducey = (slice(None),) * 2
        x_start, win_xsize, y_start, win_ysize = 0, self.xsize, 0, self.ysize
        n = len(slc) if type(slc) is tuple else 1
        if n == 1:
            y_slice = slc
        elif n == 2:
            y_slice, x_slice = slc
        ds = gdal.Open(self.raster_file_name, gdalconst.GA_ReadOnly)
        band = ds.GetRasterBand(self._band_nr)

        if np.issubdtype(type(x_slice), np.integer):
            x_start = x_slice % self.shape[1]  # supports negative index
            win_xsize = 1
            reducex = 0
        elif type(x_slice) is slice:
            x_start, x_stop, x_step = x_slice.indices(self.shape[1])
            win_xsize = len(range(x_start, x_stop))
            if x_slice.step:
                raise ValueError('slice.step is not supported (x slice)')

        if np.issubdtype(type(y_slice), np.integer):
            y_start = y_slice % self.shape[0]  # supports negative index
            win_ysize = 1
            reducey = 0
        elif type(y_slice) is slice:
            y_start, y_stop, y_step = y_slice.indices(self.shape[0])
            win_ysize = len(range(y_start, y_stop))
            if y_slice.step:
                raise ValueError('slice.step is not supported (y slice)')

        data = band.ReadAsArray(xoff=int(x_start),
                                win_xsize=int(win_xsize),
                                yoff=int(y_start),
                                win_ysize=int(win_ysize))
        del band
        del ds
        fill = None
        if self.read_masked_scaled:
            data = self._apply_mask_scale_offset(data)
            fill = np.nan
        data = self._mask_invalid(data, fill=fill)
        return data[reducey, reducex]

    @property
    def raster_file_name(self):
        return self._raster_fn

    @property
    @retry_on_tiff_read_fail_decorator
    def ds(self):
        """
        Property returns the gdal dataset object in read only
        or GA_Update if need_rw is set

        Raises
        ------
        (IOError, OSError)
            Depending on python version if ds cannot be opened (in rw-mode)

        Returns
        -------
        ds: osgeo.gdal.Dataset
        """
        if self._ds is None:
            if self._need_rw:
                self._ds = gdal.Open(self.raster_file_name, gdalconst.GA_Update)
                if self._ds is None:
                    raise IOError('File can not be openend in rw mode {}'.format(self._raster_fn))
            else:
                self._ds = gdal.Open(self.raster_file_name, gdalconst.GA_ReadOnly)
                if self._ds is None:
                    raise IOError('File can not be opened {}'.format(self._raster_fn))

        return self._ds

    @property
    def band(self):
        """
        Property returns the gdal band object

        Returns
        -------
        band :osgeo.gdal.band
            RasterBand
        """
        if self._band is None:
            self._band = self.ds.GetRasterBand(self._band_nr)
        return self._band

    @property
    def data(self):
        """
        Read and return array. This property retains the array in memory

        Returns
        -------
        np.ndarray

        """
        if self._data is None:
            self._data = self[:]
        return self._data

    @property
    def values(self):
        """
        Read and return array, don't keep in-memory

        Returns
        -------
        np.ndarray
        """
        return self[:]

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.ds.GetMetadata_Dict(None)
        return self._metadata

    def _get_raster_info(self):
        """ Load all raster info into attributes
        """
        self._gt = self.ds.GetGeoTransform()
        self._projection = self.ds.GetProjection()
        self._xsize = self.ds.RasterXSize
        self._ysize = self.ds.RasterYSize
        self._no_data_value = self.band.GetNoDataValue()

        self._data_type = gdal_array.GDALTypeCodeToNumericTypeCode(self.band.DataType)
        self._scale = self.band.GetScale() if self.band.GetScale() is not None else 1.
        self._offset = self.band.GetOffset() if self.band.GetOffset() is not None else 0.
        self._metadata = self.ds.GetMetadata_Dict(None)

        if (self._no_data_value is None) and np.issubdtype(self._data_type, np.floating):
            self._no_data_value = np.nan

        if self._close_dataset:
            self._ds = None
            self._band = None

    @property
    def gt(self):
        """
        Property returns the geo_transform of the raster.

        Returns
        -------
        gt: tuple
            geo_transform[0]: top left x
            geo_transform[1]: w-e pixel resolution
            geo_transform[2]: 0 for north up
            geo_transform[3]: top left y
            geo_transform[4]: 0 for north up
            geo_transform[5]: n-s pixel resolution (negative value)
        """
        if self._gt is None:
            self._get_raster_info()
        return self._gt

    @property
    def column_pixelsize(self):
        return self.gt[1]

    @property
    def row_pixelsize(self):
        return self.gt[5]

    @property
    def extent(self):
        """Returns the extent (outer corner coordinates) as a BoundingBox object.

        Returns
        -------
        bbox: libsat.vds_bounding_box.BoundingBox
        """
        gt = self.gt
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * self.xsize
        miny = maxy + gt[5] * self.ysize
        bbox = BoundingBox(topleft_x=minx, topleft_y=maxy,
                           bottomright_x=maxx, bottomright_y=miny)
        return bbox

    @property
    def projection(self):
        if self._projection is None:
            self._get_raster_info()
        return self._projection

    @property
    def srs(self):
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(self.projection)
        return raster_srs

    @property
    def ysize(self):
        if self._ysize is None:
            self._get_raster_info()
        return self._ysize

    @property
    def xsize(self):
        if self._xsize is None:
            self._get_raster_info()
        return self._xsize

    @property
    def shape(self):
        return self.ysize, self.xsize

    @property
    def no_data_value(self):
        if self._no_data_value is None:
            self._get_raster_info()
        try:
            ndval = self._data_type(self._no_data_value)
        except TypeError:
            ndval = None
        return ndval

    @property
    def dtype(self):
        if self.read_masked_scaled:
            return self.unscaled_dtype
        else:
            if self._data_type is None:
                self._get_raster_info()
            return self._data_type


    def get_values_at_coordinates(self, x, y):
        """
        Get values from raster at given coordinates

        Parameters
        ----------
        x: float or np.ndarray
        y: float or np.ndarray

        Returns
        -------
        values: float or np.ndarray
            Raster values at coordinates (dtype
        """
        r = np.int32((y - self.gt[3]) / self.gt[5])  # this is the same as r = int((self._gt[3] - lat) / -self._gt[5])
        c = np.int32((x - self.gt[0]) / self.gt[1])

        if np.any(c < 0) or np.any(c > self.xsize) or np.any(r < 0) or np.any(r > self.ysize):
            index = np.where((c > 0) & (c < self.xsize) & (r > 0) & (r < self.ysize))
            r = r[index]
            c = c[index]

        return self.data[r, c], index


    def warp(self, file_name=None,
             bbox=None, xres=None, yres=None,
             dst_srs=None, projection=None,
             cutline=None, crop_to_cutline=False,
             read_masked_scaled=False, **kwargs):
        """
        Method warps this RasterBand using gdal.Warp

        Parameters
        ----------
        file_name: str
            if not provided, an in-memory dataset will be created (into /vsimem/rasterband)
        bbox: BoundingBox
            Cut output to this bounding box. Coordinates expected in dst_srs
            unless outputBoundsSrs is specified
        xres: float
            output resolution in pixel direction
        yres: float
            output resolution in line direction
        dst_srs: osr.SpatialReference
            Provide destination spatial reference. By default the same as source
        projection: str
            Wkt string containing projection information for the destination. Overrules dst_srs
        cutline: str
            Cutline dataset (shapefile) name
        crop_to_cutline: bool
            Cut output to the extent of cutline
        read_masked_scaled: bool
            Automatically apply mask, scale and offset during reading of the output data
        kwargs: dict
            Parsed to gdal.WarpOptions with additional defaults:
            - creationOptions: ['COMPRESS=LZW', 'TILED=YES']
            - targetAlignedPixels: True

        Returns
        -------
        rb: RasterBand
            The warped dataset loaded as RasterBand
        """
        if file_name is None:
            file_name = '/vsimem/rasterband/{}.tif'.format(uuid4())
        if projection is not None:
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromWkt(projection)
        elif dst_srs is None:
            dst_srs = self.srs
        if bbox is not None:
            bounds = (bbox.topleft_x, bbox.bottomright_y,
                      bbox.bottomright_x, bbox.topleft_y)
        else:
            bounds = kwargs.pop('outputBounds', None)
        co = kwargs.pop('creationOptions', ['COMPRESS=LZW', 'TILED=YES'])
        tap = False
        if xres and yres:
            tap = kwargs.pop('targetAlignedPixels', True)
        wo = gdal.WarpOptions(srcSRS=self.srs,
                              dstSRS=dst_srs,
                              outputBounds=bounds,
                              xRes=xres,
                              yRes=yres,
                              creationOptions=co,
                              targetAlignedPixels=tap,
                              cutlineDSName=cutline,
                              cropToCutline=crop_to_cutline,
                              **kwargs)
        band_vrt_fn = '/vsimem/rasterband/{}.vrt'.format(uuid4())
        to = gdal.TranslateOptions(bandList=[self._band_nr])
        ds = gdal.Translate(band_vrt_fn, self.ds, options=to)
        gdal.Warp(file_name, ds, options=wo)
        ds = None
        gdal.GetDriverByName("VRT").Delete(band_vrt_fn)
        rb = RasterBand(file_name, read_masked_scaled=read_masked_scaled)
        rb.band.SetScale(self.scale)
        rb.band.SetOffset(self.offset)
        return rb

