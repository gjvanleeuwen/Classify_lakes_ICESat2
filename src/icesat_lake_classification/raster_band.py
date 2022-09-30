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
        Can also start with /vsigs or gs:// for files on cloud storage
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
    def offset(self):
        if self._offset is None:
            self._get_raster_info()
        return self._offset

    @property
    def scale(self):
        if self._scale is None:
            self._get_raster_info()
        return self._scale

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

    @property
    def unscaled_dtype(self):
        return self._unscaled_dtype

    @unscaled_dtype.setter
    def unscaled_dtype(self, dtype):
        if not issubclass(dtype, np.floating):
            raise TypeError('scaled_dtype property should be a subclass of np.floating')
        self._unscaled_dtype = dtype

    def _mask_invalid(self, arr, fill=None):
        if (self.vmin is not None) or (self.vmax is not None):
            if fill is None:
                fill = self.no_data_value
            arr = np.where(np.logical_and(arr > self.vmax if self.vmax is not None else True,
                                          arr < self.vmin if self.vmin is not None else True),
                           fill, arr)
        return arr

    def iter_data(self, chunksize_x=5000, chunksize_y=5000, overlap=0):
        """
        Iterate over the data of this RasterBand in chunks

        Parameters
        ----------
        chunksize_x: int, optional
        chunksize_y: int, optional
        overlap: int, optional

        Yields
        ------
        data: numpy.ndarray
        """
        for x_slice, y_slice in iter_slices(self.shape,
                                            chunksize_x=chunksize_x,
                                            chunksize_y=chunksize_y, overlap=overlap):
            yield self[y_slice, x_slice]


    def mean(self, chunksize_x=5000, chunksize_y=5000,
             use_area_weights=True,
             mask_rb=None,
             add_nodata=None):
        """
        Calculate the mean of the data array taking into account
        the area of each pixel and masking the defined no data value.

        Parameters
        ----------
        chunksize_x: int, optional
        chunksize_y: int, optional
        use_area_weights: boolean, optional
            if the projection is EPSG:4326 the pixels are weighted by
            the area. This is automatically disabled if the projection
            is a different one.
        mask_rb: :py:class:`vds_io.vds_raster_file.RasterBand` object
            zero - one mask to apply before calculation of the mean
            where the mask is zero the data will be masked before
            calculation
        add_nodata: list of values, optional
            all of these values will be treated as no data before calculating the mean

        Returns
        -------
        mean: float
            Mean value

        Raises
        ------
        ValueError: if no valid data was found
        """
        if add_nodata is not None:
            if type(add_nodata) != list:
                add_nodata = [add_nodata]

        sum_weights = 0
        sum_data = 0

        epsg_4326 = osr.SpatialReference()
        epsg_4326.ImportFromEPSG(4326)
        if epsg_4326.IsSame(self.srs) == 0:
            use_area_weights = False

        for (data, area), (x_slice, y_slice) in zip(self.iter_data_area(chunksize_x=chunksize_x,
                                                                        chunksize_y=chunksize_y),
                                                    iter_slices(self.shape,
                                                                chunksize_x=chunksize_x,
                                                                chunksize_y=chunksize_y)):
            if mask_rb is not None:
                mask_data = mask_rb[y_slice, x_slice]
                data[mask_data == 0] = self._no_data_value

            if add_nodata is not None:
                for add_nodata_value in add_nodata:
                    data[data == add_nodata_value] = self._no_data_value

            if self._no_data_value is not np.nan:
                valid = data != self._no_data_value
            else:
                valid = np.isfinite(data)
            if use_area_weights:
                sum_data = sum_data + np.sum(data[valid] * area[valid])
                sum_weights = sum_weights + np.sum(area[valid])
            else:
                sum_data = sum_data + np.sum(data[valid])
                sum_weights = sum_weights + data.size
        if sum_weights == 0:
            raise ValueError('No valid data found')
        return sum_data / sum_weights

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

    def get_coordinate(self, row, col):
        """
        Get the top-left pixel coordinate of given row-column combination
        only valid for vertically oriented rasters

        Parameters
        ----------
        row: int
        col: int
        """
        return get_coordinate(self.gt, row, col)


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

    def warp_like_rb(self, ref_rb, file_name=None, read_masked_scaled=False, **kwargs):
        """
        Warp this object to a RasterBand similar to a reference RasterBand

        The extent, pixelsize, spatial reference will be matched to ref_rb

        Parameters
        ----------
        ref_rb: RasterBand
            Object from which to infer
        file_name: str
            if not provided, an in-memory dataset will be created (into /vsimem/rasterband)
        read_masked_scaled: bool
            Automatically apply mask, scale and offset during reading of the output data
        **kwargs
            keyword arguments parsed to RasterBand.warp (and ultimately gdal.WarpOptions)

        Returns
        -------
        rb: RasterBand
            The warped dataset loaded as RasterBand
        """
        xres, yres = None, None
        if not ref_rb.gt[1] == self.gt[1]:
            xres = ref_rb.gt[1]
        if not ref_rb.gt[5] == self.gt[5]:
            yres = ref_rb.gt[5]
        return self.warp(file_name,
                         bbox=ref_rb.extent,
                         xres=xres,
                         yres=yres,
                         dst_srs=ref_rb.srs,
                         read_masked_scaled=read_masked_scaled,
                         **kwargs)


def iter_slices(shape, chunksize_x=5000, chunksize_y=5000, overlap=0):
    """
    iterator for the slices over this RasterBand in the given chunksizes

    Parameters
    ----------
    shape: tuple
        Two element tuple (ysize, xsize)
    chunksize_x: int, optional
        Use chunks of this many elements in x direction
    chunksize_y: int, optional
        Use chunks of this many elements in x direction
    overlap: int, optional
        Overlap slices by this many datapoints (where possible)

    Yields
    ------
    x_slice: slice
    y_slice: slice
    """

    for x_start in range(0, shape[1], chunksize_x):
        x_end = x_start + chunksize_x
        x_slice = slice(max(x_start - overlap, 0),
                        min(x_end + overlap, shape[1]),
                        None)

        for y_start in range(0, shape[0], chunksize_y):
            y_end = y_start + chunksize_y
            y_slice = slice(max(y_start - overlap, 0),
                            min(y_end + overlap, shape[0]),
                            None)

            yield x_slice, y_slice


def geotiff_to_nc(geotiff_fname, nc_fname, dt_ref,
                  bandname_mapper=None,
                  global_metadata=None,
                  variable_metadata=None):
    """
    Convert geotiff into netCDF4 file.

    Parameters
    ----------
    geotiff_fname: string
        filename of geotiff file
    nc_fname: string
        filename of netcdf4 target file
    dt_ref: datetime.datetime
        time to use for time dimension
    bandname_mapper: dict, optional
        give a name for each band in the geotiff
        If not given the bands will just be numbered 'Band 1', 'Band 2' ...
        keys: int from 1 to number of band
        values: strings for variable names
    global_metadata: dict
        Write additional metadata to netcdf file
    """
    rb = RasterBand(geotiff_fname, check_file_existence=True)

    bands = range(1, rb.ds.RasterCount + 1)

    if bandname_mapper is None:
        bandname_mapper = {band_nr: 'Band {}'.format(str(band_nr))
                           for band_nr in bands}

    for band_nr in bands:
        rb = RasterBand(geotiff_fname, bandnr=band_nr)

        rb.to_netcdf(nc_fname, dt_ref,
                     bandname_mapper[band_nr],
                     global_metadata=global_metadata)

# EOF
