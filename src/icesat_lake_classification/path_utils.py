import os

from trollsift import Parser

import icesat_lake_classification.utils as utl


class path_parser(object):
    """
    Holds all trollsift formats for all paths and fn's in ICESat_lake_classification
    """

    def __init__(self,
                 ATL03_fn = False,
                 sentinel2_l2a_file = False
                 ):
        """
        Parameters
        ----------
        ATL03_fn : bool, indicating whether or not to use the parser for this filetype
        """
        self.ATL03_fn=ATL03_fn
        self.s2_l2a = sentinel2_l2a_file


    def ATL03_fn_parser(self, pattern=False):
        template_parts = ["ATL{product:2d}", "{datetime:%Y%m%d%H%M%S}", "{rgt:4d}{cycle:2d}{orbitsegment:2d}", "{version:3d}", "{revision:2d}.h5"]
        template_joined = '_'.join(template_parts)

        if pattern:
            return template_joined

        p = Parser(template_joined)
        return p

    def sentinel2_l2a_fileformat_parser(self):
        template_parts_fname = ['T{tile_number:5s}',
                                '{datetime:%Y%m%dT%H%M%S}',
                                '{band}',
                                '{resolution:3s}.{ext:3s}']
        fname_template = '_'.join(template_parts_fname)

        template_parts_directory = ['{base_dir}','l2a','{tile:5s}','{date}', '{l2a_product_id}',
                                    'GRANULE', '{l2a_granule_id}',
                                    'IMG_DATA', 'R{resolution:3s}']
        directory_template = '/'.join(template_parts_directory)
        complete_template = os.path.join(directory_template, fname_template)
        p = Parser(complete_template)
        return p



    def path_to_dict(self,path):
        '''
        Parse the filename
        Parameters
        ----------
        path : str
            string to parse using the template
        Returns
        -------
        parsed_data: dict
            Dictionary with metadata
        '''

        if self.ATL03_fn:
            print(path)
            p = self.ATL03_fn_parser()
            return p.parse(path)

    def path_from_dict(self, path_dict, **kwargs):
        '''
        Compose filename from data dictionary
        Parameters
        ----------
        path_dict: dict
            dictionary for filling fields in the template
        **kwargs:
            replace any part of the data using keywords
        Returns
        -------
        composed_template: string
            Filled template
        '''
        if self.ATL03_fn:
            data = path_dict.copy()
            for kw in kwargs:
                data[kw] = kwargs[kw]
            p = self.ATL03_fn_parser()
            return p.compose(data)

    def get_pattern(self):
        """
        retrieve format for certain category

        Returns
        -------
        format: str,
            format string for the first true category in the parser init
        """
        if self.ATL03_fn:
            return self.ATL03_fn_parser(pattern=True)

def parse_ATL03_fullpath(datapath):
    return path_parser(ATL03_fn=True).path_to_dict(os.path.basename(datapath))


def check_existence(file_path, overwrite=False):
    """
    function to return true or false for processing a certain file based on existence and overwrite parameter
    False means process, True means skip/file exists

    Parameters
    ----------
    file_path : str
        path + fn to check
    overwrite : bool, optional
        if True always returns False (overwrite path)
        else check file existence and return based on existence
    Returns
    -------
    bool, indicating process or not

    """
    if overwrite == False:
        if os.path.exists(file_path):
            return True
        else:
            return False

    if overwrite:
        return False


def get_files_from_folder(folder_name, file_mask, recursive=False):
    """
    Retrieves files from folder

    Parameters
    ----------
    folder_name: string
        Directory where to glob
    file_mask: string
        Text pattern used for matching
    recursive: bool
        If True, searches 0 or more subdirectories

    Returns
    -------
    sorted_filelist: list
        List containing all matches

    """
    import glob

    if not folder_name or not file_mask:
        utl.log('no folder_name or file_mask in get_files_from_folder', log_level='ERROR')
        return None
    if recursive:
        return sorted(glob.glob(os.path.join(folder_name, '**', file_mask), recursive=True))
    else:
        return sorted(glob.glob(os.path.join(folder_name, file_mask)))


def filter_files_from_list(file_list, file_mask):
    """
    Filter files from a list using a glob string.
    This is useful when a list of files exists that is not easily
    accessible via the filesystem.
    Either because of a lot of subfolders or data from cloud storage

    Paramters
    ---------
    file_list: list
        list of filepaths
    file_mask: string
        glob mask to filter by
    """
    import pathlib2 as pathlib
    matched_files = []
    for fname in file_list:
        pp = pathlib.PurePath(fname)
        if pp.match(file_mask):
            matched_files.append(fname)
    return matched_files

def get_curdir():
    return os.path.abspath(os.path.curdir)

def get_filename(path):
    return os.path.basename(path)

def get_dirname(path):
    return os.path.dirname(path)

def get_filname_without_extension(path):
    return os.path.basename(path).split('.', 1)[0]

def get_file_extension(path):
    return os.path.basename(path).split('.', 1)[1]


def get_sorted_s2_filelist(inputdir, band_list=None, extension='jp2', recursive=False):
    """
    Retrieves the raw S2 bands based on input directory sorted by frequency

    Parameters
    ----------
    inputdir: string
        Locations of the inputfiles
    band_list: list, optional
        list containing bandname extensions following sentinel naming conventions of bands to be stacked.
        When supplied, retrieves a custom selection of bands.
        Not supplied, retrieves complete l2a 20m list.
    extension : str , optional
        extension, without dot, to use in the file mask. Default is 'jp2'

    Returns
    -------
    band_stack_list: list
        List of all the bands, sorted by frequency

    """
    if not band_list:
        band_list = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                     'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

    band_stack_list = []

    for band in band_list:
        filemask = '*' + band + '*.' + extension
        band_filename = get_files_from_folder(inputdir, filemask, recursive=recursive)[0]

        band_stack_list.append(band_filename)

    return band_stack_list