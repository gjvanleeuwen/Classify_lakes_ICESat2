import numpy as np
from sklearn.utils import resample

def resample_class(training_array, label, label_int_list,
                   percentage=None, exact=None, replace=False):
    """
    Function to up or downsample a training array in order to even the sample size for all labels

    Parameters
    ----------
    arr_train : np.array
        array with columns as bands and rows as observations linked to class in label_list
    label : int
        class label ought to be resampled, has to be included in label_list
    label_list : list
        list of all labels, same row dimension as arr_train holding all class labels for each row
    percentage : int, optional
        int indicating percentage the samples should be changed with (above 100 for increase)
    exact : int, optional
        int indicating n_samples for resample
    replace : bool, optional
        if True will upsample, if false will downsample this specific class

    Returns
    -------
    downsampled_train_arr : nd.array
        array same as input but 1 class up or downsampled
    downsampled_label_list : list
        list with all training labels same as input but 1 class up or downsampled

    """
    label_arr = np.array(label_int_list)

    arr_train_split, label_arr_split = training_array[label_arr == label], label_arr[label_arr == label]
    arr_train_leftover, label_arr_leftover = training_array[label_arr != label], label_arr[label_arr != label]
    del training_array

    label_count = label_int_list.count(label)
    if not percentage is None:
        exact = int((label_count/100)*percentage)
    else:
        exact=exact

    arr_train_resampled = resample(arr_train_split, replace=replace, n_samples=exact, random_state=123)
    label_arr_resampled = resample(label_arr_split,replace=replace, n_samples=exact, random_state=123)

    downsampled_train_arr = np.concatenate((arr_train_leftover, arr_train_resampled), axis=0)
    downsampled_label_arr = np.concatenate((label_arr_leftover, label_arr_resampled), axis=0)

    return downsampled_train_arr, list(downsampled_label_arr)


def filter_unique_rows_training_data(arr_train, label_list):
    """
    function will delete all duplicate rows in a 2d array and delete the same rows in the label_list.

    Due to sens_date or month being a parameter almost no valuable data is lost but only the copy rows as a result
    of using mosaics, which do not always have new data for every roi every day.

    Parameters
    ----------
    arr_train : nd.array
        array with columns as bands and rows as observations linked to class in label_list
    label_list : list
        list of all labels, same row dimension as arr_train holding all class labels for each row

    Returns
    -------
    arr_train : nd.array
        array with columns as bands and rows as observations linked to class in label_list, without duplicate rows
    label_list : list
        list of all labels, same row dimension as arr_train holding all class labels for each row, same rows deleted
    """

    b = np.ascontiguousarray(arr_train).view(np.dtype((np.void, arr_train.dtype.itemsize * arr_train.shape[1])))
    _, idx = np.unique(b, return_index=True)
    arr_train = arr_train[idx, :]
    label_list = list(np.array(label_list)[idx])