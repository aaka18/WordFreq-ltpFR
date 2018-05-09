import scipy.io
import numpy as np

def load_data(filename):
    arr = scipy.io.loadmat(filename)['data']
    dtypes = get_dtypes(arr)
    output = np.zeros([], dtype=dtypes)
    copy_values(arr, output)
    return output.view(np.recarray)

def get_dtypes(arr):
    names = arr.dtype.names
    dtypes = []
    shapes = []
    for name in names:
        sub_names = arr[name][0][0].dtype.names
        if sub_names:
            dtypes.append(get_dtypes(arr[name][0][0]))
        else:
            dtypes.append(arr[name][0][0].dtype)
        shapes.append(arr[name][0][0].squeeze().shape)
    return {'names': names, 'formats': zip(dtypes, shapes)}

def copy_values(copy_from, copy_to):
    names = copy_to.dtype.names
    for name in names:
        if isinstance(copy_to[name], (np.ndarray, np.record)) and copy_to[name].dtype.names:
            copy_values(copy_from[name][0][0], copy_to[name])
        else:
            copy_to[name] = copy_from[name][0][0].squeeze()



def test_load_data():
    data = load_data('/Volumes/rhino/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP063.mat')
    from spc.view_recarray import pprint_rec as ppr
    ppr(data)