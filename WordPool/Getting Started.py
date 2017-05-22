import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP123.mat')
test_file_path = '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat'
word_freq_path = "/Users/adaaka/Desktop/Desktop/Frequency_norms.mat"

for f in files:
    print(f)
    test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
    word_freq_file = scipy.io.loadmat(word_freq_path,squeeze_me=True, struct_as_record=False)

    pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
    pres_task = test_mat_file['data'].pres.task.astype('int16')

    print(test_mat_file['data'].pres_itemnos)

    rec_mat = test_mat_file['data'].pres.recalled