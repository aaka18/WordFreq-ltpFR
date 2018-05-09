import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from ltpFR_final.Excel import M
from ltpFR_final.pools import get_bins


# REPLICATION OF LOHNAS PAPER WITH ONLY THE ORIGINAL PARTICIPANTS
def count_presentations(a, pres_task):
    counts_with_task = []
    counts_no_task = []

    for k in range(0, 10):
        counts_with_task.append(len(np.where((a == k) & (pres_task > -1))[0]))
        counts_no_task.append(len(np.where((a == k) & (pres_task == -1))[0]))
    return np.array([counts_with_task, counts_no_task])

def count_recalls(recalled, a, pres_task):
    counts_with_task = []
    counts_no_task = []

    for k in range(0, 10):
        counts_with_task.append(len(np.where((a == k) & (recalled == 1) & (pres_task > -1))[0]))
        counts_no_task.append(len(np.where((a == k) & (recalled == 1) & (pres_task == -1))[0]))
    return np.array([counts_with_task, counts_no_task])

# How to get items recalled once in each list & not get the repetitions or intrusions within session so the shapes would match?

def prob(pres_counts, rec_counts):
    probabilities = rec_counts / pres_counts
    return probabilities


word_dict = get_bins()

all_prob = np.zeros((2,10))

#files = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP*.mat')
#test_file_path = '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat'
word_freq_path = "/Users/adaaka/Desktop/Desktop/Frequency_norms.mat"

files = [glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP63.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP64.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP65.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP66.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP67.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP69.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP70.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP73.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP74.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP75.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP76.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP77.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP79.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP81.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP82.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP84.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP85.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP86.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP87.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP88.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP89.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP90.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP91.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP92.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP93.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP94.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP95.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP96.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP98.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP99.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP100.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP101.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP102.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP103.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP104.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP105.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP106.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP107.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP108.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP110.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP111.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP112.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP113.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP114.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP115.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP117.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP118.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP119.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP120.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP122.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP123.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP124.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP125.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP127.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP128.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP130.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP131.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP132.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP133.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP134.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP135.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP136.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP137.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP138.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP139.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP140.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP141.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP142.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP143.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP144.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP145.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP146.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP147.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP148.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP149.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP150.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP151.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP153.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP155.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP159.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP166.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP168.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP174.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP184.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP185.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP186.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP187.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP188.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP190.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP191.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP192.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP193.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP194.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP195.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP196.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP197.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP198.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP199.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP200.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP201.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP202.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP207.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP209.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP210.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP211.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP212.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP215.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP227.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP228.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP229.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP230.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP231.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP232.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP233.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP234.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP235.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP236.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP237.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP238.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP239.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP240.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP241.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP242.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP243.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP244.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP245.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP246.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP247.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP249.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP250.mat'),
glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP251.mat')]
n = 0

for f in files:
    print(f)
    test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
    word_freq_file = scipy.io.loadmat(word_freq_path,squeeze_me=True, struct_as_record=False)


# Get the array of items that were presented
# Each row = 1 list of items

    pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
    pres_task = test_mat_file['data'].pres.task.astype('int16')

    #print(test_mat_file['data'].pres_itemnos)

# Testing out indexing of the presented item numbers file
print()
print("Testing indexing:")
# print(pres_mat[0])
# print()

# print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

# Get the array of items that were recalled from each list presented
    rec_mat = test_mat_file['data'].pres.recalled
#print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

    pres_bins = np.copy(pres_mat)
    rec_bins = np.copy(rec_mat)

    for i in range(pres_bins.shape[0]):
        for j in range(pres_bins.shape[1]):
            if pres_bins[i][j] not in word_dict:
                pres_bins[i][j] = -1

    rec_bins = pres_bins

    for id, bin in word_dict.items():
        pres_bins[pres_mat==id] = bin

# print()
# print(pres_bins)
# print()
# print(rec_bins)

    pres_counts = count_presentations(pres_bins, pres_task)
    rec_counts = count_recalls(rec_mat, rec_bins, pres_task)
    probi = (prob(pres_counts, rec_counts))
    all_prob += probi
    n += 1


all_prob /= n


plt.suptitle('Word Frequency', fontsize=20)
plt.xlabel('Frequency', fontsize=18)
plt.ylabel('Recall Probability', fontsize=16)
#f_task = plt.plot(M, all_prob[0])
#f_no_task = plt.plot(M, all_prob[1])
f_no_task_lohnas = plt.plot(M, all_prob[1], label='No Task')
f_task_lohnas = plt.plot(M, all_prob[0], label='Task')
plt.xscale('log')
plt.ylim((.50, .75))
plt.legend(loc='upper right', prop={'size':10.5})
plt.show()

