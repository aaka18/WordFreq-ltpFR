import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from ltpFR_final.Excel import M
from ltpFR_final.pools import get_bins


# REPLICATION OF LOHNAS PAPER WITH EXACT SAME PARTICIPANTS
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

files = ['/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP065.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP066.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP067.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP069.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP070.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP073.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP074.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP075.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP076.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP077.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP079.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP081.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP082.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP084.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP085.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP086.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP087.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP088.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP089.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP090.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP091.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP092.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP093.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP094.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP095.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP096.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP098.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP099.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP100.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP101.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP102.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP103.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP104.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP105.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP106.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP107.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP108.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP110.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP111.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP112.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP113.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP114.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP115.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP117.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP118.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP119.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP120.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP122.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP123.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP124.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP125.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP127.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP128.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP130.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP131.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP132.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP133.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP134.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP135.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP136.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP137.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP138.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP139.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP140.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP141.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP142.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP143.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP144.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP145.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP146.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP147.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP148.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP149.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP150.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP151.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP153.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP155.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP159.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP166.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP168.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP174.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP184.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP185.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP186.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP187.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP188.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP190.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP191.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP192.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP193.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP194.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP195.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP196.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP197.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP198.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP199.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP200.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP201.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP202.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP207.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP209.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP210.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP211.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP212.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP215.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP227.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP228.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP229.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP230.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP231.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP232.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP233.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP234.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP235.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP236.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP237.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP238.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP239.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP240.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP241.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP242.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP243.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP244.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP245.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP246.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP247.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP249.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP250.mat',
'/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP251.mat']

#test_file_path = '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat'
word_freq_path = "/Users/adaaka/Desktop/Desktop/Frequency_norms.mat"
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
# print()
# print("Testing indexing:")
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

#print()
#print(pres_bins)
#print()
#print(rec_bins)

    pres_counts = count_presentations(pres_bins, pres_task)
    rec_counts = count_recalls(rec_mat, rec_bins, pres_task)
    probi = (prob(pres_counts, rec_counts))
    all_prob += probi
    n += 1

all_prob /= n


plt.suptitle('Word Frequency', fontsize=20)
plt.xlabel('Frequency', fontsize=18)
plt.ylabel('Recall Probability', fontsize=16)
f_task = plt.plot(M, all_prob[0])
f_no_task = plt.plot(M, all_prob[1])
plt.xscale('log')
plt.ylim((.50, .75))
plt.show()
