import scipy.io
import glob
from pools_three import get_bins
import numpy as np
import matplotlib.pyplot as plt
from Excel_three import M

def count_presentations(a):
    counts_pres = []
    for k in range(0, 10):
        counts_pres.append(len(np.where((a == k))[0]))
    return np.array([counts_pres])

def count_recalls(recalled, a):
    counts_rec = []
    for k in range(0, 10):
        counts_rec.append(len(np.where((a == k) & (recalled == 1))[0]))
    return np.array([counts_rec])

# How to get items recalled once in each list & not get the repetitions or intrusions within session so the shapes would match?

def prob(pres_counts, rec_counts):
    probabilities = rec_counts / pres_counts
    return probabilities

if __name__ == "__main__":
    word_dict = get_bins()

    all_prob = np.zeros((1,10))


    files = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    word_freq_path = "/Users/adaaka/Desktop/wpWASfreq.mat"

    all_data = np.zeros((len(files), 10))

    plt.suptitle('Word Frequency', fontsize=20)
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Recall Probability', fontsize=16)
    n = 0
    for f in files:
        print(f)
        test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue
      #  word_freq_file = scipy.io.loadmat(word_freq_path,squeeze_me=True, struct_as_record=False)


    # Get the array of items that were presented
    # Each row = 1 list of items

        pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')


        #print(test_mat_file['data'].pres_itemnos)

    # Testing out indexing of the presented item numbers file
    # print()
    # print("Testing indexing:")
    # print(pres_mat[0])
    # print()

    #    print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

    # Get the array of items that were recalled from each list presented
        rec_mat = test_mat_file['data'].pres.recalled
     #   print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

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

        pres_counts = count_presentations(pres_bins)
        rec_counts = count_recalls(rec_mat, rec_bins)
        probi = (prob(pres_counts, rec_counts))

        all_prob += probi
        all_data[n] += probi.flatten()
        n += 1


    all_prob /= n
    all_data = all_data[:n]
    plt.xscale('log')
    difference = all_data - all_prob
    plt.plot(M, difference.T)
    plt.show()

    # all_prob /= n
    # all_data = all_data[:n]
    # plt.xscale('log')
    # difference = all_data - all_prob.mean()
    # plt.plot(M, difference.T)
    # plt.show()