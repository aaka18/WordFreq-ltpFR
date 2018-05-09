import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from ltpFR_final.pools import get_bins


def count_numbers(a):
    counts = []
    for k in range(0, 10):
        counts.append(len(np.where(a == k)[0]))
    return np.array(counts)

def prob(pres_counts, rec_counts):
    probabilities = rec_counts / pres_counts
    return probabilities

if __name__ == "__main__":
    word_dict = get_bins()


    #files = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP*.mat')
    test_file_path = ('/Users/adaaka/Desktop/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP*.mat')
    word_freq_path = "/Users/adaaka/Desktop/Desktop/Frequency_norms.mat"

    test_mat_file = scipy.io.loadmat(test_file_path,squeeze_me=True, struct_as_record=False)
    word_freq_file = scipy.io.loadmat(word_freq_path,squeeze_me=True, struct_as_record=False)


    # Get the array of items that were presented
    # Each row = 1 list of items

    pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
    print(test_mat_file['data'].pres_itemnos)

    # Testing out indexing of the presented item numbers file
    # print()
    # print("Testing indexing:")
    # print(pres_mat[0])
    # print()

    # print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

    # Get the array of items that were recalled from each list presented
    rec_mat = test_mat_file['data'].rec_itemnos
    #print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

    pres_bins = np.copy(pres_mat)
    rec_bins = np.copy(rec_mat)

    for i in range(pres_bins.shape[0]):
        for j in range(pres_bins.shape[1]):
            if pres_bins[i][j] not in word_dict:
                pres_bins[i][j] = -1

    for i in range(rec_bins.shape[0]):
        for j in range(rec_bins.shape[1]):
            if rec_bins[i][j] not in word_dict:
                rec_bins[i][j] = -1

    for id, bin in word_dict.items():
        pres_bins[pres_mat==id] = bin
        rec_bins[rec_mat==id] = bin

    #print()
    #print(pres_bins)
    #print()
    #print(rec_bins)

    pres_counts = count_numbers(pres_bins)
    rec_counts = count_numbers(rec_bins)
    prob = (prob(pres_counts, rec_counts))

    plt.suptitle('Word Freq', fontsize=20)
    plt.xlabel('Freq', fontsize=18)
    plt.ylabel('Recall Probability', fontsize=16)
    plt.plot(prob)
    plt.show()
