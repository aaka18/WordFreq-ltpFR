import scipy.io
import glob
from ltpFR2.Get_Bins import get_bins
import numpy as np
import matplotlib.pyplot as plt
from ltpFR2.WordFreq_Excel import M

# REPLICATION OF LOHNAS PAPER WITH ltpFR2

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


def prob(pres_counts, rec_counts):
    probabilities = rec_counts / pres_counts
    return probabilities

if __name__ == "__main__":
    word_dict = get_bins()

    all_prob = np.zeros((1,10))

    files = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    #test_file_path = '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat'
    word_freq_path = "/Users/adaaka/Desktop/wpWASfreq.mat"
    n = 0
    for f in files:
        print(f)
        test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue
        session_mat = test_mat_file['data'].session
        # print(len(session_mat))
        # print(np.bincount(session_mat))
        if len(session_mat) != 576:  # and not (session_mat.max() == 7 and len(session_mat) == 168); ADD TO LTPFR
            print('Skipping because participant did not finish...')
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
        n += 1

    all_prob /= n

    plt.suptitle('Word Frequency', fontsize=20)
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Recall Probability', fontsize=16)
    plt.plot(M, all_prob[0], label = 'ltpFR2')
    plt.xscale('log')
    plt.ylim((.40, .65))
    plt.legend(loc='upper right', prop={'size': 14}).draggable()
    plt.show()
