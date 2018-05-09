import glob

import numpy as np
import scipy.io

from ltpFR_final.pools import get_bins


# REPLICATION OF LOHNAS PAPER WITH ltpFR2 for every 7th session



def get_ltpFR2(files_ltpFR2):
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

    word_dict = get_bins()

    all_prob_first = np.zeros((1,10))
    all_prob_middle = np.zeros((1, 10))
    all_prob_last = np.zeros((1, 10))
    #files = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    #test_file_path = '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat'
    #word_freq_path = "/Users/adaaka/Desktop/wpWASfreq.mat"

    n_first = 0
    n_middle = 0
    n_last = 0
    participant_probs_ltpFR2_first = []
    participant_probs_ltpFR2_middle = []
    participant_probs_ltpFR2_last = []
    for f in files_ltpFR2:
        print(f)
        test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue
      #  word_freq_file = scipy.io.loadmat(word_freq_path,squeeze_me=True, struct_as_record=False)


    # Get the array of items that were presented
    # Each row = 1 list of items
        session_mat = test_mat_file['data'].session
        print(len(session_mat))
        print (np.bincount(session_mat))
        if len(session_mat) != 576:   # and not (session_mat.max() == 7 and len(session_mat) == 168); ADD TO LTPFR
            print('Skipping because participant did not finish...')
            continue

        print(session_mat)
        pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
        n_first +=1


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



    print(n_first, "THIS MANY PEOPLE PASSED OUR FILTER")
    #print()
    #print(pres_bins)
    #print()
    #print(rec_bins)
        #
            if session_mat[i] < 8:
                pres_counts_first = count_presentations(pres_bins)
                rec_counts_first = count_recalls(rec_mat, rec_bins)
                probi_first = (prob(pres_counts_first, rec_counts_first))
                participant_probs_ltpFR2_first.append(probi_first)
            all_prob_first += probi_first
            n_first += 1
            return all_prob_first, n_first

            else session_mat[i] > 7 and session_mat[i] < 18:
                pres_counts_middle = count_presentations(pres_bins)
                rec_counts_first = count_recalls(rec_mat, rec_bins)
                probi_middle = (prob(pres_counts_middle, rec_counts_middle))
                participant_probs_ltpFR2_middle.append(probi_middle)
            all_prob_middle += probi_middle
            n_middle += 1

            return all_prob_middle, n_middle

            else session_mat[i] > 17 and session_mat[i] < 25:












            pres_counts_last = count_presentations(pres_bins)
            rec_counts_last = count_recalls(rec_mat, rec_bins)
            probi_last = (prob(pres_counts_last, rec_counts_last))
                participant_probs_ltpFR2_last.append(probi_last)
            all_prob_last += probi_last
            n_last += 1
            return all_prob_last, n_last


        all_prob_first /= n_first
        all_prob_middle /= n_middle
        all_prob_last /= n_last
        print(all_prob_first, all_prob_middle, all_prob_last)


if __name__ == "__main__":
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    all_prob_first, all_prob_middle, all_prob_last = get_ltpFR2(files_ltpFR2)
#
#
#
# plt.suptitle('Word Frequency', fontsize=20)
# plt.xlabel('Frequency', fontsize=18)
# plt.ylabel('Recall Probability', fontsize=18)
# f_first_seven = plt.errorbar(M, all_prob_first,  label='Sessions 1 - 7')
# f_middle = plt.errorbar(M, all_prob_middle,  label= 'Session 8 - 17')
# f_last_seven = plt.errorbar(M, all_prob_last,  label= 'Session 17 - 24')
# plt.xscale('log')
# plt.legend(loc='upper right', prop={'size':12 }).draggable()
# plt.show()
