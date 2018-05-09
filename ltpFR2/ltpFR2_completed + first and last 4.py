import glob
from ltpFR2.Get_Bins import get_bins
import numpy as np
import matplotlib.pyplot as plt
from ltpFR2.WordFreq_Excel import M
import scipy.io
import scipy.stats as stats


# ltpFR2 with participants who finished first 4 and last 4 sessions

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

    def prob(pres_counts, rec_counts):
        probabilities = rec_counts / pres_counts
        return probabilities

    #if __name__ == "__main__":
    word_dict = get_bins()

    all_prob_first = np.zeros((1,10))
    all_prob_second = np.zeros((1, 10))
    # all_prob_third = np.zeros((1, 10))

    word_freq_path = "/Users/adaaka/Desktop/wpWASfreq.mat"
    n = 0
    participant_probs_ltpFR2_first = []
    participant_probs_ltpFR2_second = []
    # participant_probs_ltpFR2_third = []
    for f in files_ltpFR2:
        print(f)
        test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue
        if f == '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP340.mat':
            continue
        session_mat = test_mat_file['data'].session
        # print(session_mat)
        #print(np.bincount(session_mat))
        if session_mat.max() < 24:
            print('Skipping because participant did not finish...')
            continue


    # Get the array of items that were presented
    # Each row = 1 list of items

        pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')


        #print(pres_mat)

    # Testing out indexing of the presented item numbers file
    # print()
    # print("Testing indexing:")
    # print(pres_mat[0])
    # print()

    #    #print("LENGTH IS:", len(test_mat_file['data'].pres_itemnos))

    # Get the array of items that were recalled from each list presented
        rec_mat = test_mat_file['data'].pres.recalled
        #print(rec_mat)
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

        inds_first = np.where(np.logical_and(session_mat > 0, session_mat < 5))[0]
        inds_second = np.where(np.logical_and(session_mat > 19, session_mat< 24))[0]

        #print(inds_first)
        #print(inds_second)

        pres_counts_first = count_presentations(pres_bins[inds_first])
        pres_counts_second = count_presentations(pres_bins[inds_second])

        #print("Pres counts first", pres_counts_first)
        print("Pres counts second", pres_counts_second)

        rec_counts_first = count_recalls(rec_mat[inds_first], rec_bins[inds_first])
        rec_counts_second = count_recalls(rec_mat[inds_second], rec_bins[inds_second])

        #print("Rec counts first", rec_counts_first)
        print("Rec counts second", rec_counts_second)

        probi_first = (prob(pres_counts_first, rec_counts_first))
        probi_second = (prob(pres_counts_second, rec_counts_second))

        #print("Probi first", probi_first)
        print("Probi second", probi_second)

        participant_probs_ltpFR2_first.append(probi_first)
        participant_probs_ltpFR2_second.append(probi_second)

        all_prob_first += probi_first
        all_prob_second += probi_second

        # print('All prob first', all_prob_first)
        print('All prob second', all_prob_second)

        n += 1




    #stats.f_oneway(all_prob_first, all_prob_second, all_prob_third)
    print(n)
    all_prob_first /= n
    all_prob_second /= n



    # matrix of individual participant probabilities across bins
    participant_probs_ltpFR2_first = np.array(participant_probs_ltpFR2_first)
    participant_probs_ltpFR2_second = np.array(participant_probs_ltpFR2_second)


    # standard error of the mean, for each bin, across participants
    sem_ltpFR2_first = scipy.stats.sem(participant_probs_ltpFR2_first, axis=0)
    sem_ltpFR2_second = scipy.stats.sem(participant_probs_ltpFR2_second, axis=0)


    return all_prob_first, all_prob_second, sem_ltpFR2_first, sem_ltpFR2_second

if __name__ == "__main__":
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')

    all_prob_first, all_prob_second, sem_ltpFR2_first, sem_ltpFR2_second = get_ltpFR2(files_ltpFR2)
    print("All_Prob_first: ", all_prob_first)
    print("All_Prob_second: ", all_prob_second)
    print("SEM first:", sem_ltpFR2_first)
    print("SEM second:", sem_ltpFR2_second)




plt.suptitle('Word Frequency (ltpFR2)', fontsize=20)
plt.xlabel('Frequency', fontsize=18)
plt.ylabel('Recall Probability', fontsize=16)
plt.errorbar(M, all_prob_first[0], yerr = sem_ltpFR2_first[0], label = 'ltpFR2 Sessions 1 - 4')
plt.errorbar(M, all_prob_second[0], yerr = sem_ltpFR2_second[0], label='ltpFR2 Sessions 20 - 23')
plt.legend(loc='upper right', prop={'size': 14}).draggable()
plt.xscale('log')
plt.show()
