import glob
from ltpFR2.GetBinsNoCon import get_bins
import numpy as np
import matplotlib.pyplot as plt
from ltpFR2.ltpFR2_noconcorimagExcel import M
import scipy.io
import scipy.stats as stats


# REPLICATION OF LOHNAS PAPER WITH ltpFR2 with participants who finished sessions

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
    all_prob_third = np.zeros((1, 10))

    #files = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    #test_file_path = '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat'
    word_freq_path = "/Users/adaaka/Desktop/wpWASfreq.mat"
    n = 0
    participant_probs_ltpFR2_first = []
    participant_probs_ltpFR2_second = []
    participant_probs_ltpFR2_third = []
    for f in files_ltpFR2:
        print(f)
        test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue
        session_mat = test_mat_file['data'].session
        #print(len(session_mat))
        #print(np.bincount(session_mat))
        if len(session_mat) < 576:  # and not (session_mat.max() == 7 and len(session_mat) == 168); ADD TO LTPFR
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
        pres_mat = pres_mat[np.where(session_mat != 24)]
        rec_mat = rec_mat[np.where(session_mat != 24)]

        pres_bins = np.copy(pres_mat)
        rec_bins = np.copy(rec_mat)

        for i in range(pres_bins.shape[0]):
            for j in range(pres_bins.shape[1]):
                if pres_bins[i][j] not in word_dict:
                    pres_bins[i][j] = -1


        for id, bin in word_dict.items():
            pres_bins[pres_mat==id+1] = bin
        rec_bins = pres_bins


            #print()
    #print(pres_bins)
    #print()
    #print(rec_bins)
        inds_firs = np.where(session_mat == 1 )[0]
        inds_second = np.where(np.logical_and(session_mat > 1, session_mat < 23))[0]
        #inds_third = np.where(np.logical_and(session_mat > 16, session_mat< 24))[0]
        inds_third = np.where(session_mat == 23 )[0]
        pres_counts_first = count_presentations(pres_bins[inds_firs])
        pres_counts_second = count_presentations(pres_bins[inds_second])
        pres_counts_third = count_presentations(pres_bins[inds_third])
        rec_counts_first = count_recalls(rec_mat[inds_firs], rec_bins[inds_firs])
        rec_counts_second = count_recalls(rec_mat[inds_second], rec_bins[inds_second])
        rec_counts_third = count_recalls(rec_mat[inds_third], rec_bins[inds_third])
        probi_first = (prob(pres_counts_first, rec_counts_first))
        probi_second = (prob(pres_counts_second, rec_counts_second))
        probi_third = (prob(pres_counts_third, rec_counts_third))
        participant_probs_ltpFR2_first.append(probi_first)
        participant_probs_ltpFR2_second.append(probi_second)
        participant_probs_ltpFR2_third.append(probi_third)
        all_prob_first += probi_first
        all_prob_second += probi_second
        all_prob_third += probi_third
        n += 1




    #stats.f_oneway(all_prob_first, all_prob_second, all_prob_third)
    all_prob_first /= n
    all_prob_second /= n
    all_prob_third /= n

    # matrix of individual participant probabilities across bins
    participant_probs_ltpFR2_first = np.array(participant_probs_ltpFR2_first)
    participant_probs_ltpFR2_second = np.array(participant_probs_ltpFR2_second)
    participant_probs_ltpFR2_third = np.array(participant_probs_ltpFR2_third)

    # standard error of the mean, for each bin, across participants
    sem_ltpFR_2_first = scipy.stats.sem(participant_probs_ltpFR2_first, axis=0)
    sem_ltpFR_2_second = scipy.stats.sem(participant_probs_ltpFR2_second, axis=0)
    sem_ltpFR_2_third = scipy.stats.sem(participant_probs_ltpFR2_third, axis=0)
    sd_ltpFR_2_first = np.std(participant_probs_ltpFR2_first, axis=0)
    sd_ltpFR_2_second = np.std(participant_probs_ltpFR2_second, axis=0)
    sd_ltpFR_2_third = np.std(participant_probs_ltpFR2_third, axis=0)

    return all_prob_first, all_prob_second, all_prob_third, participant_probs_ltpFR2_first, participant_probs_ltpFR2_second, participant_probs_ltpFR2_first, sd_ltpFR_2_first, sd_ltpFR_2_second, sd_ltpFR_2_third

if __name__ == "__main__":
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    # all_prob_first, all_prob_second, all_prob_third = get_ltpFR2(files_ltpFR2)
    all_prob_first, all_prob_second, all_prob_third, participant_probs_ltpFR2_first, participant_probs_ltpFR2_second, participant_probs_ltpFR2_third, sd_ltpFR_2_first, sd_ltpFR_2_second, sd_ltpFR_2_third = get_ltpFR2(files_ltpFR2)
    print("All_Prob_first: ", all_prob_first)
    print("All_Prob_second: ", all_prob_second)
    print("All_Prob_third", all_prob_third)
    print("Participant_probs_first", participant_probs_ltpFR2_first)
    print("Participant_probs_second", participant_probs_ltpFR2_second)
    print("Participant_probs_third", participant_probs_ltpFR2_third)
    print("SD_first", sd_ltpFR_2_first)
    print("SD_second", sd_ltpFR_2_second)
    print("SD_third", sd_ltpFR_2_third)

plt.suptitle('Word Frequency', fontsize=20)
plt.xlabel('Frequency', fontsize=18)
plt.ylabel('Recall Probability', fontsize=16)
plt.plot(M, all_prob_first[0], label = 'ltpFR2 Session 1')
plt.plot(M, all_prob_second[0], label='ltpFR2 Sessions 2 - 22 ')
plt.plot(M, all_prob_third[0], label='ltpFR2 Session 23')
plt.xscale('log')
plt.legend(loc='upper right', prop={'size': 14}).draggable()
plt.show()
