import glob
from Concreteness.Conc_Getbins import get_bins
import numpy as np
import matplotlib.pyplot as plt
from Meaningfulness.Meaningfulness_pool_Excel import M
import scipy.io
import scipy.stats as stats


# REPLICATION OF LOHNAS PAPER WITH ltpFR2 with participants who finished sessions, but with concreteness

def get_ltpFR2(files_ltpFR2):


    def count_presentations(a):
    # Count the number of items presented in each bin
        counts_pres = []
        for k in range(0, 10):
            counts_pres.append(len(np.where((a == k))[0]))
        return np.array([counts_pres])

    def count_recalls(recalled, a):
        # Count the number of recalled items recalled in each bin
        counts_rec = []
        for k in range(0, 10):
            counts_rec.append(len(np.where((a == k) & (recalled == 1))[0]))
        return np.array([counts_rec])


    def prob(pres_counts, rec_counts):

        # Find the probability of recall by dividing counts of presented by counts recalled
        probabilities = rec_counts / pres_counts
        return probabilities

    # Running the get_bins() function from another Python file to get the word dict with word frequencies
    word_dict = get_bins()

    # Creating an empty array to put in probability of recall in each of the ten bins
    all_prob = np.zeros((1,10))

    # Participant counter
    n = 0

    participant_probs_ltpFR2 = []

    # for each of the participants
    for f in files_ltpFR2:
        # Print their number, skip if there's no data or they haven't completed all 24 sessions
        # Get the matrix of presented and recalled items
        print(f)
        test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue
        session_mat = test_mat_file['data'].session
        if len(session_mat) < 576:  # and not (session_mat.max() == 7 and len(session_mat) == 168); ADD TO LTPFR
            print('Skipping because participant did not finish...')
            continue

        pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
        rec_mat = test_mat_file['data'].pres.recalled

        pres_mat = pres_mat[np.where(session_mat != 24)]
        rec_mat = rec_mat[np.where(session_mat != 24)]
        pres_bins = np.copy(pres_mat)
        rec_bins = np.copy(rec_mat)

        # If the word id is not in the word dict, change their bin number to -1
        for i in range(pres_bins.shape[0]):
            for j in range(pres_bins.shape[1]):
                if pres_bins[i][j] not in word_dict:
                    pres_bins[i][j] = -1



        # Change the word ids to bin numbers
        for id, bin in word_dict.items():
            pres_bins[pres_mat==id] = bin
        rec_bins = pres_bins

        # Count the number of word presentations, recalls, and find probability of recall for each bin for each participant
        pres_counts = count_presentations(pres_bins)
        rec_counts = count_recalls(rec_mat, rec_bins)
        probi = (prob(pres_counts, rec_counts))
        participant_probs_ltpFR2.append(probi)
        all_prob += probi
        n += 1

    # Divide the total p_rec in each bin by the total number of participants to find the single p_rec value of each bin
    all_prob /= n
    participant_probs_ltpFR2 = np.array(participant_probs_ltpFR2)
    # Find the SEM & SDs for the error bars of the graph
    sem_ltpFR_2 = scipy.stats.sem(participant_probs_ltpFR2, axis=0)
    sd_ltpFR_2 = np.std(participant_probs_ltpFR2, axis=0)
    return sem_ltpFR_2, all_prob, sd_ltpFR_2


if __name__ == "__main__":
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    sem_ltpFR_2, all_prob, sd_ltpFR_2 =  get_ltpFR2(files_ltpFR2)
    print(sem_ltpFR_2, all_prob, sd_ltpFR_2)


    plt.suptitle('Pool Meaningfulness', fontsize=20)
    plt.xlabel('Pool Meaningfulness', fontsize=18)
    plt.ylabel('Recall Probability', fontsize=16)
    f_all_ltpFR2 = plt.errorbar(M, all_prob[0], yerr = sem_ltpFR_2[0], label='ltpFR2')
    plt.ylim(.40, .65)
    plt.legend(loc='upper right', prop={'size': 14}).draggable()
    plt.savefig("MPoolvsprectest.pdf")
    plt.show()
