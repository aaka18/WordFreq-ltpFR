import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from ltpFR_Quartiles.Quartiles_Excel import Means
from ltpFR_Quartiles.Quartiles_Pools import get_bins_quartile
import scipy.stats as stats


# ltpFR participants who finished 20 sessions
# Sort the word frequencies of ltpFR1 in 4 quartiles
# P-rec of top quartile or bottom quartile - middle two quartiles
# One number for each subject for each session
# Graph of Recall probability as a function early (first 4) or late (last 4) sessions.
# Each point is average of

def get_ltpFR(files_ltpFR):


    def count_presentations(a):
    # Count the number of items presented in each bin
        counts_pres = []
        for k in range(0, 4):
            counts_pres.append(len(np.where((a == k))[0]))
        #print(np.array([counts_pres]))
        return np.array([counts_pres])

    def count_recalls(recalled, a):
        # Count the number of recalled items recalled in each bin
        counts_rec = []
        for k in range(0, 4):
            counts_rec.append(len(np.where((a == k) & (recalled == 1))[0]))
        #print(np.array([counts_rec]))
        return np.array([counts_rec])


    def prob(pres_counts, rec_counts):

        # Find the probability of recall by dividing counts of presented by counts recalled
        probabilities = rec_counts / pres_counts
        #print( "Probabilites", probabilities)
        return probabilities


    # Running the get_bins() function from another Python file to get the word dict with word frequencies
    word_dict = get_bins_quartile()

    # Creating an empty array to put in probability of recall in each of the four bins
    all_prob_first = np.zeros((1,4))
    all_prob_second = np.zeros((1,4))
    all_single_numbers = []
    # Participant counter

    participant_probs_ltpFR_first = []
    participant_probs_ltpFR_second = []
    # for each of the participants
    subj_single_numbers = []
    n = 0
    for f in files_ltpFR:
        # Print their number, skip if there's no data or they haven't completed all 24 sessions
        # Get the matrix of presented and recalled items
        print(f)
        test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue
        session_mat = test_mat_file['data'].session
        if session_mat.max() != 20: # Anything to else to add to make sure people finished?
            print('Skipping because participant did not finish...')
            continue
        # for each session of each participant
        this_subj_probi = []


        pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
        rec_mat = test_mat_file['data'].pres.recalled

        pres_bins = np.copy(pres_mat)
        rec_bins = np.copy(rec_mat)

        # If the word id is not in the word dict, change their bin number to -1
        for i in range(pres_bins.shape[0]):
            for j in range(pres_bins.shape[1]):
                if pres_bins[i][j] not in word_dict:
                    pres_bins[i][j] = -1

        rec_bins = pres_bins

        # Change the word ids to bin numbers
        for id, bin in word_dict.items():
            pres_bins[pres_mat==id] = bin

        inds_firs = np.where(session_mat < 5)[0]
        inds_second = np.where(np.logical_and(session_mat > 16, session_mat < 21))[0]

        pres_counts_first = count_presentations(pres_bins[inds_firs])
        pres_counts_second = count_presentations(pres_bins[inds_second])
        #print("pres counts first", pres_counts_first)
        #print("pres counts second", pres_counts_second)

        rec_counts_first = count_recalls(rec_mat[inds_firs], rec_bins[inds_firs])
        rec_counts_second = count_recalls(rec_mat[inds_second], rec_bins[inds_second])
        #print("rec counts first", rec_counts_first)
        #print("rec counts second", rec_counts_second)

        probi_first = (prob(pres_counts_first, rec_counts_first))
        probi_second = (prob(pres_counts_second, rec_counts_second))
        #print("probi first", probi_first)
        #print("probi second", probi_second)

        participant_probs_ltpFR_first.append(probi_first)
        participant_probs_ltpFR_second.append(probi_second)

        all_prob_first += probi_first
        all_prob_second += probi_second
        #print("all prob first", all_prob_first)
        #print("all prob second", all_prob_second)
        n += 1


    all_prob_first /= n
    all_prob_second /= n
    #print("all prob first", all_prob_first)
    #print("all prob second", all_prob_second)

    # matrix of individual participant probabilities across bins
    participant_probs_ltpFR_first = np.array(participant_probs_ltpFR_first)
    participant_probs_ltpFR_second = np.array(participant_probs_ltpFR_second)

    # standard error of the mean, for each bin, across participants
    sem_ltpFR_first = scipy.stats.sem(participant_probs_ltpFR_first, axis=0)
    sem_ltpFR_second = scipy.stats.sem(participant_probs_ltpFR_second, axis=0)
    print("ALL PROB FIRST", all_prob_first)
    print("ALL PROB SECOND", all_prob_second)
    print("SEM FIRST", sem_ltpFR_first)
    print("SEM SECOND", sem_ltpFR_second)
    return all_prob_first, all_prob_second, sem_ltpFR_first, sem_ltpFR_second



if __name__ == "__main__":
    files_ltpFR = ['/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP063.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP064.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP065.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP066.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP067.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP068.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP069.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP070.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP073.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP074.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP075.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP076.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP077.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP078.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP079.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP080.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP081.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP082.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP083.mat',
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
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP097.mat',
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
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP109.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP110.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP111.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP112.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP113.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP114.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP115.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP116.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP117.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP118.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP119.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP120.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP121.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP122.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP123.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP124.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP125.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP126.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP127.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP128.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP129.mat',
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
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP152.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP153.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP155.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP158.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP159.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP161.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP166.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP167.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP168.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP172.mat',
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
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP214.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP215.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP221.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP224.mat',
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
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP248.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP249.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP250.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP251.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP252.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP253.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP254.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP256.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP258.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP259.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP260.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP261.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP263.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP264.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP265.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP267.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP268.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP269.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP270.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP271.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP272.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP273.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP274.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP275.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP276.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP277.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP278.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP279.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP280.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP281.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP282.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP283.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP284.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP285.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP286.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP287.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP288.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP289.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP290.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP291.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP292.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP293.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP294.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP331.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP332.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP333.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP334.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP335.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP336.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP337.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP338.mat',
    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP339.mat']
    all_prob_first, all_prob_second, sem_ltpFR_first, sem_ltpFR_second =  get_ltpFR(files_ltpFR)

plt.suptitle('Word Frequency (ltpFR)', fontsize=20)
plt.xlabel('Frequency', fontsize=18)
plt.ylabel('Recall Probability', fontsize=16)

#plt.plot(Means, all_prob_first[0],label='ltpFR Sessions 1 - 4')
#plt.plot(Means, all_prob_second[0],label='ltpFR Sessions 17 - 20')
plt.errorbar(Means, all_prob_first[0], yerr=sem_ltpFR_first[0], label='ltpFR Sessions 1 - 4')
plt.errorbar(Means, all_prob_second[0], yerr=sem_ltpFR_second[0], label='ltpFR Sessions 17 - 20')
plt.legend(loc='upper right', prop={'size': 14}).draggable()
plt.xscale('log')
plt.show()

