import glob

import numpy as np
import scipy.io

from ltpFR_Quartiles.Quartiles_Pools import get_bins_quartile


# REPLICATION OF LOHNAS PAPER WITH ltpFR2 with participants who finished sessions
# Sort the word frequencies of ltpFR1 in 4 quartiles
# P-rec of top quartile + bottom quartile - middle two quartiles
# One number for each subject for each session
# Graph of Recall probability as a function of session # so 23 points
# Each point is average of

def get_ltpFR2(files_ltpFR2):


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

    def quarts(probabilities):

        # P-rec of bottom quartile - middle two quartiles (averaged)
        single_number = probabilities[0][0] - ((probabilities[0][1] + probabilities[0][2])/2)
        #print("Single number", single_number)
        return single_number


    # Running the get_bins() function from another Python file to get the word dict with word frequencies
    word_dict = get_bins_quartile()

    # Creating an empty array to put in probability of recall in each of the four bins
    # all_prob = np.zeros((1,4))
    all_single_numbers = []
    # Participant counter
    n = 0

    participant_probs_ltpFR2 = []

    # for each of the participants
    subj_single_numbers = []
    for f in files_ltpFR2:
        # Print their number, skip if there's no data or they haven't completed all 24 sessions
        # Get the matrix of presented and recalled items
        print(f)
        test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue
        session_mat = test_mat_file['data'].session
        if len(session_mat) != 576:  # and not (session_mat.max() == 7 and len(session_mat) == 168); ADD TO LTPFR
            print('Skipping because participant did not finish...')
            continue

        # for each session of each participant
        this_subj_single_numbers = []
        for s in range(1, 24):
            session_num = session_mat[np.where(session_mat == s)[0]]
            #print(session_num)
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

            # Count the number of word presentations, recalls, and find probability of recall for each bin for each participant
            pres_counts = count_presentations(pres_bins[session_num])
            # print(pres_counts)
            rec_counts = count_recalls(rec_mat[session_num], rec_bins[session_num])
            # print(rec_counts)
            probi = (prob(pres_counts, rec_counts))
            # print(probi)
            #participant_probs_ltpFR2.append(probi)
            #all_prob += probi
            this_subj_single_numbers.append(quarts(probi))
            # print(this_subj_single_numbers)

        subj_single_numbers.append(this_subj_single_numbers)
        n += 1

        # Divide the total p_rec in each bin by the total number of participants to find the single p_rec value of each bin
        # all_single_numbers /= n

    #print("\n This subject single numbers", this_subj_single_numbers)

    all_subjects = np.asmatrix(subj_single_numbers)
    # print("Subject single numbers as matrix", all_subjects)

    all_subjects_nanmean = np.nanmean(all_subjects, axis = 0)
    print("All subjects nanmean:", all_subjects_nanmean)

    return all_subjects_nanmean



if __name__ == "__main__":
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP123.mat')
    all_subjects_nanmean =  get_ltpFR2(files_ltpFR2)



# plt.suptitle('Word Frequency', fontsize=20)
# plt.xlabel('Sessions', fontsize=18)
# plt.ylabel('Recall Probability', fontsize=16)

# The matrix that is printed out should be taken to another plot file for being plotted