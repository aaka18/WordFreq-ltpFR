import numpy as np
import scipy.io


# ltpFR with participants who finished sessions
# Only data from phases 1 and 3 and analyzing data from first 4 and last 4 sessions

def get_ltpFR(files_ltpFR):
    # counting the number of times a word is presented in each bin
    def count_presentations(a, pres_task):
        counts_with_task = []
        counts_no_task = []

        for k in range(0, 4):
            counts_with_task.append(len(np.where((a == k) & (pres_task > -1))[0]))
            counts_no_task.append(len(np.where((a == k) & (pres_task == -1))[0]))
        return np.array([counts_with_task, counts_no_task])

    # counting the number of times a word is recalled in each bin
    def count_recalls(recalled, a, pres_task):
        counts_with_task = []
        counts_no_task = []

        for k in range(0, 4):
            counts_with_task.append(len(np.where((a == k) & (recalled == 1) & (pres_task > -1))[0]))
            counts_no_task.append(len(np.where((a == k) & (recalled == 1) & (pres_task == -1))[0]))
        return np.array([counts_with_task, counts_no_task])

    # Get the recall probabilities of each bin
    def prob(pres_counts, rec_counts):
        probabilities = rec_counts / pres_counts
        return probabilities

    def quartslow(probabilities):

        # P-rec of top quartile + bottom quartile - middle two quartiles
        single_number_task = probabilities[0][0] - probabilities[0][1] - probabilities[0][2]
        single_number_notask = probabilities[1][0] - probabilities[1][1] - probabilities[1][2]
        #print("Single number", single_number)
        return single_number

    def quartshigh(probabilities):

        # P-rec of top quartile + bottom quartile - middle two quartiles
        single_number = probabilities[0][3] - probabilities[0][1] - probabilities[0][2]
        #print("Single number", single_number)
        return single_number

    word_dict = get_bins_ltpFR()


    word_freq_path = "/Users/adaaka/Desktop/Desktop/Frequency_norms.mat"


    #if __name__ == "__main__":
    word_dict = get_bins_ltpFR()

    all_prob_first_four = np.zeros((2,10))
    all_prob_last_four = np.zeros((2, 10))


    #files = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    #test_file_path = '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP189.mat'
    word_freq_path = "/Users/adaaka/Desktop/wpWASfreq.mat"
    n = 0
    participant_probs_ltpFR_firstfour = []
    participant_probs_ltpFR_lastfour= []

    for f in files_ltpFR:
        print(f)
        test_mat_file = scipy.io.loadmat(f,squeeze_me=True, struct_as_record=False)
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue
        session_mat = test_mat_file['data'].session
        #print(len(session_mat))
        #print(np.bincount(session_mat))
        if session_mat.max() != 20: # Anything to else to add to make sure people finished?
            print('Skipping because participant did not finish...')
            continue
      #  word_freq_file = scipy.io.loadmat(word_freq_path,squeeze_me=True, struct_as_record=False)


    # Get the array of items that were presented
    # Each row = 1 list of items
            # for each session of each participant
        this_subj_single_numbers = []
        for s in range(1, 24):
            session_num = session_mat[np.where(session_mat == s)[0]]
            print(session_num)
            session_of_interest = [1, 2, 3, 4, 17, 18, 19, 20]
                if session_num not in session_of_interest:
                    print('Skipping because it is not a session of interest...')
                    continue
                pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
                pres_task = test_mat_file['data'].pres.task.astype('int16')
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
                    pres_bins[pres_mat == id] = bin

                # Count the number of word presentations, recalls, and find probability of recall for each bin for each participant
                pres_counts = count_presentations(pres_bins[session_num], pres_task[session_num])
                print(pres_counts)
                rec_counts = count_recalls(rec_mat[session_num], rec_bins[session_num], pres_task[session_num])
                print(rec_counts)
                probi = (prob(pres_counts, rec_counts))
                print(probi)
                # participant_probs_ltpFR2.append(probi)
                # all_prob += probi
                this_subj_single_numbers.append(probi)
                # print(this_subj_single_numbers)

        subj_single_numbers.append(this_subj_single_numbers)
        n += 1

            # Divide the total p_rec in each bin by the total number of participants to find the single p_rec value of each bin
            # all_single_numbers /= n

        print("\n This subject single numbers", this_subj_single_numbers)

    all_subjects = np.asmatrix(subj_single_numbers)
    print("Subject single numbers as matrix", all_subjects)
    all_subjects =/ n

    return

        # FIGURE OUT WHERE TO PUT THESE

        # sem_all = np.array(all_subjects)
        # # standard error of the mean
        # sem= scipy.stats.sem(sem_all, axis=0)
        # sd = np.std(sem_all, axis=0)



        print("All subjects nanmean:", all_subjects_nanmean)
        print("All subjects SEM:", sem)
        print("All subjects standard deviation:", sd)
        return all_subjects_nanmean, sem, sd

    if __name__ == "__main__":
        files_ltpFR2 = glob.glob(
            '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP123.mat')
        all_subjects_nanmean, sem, sd = get_ltpFR2(files_ltpFR2)

        ##################

        inds_firs = np.where(session_mat == 1)[0]
        inds_second = np.where(np.logical_and(session_mat > 1, session_mat < 23))[0]
        #inds_third = np.where(np.logical_and(session_mat > 16, session_mat< 24))[0]
        inds_third = np.where(session_mat  == 23)[0]
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
plt.ylim((.40, .65))
plt.legend(loc='upper right', prop={'size': 14}).draggable()
plt.show()
