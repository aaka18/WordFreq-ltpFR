import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statistics
import math
from matplotlib import pyplot as pl
import numpy as np
from scipy import stats

#FINDING PROBABILITY OF RECALL FOR EACH WORD IN LTPFR2


def get_ltpFR2_precall(files_ltpFR2):

    """This function gets the recall probability of each word for each participant
    dimenstions should be N participants * 576 words """

    # determining whether each word is recalled or not when it is presented to the participant
    # dimenstions should be (576 words * times word was presented (23) for each participant

    def count_recalls(rec_mat, pres_mat):
        word_id_recalled_or_not_this_part = []
        # print("rec mat", rec_mat)
        for word_id in np.unique(pres_mat):
            #print("Word ID:", word_id)
            word_id_recalled_or_not_this_part.append(rec_mat[np.where((pres_mat) == word_id)])
            # print(word_id_recalled_or_not_this_part)
            # print(np.array(word_id_recalled_or_not_this_part).shape)
        #print("Shape Word id this part", np.shape(word_id_recalled_or_not_this_part))
        #print(word_id_recalled_or_not_this_part)
        return word_id_recalled_or_not_this_part

    all_participants_all_recalled = []

    # Do this for every participant
    for f in files_ltpFR2:
        if f not in ['/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP331.json']:

    # Read in data
#test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)
            with open(f, 'r') as jsfile:
                data = json.load(jsfile)
            """
            if isinstance(test_mat_file['data'], np.ndarray):
                #print('Skipping...')
                continue
            session_mat = test_mat_file['data'].session
            """
            session_mat = np.array(data['session'])
            # print(len(session_mat))
            # print(np.bincount(session_mat))
            #if len(session_mat) < 576:
            #    print('Skipping because participant did not finish...')
            #    continue
            #else:
            print(f)
            pres_mat = np.array(data['pres_nos'], dtype='int16')
            pres_mat = pres_mat[np.where(session_mat != 23)]
            rec_mat = np.array(data['recalled'])
            rec_mat = rec_mat[np.where(session_mat != 23)]
            # print("Legth of pres_mat", (pres_mat).shape)
            #print("Length of rec_mat", (rec_mat))

            # print("Pres mat", pres_mat)
            # print("Rec mat", rec_mat)
            #print(rec_mat.shape)

            # For each list of this participant's data
            this_participant_probs = []

            word_id_recalled_this_part = count_recalls(rec_mat, pres_mat)

            # Append this partic p_rec to everyone else

            all_participants_all_recalled.append(np.nanmean(word_id_recalled_this_part, axis = 1))


        #Get the average of all participants
    return all_participants_all_recalled

if __name__ == "__main__":
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
    #files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')

    all_participants_all_recalled = get_ltpFR2_precall(files_ltpFR2)
    #print(all_participants_all_recalled)
    print("Shape of all participants all recalled", np.shape(all_participants_all_recalled))

    p_rec_word = np.mean(all_participants_all_recalled, axis = 0)
    for i in p_rec_word:
        print(i)
    sem_word = stats.sem(all_participants_all_recalled, axis = 0)

    ix_sorted = np.argsort(p_rec_word)

    ci = (sem_word[ix_sorted])

    print("Shape of sem_word", np.shape(np.array(sem_word)))
    pl.plot(range(576), np.sort(p_rec_word), 'k-')
    pl.fill_between(range(576), np.sort(p_rec_word) - ci, np.sort(p_rec_word) + ci, color = 'gray')

    pl.xlabel("Word ID (Ranked)", size = 13.5)
    pl.ylabel("Recall Probability", size = 13.5)
    pl.savefig("Prec_words_N88_updated.pdf")
    plt.show()

    """ 
    print(np.array(sem_word).shape)
    #print("Average p_rec_word for all participants", p_rec_word)
    # for i in p_rec_word:
    #     print(i)
    y = []
    for i in range(len(all_participants_all_recalled[0])):
        y.append(np.array(all_participants_all_recalled)[:, i])
    print(np.array(y).shape)

    y = np.array(y)
    means = np.mean(y, axis=1)
    ind = np.argsort(means)
    y_ascending = y[ind]

    # for i in range(len(all_participants_all_recalled[0])):
    #     x = [i] * (len(all_participants_all_recalled))
    #     y = y_ascending[i]
    #     plt.scatter(x, y, color='gray', s=.8)


    #pl.plot((range(576)), np.sort(means), color='black')
    
    #pl.savefig("Prec_words_N86.pdf")

    # pl.plot(x, y, 'k-')
    # pl.fill_between(x, y - error, y + error)
    # pl.show()
    #
    """


