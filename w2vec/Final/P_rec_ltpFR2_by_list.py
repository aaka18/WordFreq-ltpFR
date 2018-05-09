import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statistics
import math
import json


#FINDING TOTAL PROBABILITY OF RECALL FOR LTPFR2 PARTICIPANTS PER LIST


def get_ltpFR2_precall(files_ltpFR2):


    #  Get the recall probabilities of each bin
    def prob(pres_counts, rec_counts):
        probabilities = rec_counts / pres_counts
        return probabilities

    pres_counts = np.zeros(552)
    rec_counts = np.zeros(552)

    all_probs = []
    words_presented = []
    words_recalled  = []
    prob            = []
    # Do this for every participant
    # for f in files_ltpFR2:
    #     # Read in data
    #     # print(f)
    #     test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)
    #     if isinstance(test_mat_file['data'], np.ndarray):
    #         # print('Skipping...')
    #         continue
    #     session_mat = test_mat_file['data'].session
    #     #list_mat    = [np.arange(1,25) for s in np.unique(session_mat)]
    #     #list_mat   = np.array([item for sublist in list_mat for item in sublist])
    #     # print(len(session_mat))
    #     # print(np.bincount(session_mat))
    #     if len(session_mat) < 576:
    #         #print('Skipping because participant did not finish...')
    #         continue
    #     else:
    #         print(f)
    #         #pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
    #         #pres_mat = pres_mat[np.where(session_mat != 24)]
    #         rec_mat = test_mat_file['data'].pres.recalled
    #         rec_mat = rec_mat[np.where(session_mat != 24)]

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

            # For each list of this participant's data
            probabilities_this_part = []
            word_id_presented_this_part = []
            word_id_recalled_this_part = []

            words_p = [len(rec_mat[l_ind,:]) for l_ind in   range(rec_mat.shape[0])]#[np.isfinite(rec_mat[l_ind,:])]
            words_r = [np.sum(rec_mat[l_ind,:]) for l_ind in range(rec_mat.shape[0])]
            probs = (np.array(words_r) / np.array(words_p)).tolist()
            #probs = [p for p in probs if np.isfinite(p)]
            # words_presented.append(words_p)
            # words_recalled.append(words_r)

            prob.append(probs)
    return prob
if __name__ == "__main__":
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
    prob = get_ltpFR2_precall(files_ltpFR2)

    all_participants = np.concatenate(prob)
    print(np.shape(all_participants))

    plt.hist(all_participants[~np.isnan(all_participants)], bins=20, color = 'gray')
    plt.xlabel("Recall Probability", size = 13)
    plt.ylabel("Number of Lists", size =13)
    plt.savefig("Hist_listprec_bins20.pdf")
    plt.show()