import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statistics
import math
#from ltpFR2.ltpFR_ltpFR2 import sem_ltpFR, sem_ltpFR2


#FINDING TOTAL PROBABILITY OF RECALL FOR LTPFR2

#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')

def get_ltpFR2_precall(files_ltpFR2):
    # a is the file for a single participant
    def count_presentations(a):
         counts_pres = []
    # how to get the presented items without a task
         return np.array([len(np.where((pres_mat < 0))[0])])


    # counting the number of occurances a word is recalled
    def count_recalls(recalled):
         counts_rec = []
         return np.array([len(np.where((recalled == 1))[0])])


    # How to get items recalled once in each list & not get the repetitions or intrusions within session so the shapes would match?

    #  Get the recall probabilities of each bin
    def prob(pres_counts, rec_counts):
        probabilities = rec_counts / pres_counts
        return probabilities

    pres_counts = np.zeros(1)
    rec_counts = np.zeros(1)

    all_probs = []

    # Do this for every participant
    for f in files_ltpFR2:
        # Read in data
        test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue
        session_mat = test_mat_file['data'].session
        # print(len(session_mat))
        # print(np.bincount(session_mat))
        if len(session_mat) != 576:
            print('Skipping because participant did not finish...')
            continue
        else:
            print(f)
            pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
            rec_mat = test_mat_file['data'].pres.recalled
            #print(pres_mat)
            #print(rec_mat)

        #print(rec_mat.shape)

        # For each list of this participant's data
        this_participant_probs = []
        n =0

        for idx, row in enumerate(rec_mat):
            # sum up the 1s in the recall matrix because that is everything they've remembered
            num_recalled = sum(row)
            # length of the row of presented items
            num_presented = len(pres_mat[idx])

            #finding each lists' probability of recall
            list_prob = num_recalled/num_presented

            # Add these list probs to this participant's probabilities list
            this_participant_probs.append(list_prob)
            n += 1

    #    get their mean for the participants overall p_rec
        #print(this_participant_probs)
        #print(type(this_participant_probs))


        mean_pt_probs = np.nanmean(this_participant_probs)

        # Append this partic p_rec to everyone else
        all_probs.append(mean_pt_probs)

    #Get the average of all participants
    p_rec_all = np.nanmean(all_probs)
    sd_prec = np.std(all_probs)
    print(sd_prec)
    print(p_rec_all)

if __name__ == "__main__":
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    sd_prec, p_rec_all = get_ltpFR2_precall(files_ltpFR2)