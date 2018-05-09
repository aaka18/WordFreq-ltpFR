import numpy as np
import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import json

def measures_by_list(files_ltpFR2):
    """This is the loop to find all participants' session number and list number per list"""
    all_mean_session_no_participants = []
    all_mean_trial_no_participants = []

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
            session_mat = session_mat[np.where(session_mat != 23)]
            rec_mat = np.array(data['recalled'])
            rec_mat = rec_mat[np.where(session_mat != 23)]

            # Add each participants' values to correct all_mean lists
            all_mean_session_no_participants.append(session_mat )

            all_mean_trial_no_participants.append([(n % 24) + 1 for n in range(552)])
                # all_mean_trial_no_participants.append(trial_mat)
            # print(all_mean_session_no_participants)
            # print(np.shape(all_mean_session_no_participants))
            # print(all_mean_trial_no_participants)
            # print(np.shape(all_mean_trial_no_participants))
    return all_mean_session_no_participants, all_mean_trial_no_participants


if __name__ == "__main__":
    # Get the files
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
    all_mean_session_no_participants, all_mean_trial_no_participants = measures_by_list(files_ltpFR2)
    print("ALl mean session no participants:", all_mean_session_no_participants)
    print("Shape of ", np.shape(np.array(all_mean_session_no_participants)))
    # print(" ")
    print("All mean trial no participants:", all_mean_trial_no_participants)
    print("Shape of", np.shape(np.array(all_mean_trial_no_participants)))
    print(" ")
    # print("All mean word len participants:", all_mean_wordlen_participants)
    # print("Shape of all_mean_wordlen", np.shape(np.array(all_mean_wordlen_participants)))
