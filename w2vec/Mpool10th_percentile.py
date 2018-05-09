
import glob

import numpy as np
import scipy.io
import scipy.stats


# This is the code to get how many top 10th percentile semanctic meaningfulness to all other words there are in each list

def get_m_pool(files_ltpFR2):

    # ltpFR2 word indices from the whole ltpFR word pool that are in the top 10 percentile
    larger_90 = [33, 35, 100, 109, 119, 198, 210, 304, 397, 401, 403, 429, 484, 490, 496, 570, 584, 598, 623, 684, 713, 755, 778, 782, 783, 785, 814, 834, 929, 932, 940, 953, 982, 1002, 1016, 1022, 1031, 1044, 1112, 1144, 1152, 1158, 1209, 1224, 1232, 1238, 1333, 1351, 1355, 1357, 1388, 1403, 1472, 1478, 1520, 1522, 1527, 1593]
    print("# of the ltpFR2 word pool indices that are larger than 90th percentile value:", len(larger_90))


    def top_ten_words_participant(pres_mat):
        """This method spits out the number of items presented in each list that is has top 10 percentile W2V similarity"""
        num_top_ten_words_this_participant = []

        # For each list of the presented items
        for each_list in pres_mat:
            #print(each_list)
            num_top_ten_words_this_list = 0
            # Increment num_top_ten_words_this _list if any word in the list is in the top 10
            for each_word in each_list:
                #print(each_word)
                if each_word in larger_90:
                    num_top_ten_words_this_list += 1
            num_top_ten_words_this_participant.append(num_top_ten_words_this_list)
        return num_top_ten_words_this_participant
        #print("Num top ten words this participant", num_top_ten_words_this_participant)
        #print("Length of top ten words this participant", len(num_top_ten_words_this_participant))


    all_num_top_ten = []
    m_pool = []

    for f in files_ltpFR2:
        # Read in data
        test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)

        # Skip if there is no data for a participant
        if isinstance(test_mat_file['data'], np.ndarray):
            print('Skipping...')
            continue

        # Get the session matrix for the participant
        session_mat = test_mat_file['data'].session


        # Skip if the participant did not finish
        if len(session_mat) < 576:
            print('Skipping because participant did not finish...')
            continue

        # Get the presented items matrix for the participant
        else:
            print(f)
            pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
            pres_mat = pres_mat[np.where(session_mat!=24)]
            # Get just the lists in sessions 1-23rd sessions


        # Get all means and all sems for this participant
        nums_participant = top_ten_words_participant(pres_mat)
        #print(nums_participant)
        m_pool.append(nums_participant)
        #print("Nums Part Length", np.size(nums_participant))
        #print("Size of nums participant", len(nums_participant.tolist()))
    return m_pool


    # all_probs = prbl.get_ltpFR2_precall(files_ltpFR2)
    # correlations = scipy.stats.pearsonr(all_num_top_ten, all_probs)

if __name__ == "__main__":
    # Get file path
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    m_pool = get_m_pool(files_ltpFR2)
    print("All m_pool", m_pool)
    print("Shape of m_pool", np.shape(m_pool))
