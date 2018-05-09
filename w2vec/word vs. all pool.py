import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats


def fun(a):
    # Load the word to vec path
    w2v_path = "/Users/adaaka/Desktop/w2v.mat"
    w2v = scipy.io.loadmat(w2v_path, squeeze_me=True, struct_as_record=False)['w2v']
    #print(w2v)


    # Read in data
    test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)
    # Skip if there is no data for a participant
    if isinstance(test_mat_file['data'], np.ndarray):
        print('Skipping...')
        continue
    session_mat = test_mat_file['data'].session
    # print(len(session_mat))
    # print(np.bincount(session_mat))

    # Skip if the participant did not finish
    if len(session_mat) < 552:
        print('Skipping because participant did not finish...')
        continue


    else:
        print(f)
        pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')
        #print(pres_mat)
        #print(pres_mat.shape)
        #return pres_mat

        # Zeros matrix for each of the word similarity values and their standard error
        similarity_other = [[] for i in range(1638)]
        check_similarity = []
        # loop over each row of a given participant
        for trial in pres_mat[np.where(session_mat!=24)]:
            # adjust id numbers from MATLAB to Python
            trial -= 1
            # go over each word in that trial/row
            for word in trial:

                # list of all other words in the  row
                print("Word", word)
                other = deepcopy(trial).tolist()

                # remove the word
                other.remove(word)
                print("Other", other)
                # append similarities with all other words to that word's list of similarities
                similarity_other[word] += w2v[word, other].tolist()
                #print(similarity_other)
                check_similarity.append(np.nanmean(w2v[word, other]))
                print(check_similarity)
                raise ValueError("Stop and check values")
    similarity_other = np.array([row for row in similarity_other if len(row) > 0])

    sem = stats.sem(similarity_other, axis = 1)
    mean = np.mean(similarity_other, axis = 1)
    #print(sem)
    #print(sem)
    np.set_printoptions(threshold=np.inf)
    print("Similarity other", similarity_other)
    print("MEAN similarity", len(mean))
    print("SEM similarity", len(sem))
    # printing id's of the ltpFR2 word pool that was in this analyses: print(np.where(similarity_other > 0)[0]+1)








if __name__ == "__main__":
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    pres_mat = fun(files_ltpFR2)