# One scatter plot for p_rec of each word by each subject

import glob

import matplotlib.pyplot as plt
import numpy as np

from w2vec.Final import P_rec_ltpFR2_words as p_rec_word

files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
print(p_rec_word)
word = range(0, 576)


p_rec_word = np.mean(p_rec_word, axis = 0)
print(p_rec_word)
#plt.title("Variability in Word Recallability")
plt.scatter(word, sorted(p_rec_word), marker='.')
plt.xticks([], [])
plt.xlabel("Words")
plt.ylabel("Recall Probability")
plt.show()
plt.savefig('VariabilityWordRecallability.png')

# files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
# p_rec_list = p_rec.get_ltpFR2_precall(files_ltpFR2)
# p_rec_list = np.mean(p_rec_list, axis = 0)
# plt.title("Variability in List Recallability")
# plt.scatter(range(552), sorted(p_rec_list), marker='.')
# plt.xticks([], [])
# plt.xlabel("Lists")
# plt.ylabel("Recall Probability")
# plt.show()
# plt.savefig('VariabilityWordRecallability.png', )
#

# for i in word:
#     out = [x[i] for x in p_rec_word]
#     out = sorted(out)
#     #print(out)
#     x = [float(i)] * len(out)
#     plt.scatter(x, out)
# plt.show()

# sorted = []
# for i in word:
#     sorted.append(p_rec_word[:,i])
# print(sorted)
# print(sorted.shape)