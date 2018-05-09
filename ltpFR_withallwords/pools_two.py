import scipy.io
import numpy as np
import copy

# DO NOT USE

def get_bins():
    word_freq_path = "/Users/adaaka/Desktop/wpWASfreq.mat"
    word_freq = scipy.io.loadmat(word_freq_path, squeeze_me=True, struct_as_record=False)

    final_freq = word_freq['F'].astype('int')
    print(final_freq)
    print("Frequencies: Length is:", len(word_freq['F']), "\n")

    final_indexes = list(range(0,1638))
    print((final_indexes))


    bins = (bin(final_freq))
    print("BINS", bins)

    print((bin(final_freq)).count(0))
    print((bin(final_freq)).count(1))
    print((bin(final_freq)).count(2))
    print((bin(final_freq)).count(3))
    print((bin(final_freq)).count(4))
    print((bin(final_freq)).count(5))
    print((bin(final_freq)).count(6))
    print((bin(final_freq)).count(7))
    print((bin(final_freq)).count(8))
    print((bin(final_freq)).count(9))
    print(len(bin(final_freq)))

    word_dict = {}
    for i in range(len(final_indexes)):
        word_dict[final_indexes[i]] = bins[i]
    return word_dict


def bin(final_freq):
    bins = []
    for i in range(len(final_freq)):
        if final_freq[i] >= 0 and final_freq[i] <= 15:
            bins.append(0)
        elif final_freq[i] > 15 and final_freq[i] <= 33:
            bins.append(1)
        elif final_freq[i] > 33 and final_freq[i] <= 51:
            bins.append(2)
        elif final_freq[i] > 51 and final_freq[i] <= 84:
            bins.append(3)
        elif final_freq[i] > 84 and final_freq[i] <= 132:
            bins.append(4)
        elif final_freq[i] > 132 and final_freq[i] <= 187:
            bins.append(5)
        elif final_freq[i] > 187 and final_freq[i] <= 285:
            bins.append(6)
        elif final_freq[i] > 285 and final_freq[i] <= 478:
            bins.append(7)
        elif final_freq[i] > 478 and final_freq[i] <= 1000:
            bins.append(8)
        elif final_freq[i] > 1000 and final_freq[i] <= 26215:
            bins.append(9)
    return(bins)





if __name__ == "__main__":
    get_bins()

