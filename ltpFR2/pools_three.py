import scipy.io
import numpy as np
import copy

def get_bins():
    conc_file_path = "/Users/adaaka/Desktop/Desktop/concreteness_imageability_norms.mat"
    word_freq_path = "/Users/adaaka/Desktop/Desktop/Frequency_norms.mat"

    conc_file = scipy.io.loadmat(conc_file_path, squeeze_me=True, struct_as_record=False)
    word_freq = scipy.io.loadmat(word_freq_path, squeeze_me=True, struct_as_record=False)

    #print(word_freq['F'].astype('int'))
    #print("Frequencies: Length is:", len(word_freq['F']), "\n")

    #print(conc_file['I'].astype('int'))
    #print("Imaginability: Length is:", len(conc_file['I']), "\n")

    #print(conc_file['C'].astype('int'))
    #print("Concreteness: Length is:", len(conc_file['C']))

    conc_index = []
    imag_index = []
    conc_index = concreteness_index(conc_file['C'])
    imag_index = imaginability_index(conc_file['I'])
    final_indexes = []

    for element in conc_index:
        if element in imag_index:
            final_indexes.append(element)
    print("FINAL_INDEXES:",final_indexes)
    print("LENGTH IS:", len(final_indexes))

    final_freq = pool_freq(word_freq, final_indexes)
    bins = (bin(final_freq))
    print(bins)
    word_dict = {}
    for i in range(len(final_indexes)):
        word_dict[final_indexes[i]] = bins[i]
    return word_dict



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



def concreteness_index(a):
    conc_index = []
    for index in range(len(a)):
        if a[index] > 0:
            conc_index.append(index+1)
    #print("CONC_INDEX:", conc_index)
    #print("LENGTH IS:", len(conc_index))
    return(conc_index)

def imaginability_index(a):
    imag_index = []
    for index in range(len(a)):
        if a[index] > 0:
            imag_index.append(index+1)
    #print("IMAG_INDEX:",imag_index)
    #print("LENGTH IS:", len(imag_index))
    return(imag_index)

def pool_freq(word_freq, final_indexes):
    frequncy = word_freq['F'].astype('int')
    final_freq = []
    for i in final_indexes:
        final_freq.append(frequncy[i-1])
    #print(final_freq)
    #print("LENGTH IS:", len(final_freq))
    return(final_freq)

def bin(final_freq):
    bins = []
    for i in range(len(final_freq)):
        if final_freq[i] >= 2 and final_freq[i] <= 37:
            bins.append(0)
        elif final_freq[i] > 37 and final_freq[i] <= 71:
            bins.append(1)
        elif final_freq[i] > 71 and final_freq[i] <= 116:
            bins.append(2)
        elif final_freq[i] > 116 and final_freq[i] <= 165:
            bins.append(3)
        elif final_freq[i] > 165 and final_freq[i] <= 237:
            bins.append(4)
        elif final_freq[i] > 237 and final_freq[i] <= 343:
            bins.append(5)
        elif final_freq[i] > 243 and final_freq[i] <= 496:
            bins.append(6)
        elif final_freq[i] > 496 and final_freq[i] <= 815:
            bins.append(7)
        elif final_freq[i] > 815 and final_freq[i] <= 1575:
            bins.append(8)
        elif final_freq[i] > 1575 and final_freq[i] <= 26215:
            bins.append(9)
    return(bins)








if __name__ == "__main__":
    get_bins()

