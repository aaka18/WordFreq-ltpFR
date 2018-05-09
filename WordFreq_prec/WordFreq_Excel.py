import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table,Column
import math
import numpy.ma as ma


concreteness = [4.57, 4.54, 3.61, 4.96, 4.87, 4.81, 4.86, 5, 4.87, 4.96, 4.7, np.nan, 4.79, 3.34, 4.26, 4.14, 5, 4.19, 4.9, 4.93, 4.92, 4.9, 4.78, 4.43, 4, 4.59, 4.86, 4.89, 4.92, 4.72, 4.63, 4.68, 4.74, 4.8, 5, 4.89, 4.68, 4.72, 4.93, 5, 4.25, 4.96, 4.77, 4.57, 4.79, 4.74, 4.9, 4.59, 5, 4.44, 4.9, 4.81, 4.89, 4.83, 4.86, 4.43, 4.43, 4.6, 4.96, 5, 4.18, 4.83, 4.97, 4.04, 4.44, 4.65, 4.75, 4.92, 4.96, 4.93, 4.68, 4.83, 4.81, 3.03, 4.86, 5, 4.92, 4.89, 4.86, 4.44, 4.64, 4.85, 4.68, 4.82, 4.6, 4.43, 4.24, 4.93, 4.78, 4.97, 4.9, 4.93, 4.21, 4.89, 4.53, 4.5, 5, 4.76, 4.54, 5, 4.4, 4.83, 4.61, 4.81, 4.89, 4.62, 3.89, 4.67, 4.66, 4.35, 4.15, 4.11, 4.32, 4.9, 4.4, 4.57, 4.85, 4.71, 4.17, 4.04, 3.82, 3.7, 4.72, 4.9, 4.61, 4.87, 4.07, 4.43, 4.86, 4.77, 3.55, 4.44, 4.81, 4.5, 4.79, 4.82, 4.85, 4.48, 5, 4.75, 4.79, 4.61, 4.79, 4.77, 4.93, 4.85, 4.82, 4.69, 4.96, 5, 3.54, 4.41, 4.96, 4.39, 4.6, 4.93, 4.96, 4.4, 4.76, 4.71, 4.48, 4.4, 5, 5, np.nan, 5, 3, np.nan, 2.85, 5, 4.54, 4.57, 4.81, 4.71, 5, 4.8, 4.68, 4.79, 5, 4.79, 3.81, 4.59, 4.26, 5, 5, 4.9, 4.73, 4.9, 4.76, 4.97, 4.03, 4.3, np.nan, 4.56, 4.87, 3.88, 3.07, 4.81, 4.59, 3.92, 3.93, 4.69, 4.73, 4.89, 4.88, 4.72, 3.82, 4.56, 4.85, 4.82, 4.59, 4.97, 4.38, 5, 4.56, 5, 4.61, 4.86, 3.17, 4.77, 4.04, 4.9, 4.85, 4.21, 4.72, 4.93, 4.85, 4.93, 4.93, 5, 4.52, 4.54, 4.92, 3.07, 4.72, 4.53, 4.88, 4.88, 4.79, 4.96, 5, 4.12, 4.48, 4.93, 4.11, 4.96, 4.66, 3.63, 4.73, 4.93, 4.19, 4.96, 4.41, 4.82, 5, np.nan, 4.93, 4.63, 3.75, 4.5, 4.66, 4.64, 3, 5, 4.96, 4.97, 4.92, 4.9, 4.9, 4.46, 4.85, 5, 4.33, 4.5, 4.88, 4.97, 4.56, 4.5, 4.82, 3.89, 4.83, 5, 4.97, 4.59, 4.69, 4.96, 4.9, 4.68, 4.68, 4, 4.32, np.nan, 3.68, 4.83, 4.56, 4.31, 4.25, 5, 4.57, 4.59, 4.46, 4.25, 4.62, 4.7, 4.48, 4.48, 4.85, 4.96, 4.14, 5, 4.9, 4.89, 3.97, 4.57, 4.92, 4.83, 3.15, 4.54, 3.72, 4.97, 4.93, 4.84, 4.78, 4.72, 4.93, 4.93, 4.96, 2.69, 4.9, 4.1, 4.92, 4.21, 4.39, 4.93, 4.5, 4.93, 4.86, 4.66, 4.92, 4.61, 4.12, 3.61, 4.86, 4.85, 3.5, 4.72, 4.52, 4.8, 4.93, 4.57, 4.93, 4.93, 4.56, 5, 4.77, 3.53, 3.8, 4.86, 4.97, 2.5, 3.86, 4.9, 4.89, 4.86, 4.87, 4.44, 5, 4.59, 4.1, 4.66, 3.43, 4.9, 4.83, 4.52, 4.71, 4.4, 4.67, 4.77, 4.89, 4.81, 5, 4.23, 4.59, 4.77, 4.77, 4.44, 4.93, np.nan, 4.68, 4.36, 4.27, 4.79, 5, 4.79, 4.9, 4.73, 4.37, 4.76, 4.7, 4.5, 4.44, 4.72, 3, 4.9, 4.67, 4.55, 4.78, 4.65, 4.43, 4.45, 4.93, 4.26, 4.87, 5, 4.07, 4.9, 3.07, 4.86, 4.15, 3.3, 3.92, 4.85, 4.89, 4.31, 4.61, 4.65, 4.73, 4.43, 4.75, 4.79, 4.52, 4.85, 4.97, 4.81, 4.89, 4.9, 4.88, 4.61, 4.86, 4.68, 4.97, 4.85, 4.55, 4.1, 4.79, 4.83, 5, 4.63, 4.64, 4.55, 4.93, 4.96, 4.64, 4.5, 4.94, 4.41, 4.82, 4.93, 4.97, 4.92, 4.37, 4.96, 4.7, 4, 4.56, 4.73, 4.82, 4.48, 4.48, 4.07, 4.64, 4.14, 4.36, 4.93, 5, 4.97, 4.89, 3.54, 4.85, 4.83, 4.97, 5, 4.62, 4.96, 3.85, 4.21, 4.72, 4.7, 4.34, 4.69, 4.93, 4.67, 4.89, 4.72, 4.96, 4.5, 4.92, 4.86, 4.97, 4.21, 4.69, 4.54, 4.63, 4.08, 2.59, 4.96, 4.77, 4.07, 4.93, 4.9, 4.82, 3.27, 4.93, 4.52, 4.53, 4.7, 4.37, 4.83, 4.64, 4.68, 4.9, 4.71, 4.87, 4.59, 5, 3.77, 3.79, 4.46, 4.14, 4.9, 4.72, 4.84, 4.86, 4.82, 4.89, 4.79, 5, 4.68, 4.96, 4.75, 4.44, 4.41, 4.69, 4.27, 4.24, 3.46, 4.72, 4.83, 4.44, 4.54, 3.59, np.nan, 3.48, 4.89, 4.67, 4.56, 4.67, 4.7, 4.96, 4.89, 4.42, 4.33, 4.13, 4.86, 4.07, 4.46, 4.59, 4.36, 4.93, 4.93, 3.96, 4.97, 4.93, 4.78, 4.86, 4.83]
# concreteness_masked = ma.masked_invalid(concreteness)
# concreteness_masked = np.sort(concreteness_masked)
# concreteness_masked_final= (concreteness_masked[:568])
# print(concreteness_masked_final)

word_freq = [785, 254, 766, 73, 932, 185, 35, 315, 131, 1860, 1933, 475, 40, 187, 522, 169, 3281, 78, 288, 117, 58, 6, 2333, 129, 67, 174, 253, 224, 31, 21, 298, 37, 273, 305, 149, 16, 26, 96, 205, 22, 48, 156, 53, 1525, 5243, 75, 704, 80, 33, 65, 961, 291, 774, 498, 144, 74, 1531, 67, 237, 36, 15, 229, 19, 289, 44, 87, 144, 485, 342, 146, 211, 113, 149, 81, 231, 99, 8, 33, 35, 21, 568, 463, 183, 285, 372, 83, 238, 778, 7645, 2, 2844, 882, 1, 13, 19, 94, 637, 2225, 536, 35, 147, 31, 87, 1644, 132, 1380, 1610, 26, 82, 272, 538, 30, 266, 8, 39, 367, 555, 161, 6036, 785, 11914, 400, 84, 80, 85, 13, 619, 33, 19, 520, 786, 400, 412, 50, 270, 346, 54, 10, 531, 146, 6, 37, 1797, 50, 115, 184, 47, 16, 30, 162, 48, 72, 20, 135, 485, 1332, 83, 109, 1414, 746, 344, 762, 8, 129, 584, 280, 381, 2784, 331, 52, 1017, 1044, 28, 1546, 867, 20, 148, 168, 77, 76, 240, 930, 11, 476, 44, 1753, 573, 438, 1219, 177, 46, 442, 1779, 1, 61, 19, 3087, 513, 94, 117, 50, 166, 1954, 115, 12, 13, 205, 560, 4944, 2246, 184, 82, 6, 37, 270, 185, 106, 49, 3, 3084, 658, 102, 10, 29, 7889, 137, 43, 28, 67, 13, 2597, 178, 166, 545, 285, 12, 371, 77, 487, 10, 1518, 164, 33, 5113, 2405, 41, 42, 86, 8, 372, 106, 1209, 350, 541, 229, 8, 172, 318, 732, 6, 254, 521, 96, 26, 84, 1890, 42, 15, 635, 223, 186, 238, 1219, 56, 718, 381, 50, 31, 63, 1227, 1137, 118, 115, 253, 120, 152, 122, 242, 47, 222, 46, 5368, 430, 201, 25, 1381, 1469, 24, 3, 116, 56, 183, 37, 2374, 50, 166, 213, 237, 775, 170, 1280, 99, 1214, 34, 1792, 465, 68, 7226, 262, 38, 125, 465, 6, 198, 84, 84, 41, 168, 45, 46, 137, 495, 556, 4460, 46, 1, 170, 258, 44, 68, 182, 8, 79, 64, 62, 281, 31, 11, 48, 761, 50, 3128, 1266, 45, 118, 459, 655, 36, 129, 707, 161, 47, 49, 85, 7, 22, 69, 117, 34, 116, 70, 466, 184, 1905, 38, 31, 235, 11, 254, 56, 28, 49, 209, 656, 211, 61, 26, 20, 1003, 305, 204, 3694, 14, 150, 335, 20, 32, 327, 58, 76, 592, 217, 55, 198, 37, 617, 87, 7, 911, 889, 189, 139, 11, 57, 47, 139, 74, 122, 1077, 99, 1860, 286, 1938, 42, 207, 67, 142, 503, 10, 321, 67, 148, 281, 108, 629, 184, 103, 17, 18, 14, 142, 79, 153, 15, 373, 9, 18, 88, 892, 12, 246, 246, 358, 49, 812, 2, 195, 1196, 57, 66, 26, 104, 440, 1464, 122, 2, 351, 153, 35, 330, 38, 32, 58, 46, 251, 148, 8, 2259, 47, 15, 73, 107, 56, 201, 50, 231, 24, 183, 73, 952, 273, 35, 705, 737, 294, 661, 3587, 37, 220, 241, 51, 165, 480, 543, 133, 76, 41, 315, 237, 3645, 48, 39, 509, 1490, 416, 42, 120, 182, 175, 58, 11, 29, 71, 333, 123, 49, 145, 94, 416, 11, 280, 444, 5, 237, 173, 49, 20, 0, 29, 44, 8, 38, 29, 25, 1061, 9, 885, 87, 206, 108, 497, 17, 167, 143, 299, 138, 160, 30, 43, 41, 140, 254, 3777, 2372, 278, 6072, 3020, 13345, 47, 358, 2, 83, 24, 47, 12, 24]
mask = np.logical_not(np.isnan(concreteness))
word_freq_masked = np.array(word_freq)[mask]
print(len(word_freq_masked))
word_freq_masked = np.sort(word_freq_masked)
word_freq_masked_final= (word_freq_masked[:568])
print(word_freq_masked_final)
# print(len(x_ltpFR))
# plt.hist(x_ltpFR, bins =10)
# plt.suptitle('ltpFR Word Pool Binned With ltpFR Bin Boundaries', fontsize=20)
# plt.xlabel('Bins 1 Through 10', fontsize=18)
# plt.ylabel('Total Number of Words In Each Word Bin', fontsize=16)
# plt.xticks([])
# plt.show()

print("Mean is = ", np.mean(word_freq_masked_final))
print('Length of list of array_sorted:', len(word_freq_masked_final))

bin_edges = []

# Finding bin edges by looking at the word pool in groups of 1/10 and identifying the max bin values
for i in range(1,21):
    edge = word_freq_masked_final[int(i*len(word_freq_masked_final)/20)-1]
    bin_edges.append(edge)

print("Bins",bin_edges)
for i, number in enumerate(bin_edges):
    print(i, number)
print("Len", len(bin_edges))
bin_zero = []
bin_one = []
bin_two = []
bin_three = []
bin_four = []
bin_five = []
bin_six = []
bin_seven = []
bin_eight = []
bin_nine = []
bin_ten = []
bin_eleven = []
bin_twelve = []
bin_thirteen = []
bin_fourteen = []
bin_fifteen = []
bin_sixteen = []
bin_seventeen = []
bin_eighteen = []
bin_nineteen = []


# Add the word frequency values to their corresponding bins depending on the max bin values
for i, freq in enumerate(word_freq_masked_final):
    if freq >= 0 and freq <= bin_edges[0]:
        bin_zero.append([i, freq])
    elif (bin_edges[0] < freq) and (freq <= bin_edges[1]):
        bin_one.append([i, freq])
    elif (bin_edges[1] < freq) and (freq <= bin_edges[2]):
        bin_two.append([i, freq])
    elif (bin_edges[2] < freq) and (freq <= bin_edges[3]):
        bin_three.append([i, freq])
    elif (bin_edges[3] < freq) and (freq <= bin_edges[4]):
        bin_four.append([i, freq])
    elif (bin_edges[4] < freq) and (freq <= bin_edges[5]):
        bin_five.append([i, freq])
    elif (bin_edges[5] < freq) and (freq <= bin_edges[6]):
        bin_six.append([i, freq])
    elif (bin_edges[6] < freq) and (freq <= bin_edges[7]):
        bin_seven.append([i, freq])
    elif (bin_edges[7] < freq) and (freq <= bin_edges[8]):
        bin_eight.append([i, freq])
    elif (bin_edges[8] < freq) and (freq  <= bin_edges[9]):
        bin_nine.append([i, freq])
    elif (bin_edges[9] < freq) and (freq <= bin_edges[10]):
        bin_ten.append([i, freq])
    elif (bin_edges[10] < freq) and (freq <= bin_edges[11]):
        bin_eleven.append([i, freq])
    elif (bin_edges[11] < freq) and (freq <= bin_edges[12]):
        bin_twelve.append([i, freq])
    elif (bin_edges[12] < freq) and (freq <= bin_edges[13]):
        bin_thirteen.append([i, freq])
    elif (bin_edges[13] < freq) and (freq <= bin_edges[14]):
        bin_fourteen.append([i, freq])
    elif (bin_edges[14] < freq) and (freq <= bin_edges[15]):
        bin_fifteen.append([i, freq])
    elif (bin_edges[15] < freq) and (freq <= bin_edges[16]):
        bin_sixteen.append([i, freq])
    elif (bin_edges[16] < freq) and (freq <= bin_edges[17]):
        bin_seventeen.append([i, freq])
    elif (bin_edges[17] < freq) and (freq  <= bin_edges[18]):
        bin_eighteen.append([i, freq])
    else:
        bin_nineteen.append([i, freq])
# Print lengths of each bin
#
print(len(bin_zero))
print(len(bin_one))
print(len(bin_two))
print(len(bin_three))
print(len(bin_four))
print(len(bin_five))
print(len(bin_six))
print(len(bin_seven))
print(len(bin_eight))
print(len(bin_nine))
print(len(bin_ten))
print(len(bin_eleven))
print(len(bin_twelve))
print(len(bin_thirteen))
print(len(bin_fourteen))
print(len(bin_fifteen))
print(len(bin_sixteen))
print(len(bin_seventeen))
print(len(bin_eighteen))
print(len(bin_nineteen))

# Sum of the lengths of each bin should be equal to the total number of words in the word pool
print((len(bin_zero) + len(bin_one) + len(bin_two) + len(bin_three) + len(bin_four) + len(bin_five) + len(bin_six) + len(bin_seven) + len(bin_eight) + len(bin_nine) + len(bin_ten)+
       len(bin_eleven) + len(bin_twelve) + len(bin_thirteen) + len(bin_fourteen) + len(bin_fifteen) + len(bin_sixteen) + len(bin_seventeen) + len(bin_eighteen) + len(bin_nineteen )))

# Finding the mean word frequency of each word bin
mean_zero = (np.mean(np.asarray(bin_zero), axis=0)[1])
mean_one = (np.mean(np.asarray(bin_one), axis=0)[1])
mean_two = (np.mean(np.asarray(bin_two), axis=0) [1])
mean_three = (np.mean(np.asarray(bin_three), axis=0)[1])
mean_four = (np.mean(np.asarray(bin_four), axis=0) [1])
mean_five = (np.mean(np.asarray(bin_five), axis=0)[1])
mean_six = (np.mean(np.asarray(bin_six), axis=0)[1])
mean_seven = (np.mean(np.asarray(bin_seven), axis=0)[1])
mean_eight = (np.mean(np.asarray(bin_eight), axis=0)[1])
mean_nine = (np.mean(np.asarray(bin_nine), axis=0) [1])
mean_ten = (np.mean(np.asarray(bin_ten), axis=0) [1])
mean_eleven = (np.mean(np.asarray(bin_eleven), axis=0) [1])
mean_twelve = (np.mean(np.asarray(bin_twelve), axis=0) [1])
mean_thirteen = (np.mean(np.asarray(bin_thirteen), axis=0)[1])
mean_fourteen = (np.mean(np.asarray(bin_fourteen), axis=0) [1])
mean_fifteen = (np.mean(np.asarray(bin_fifteen), axis=0)[1])
mean_sixteen = (np.mean(np.asarray(bin_sixteen), axis=0)[1])
mean_seventeen = (np.mean(np.asarray(bin_seventeen), axis=0)[1])
mean_eightteen = (np.mean(np.asarray(bin_eighteen), axis=0)[1])
mean_nineteen = (np.mean(np.asarray(bin_nineteen), axis=0) [1])
# Creating a table: Frequency Information for Each Word Bin

M = [mean_zero, mean_one, mean_two, mean_three,mean_four, mean_five,mean_six, mean_seven, mean_eight, mean_nine, mean_ten,
     mean_eleven, mean_twelve, mean_thirteen, mean_fourteen, mean_fifteen, mean_sixteen, mean_seventeen, mean_eightteen, mean_nineteen]

# M = [mean_zero, mean_one, mean_two, mean_three,mean_four, mean_five,mean_six, mean_seven, mean_eight, mean_nine, mean_ten,
#      mean_eleven, mean_twelve, mean_thirteen]

print(M)

for i, number in enumerate(M):
    print(i, number)
