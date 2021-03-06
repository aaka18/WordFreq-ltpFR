import scipy.io
import numpy as np
import copy
from ltpFR2.WordFreq_Excel import bin_edges
def get_bins():
    word_freq_path = "/Users/adaaka/Desktop/Desktop/Frequency_norms.mat"
    word_freq = scipy.io.loadmat(word_freq_path, squeeze_me=True, struct_as_record=False)
    print('First WFreq: %d' % word_freq['F'][0])
    print(word_freq.keys())

# The array below includes the final indices of 576 ltpFR2 words in the ltpFR word pool, so that their word frequencies could be found


    final_indexes = [5, 6, 10, 12, 13, 24, 29, 33, 35, 38, 39, 46, 53, 55, 61, 63, 64, 66, 67, 68, 75, 82, 83, 84, 86,
                    88, 91, 94, 100, 104, 108, 109, 113, 119, 125, 127, 131, 133, 134, 136, 137, 139, 142, 143, 145,
                    157, 159, 162, 164, 166, 168, 169, 171, 172, 175, 177, 179, 183, 184, 187, 188, 194, 198, 199, 200,
                    203, 210, 211, 216, 223, 226, 230, 235, 240, 248, 249, 253, 254, 255, 258, 261, 265, 267, 278, 280,
                    283, 292, 295, 299, 304, 307, 308, 313, 315, 316, 323, 325, 327, 328, 336, 338, 339, 340, 341, 343,
                    346, 348, 353, 358, 360, 368, 371, 372, 373, 378, 383, 384, 386, 388, 389, 390, 392, 395, 397, 400,
                    401, 403, 405, 407, 408, 410, 413, 415, 417, 425, 427, 429, 431, 435, 437, 439, 441, 442, 446, 447,
                    466, 467, 476, 483, 484, 485, 488, 490, 492, 494, 495, 496, 497, 498, 499, 501, 508, 509, 512, 519,
                    520, 530, 541, 543, 546, 553, 559, 561, 565, 567, 569, 570, 573, 575, 576, 578, 579, 580, 584, 585,
                    590, 591, 592, 593, 598, 599, 600, 601, 602, 604, 605, 606, 610, 614, 618, 620, 622, 623, 624, 627,
                    628, 633, 635, 637, 639, 641, 642, 645, 654, 658, 659, 660, 661, 662, 666, 667, 670, 675, 682, 683,
                    684, 689, 691, 692, 696, 697, 699, 702, 706, 707, 708, 713, 714, 715, 718, 719, 723, 724, 726, 728,
                    731, 732, 733, 734, 737, 738, 745, 746, 749, 752, 754, 755, 760, 761, 763, 764, 766, 769, 771, 776,
                    778, 782, 783, 785, 790, 792, 793, 794, 795, 796, 798, 800, 801, 804, 807, 809, 812, 814, 820, 821,
                    827, 829, 833, 834, 839, 840, 843, 846, 847, 848, 849, 852, 857, 858, 861, 866, 869, 870, 871, 872,
                    873, 874, 876, 878, 879, 883, 885, 888, 893, 895, 900, 901, 908, 912, 916, 919, 922, 929, 932, 935,
                    937, 940, 947, 953, 955, 959, 960, 963, 972, 974, 976, 977, 980, 982, 987, 988, 989, 994, 995, 996,
                    997, 999, 1000, 1002, 1006, 1009, 1011, 1013, 1015, 1016, 1017, 1019, 1021, 1022, 1024, 1027, 1031,
                    1032, 1035, 1036, 1037, 1044, 1047, 1048, 1049, 1051, 1057, 1060, 1061, 1065, 1066, 1070, 1071,
                    1075, 1076, 1079, 1080, 1085, 1086, 1087, 1088, 1089, 1092, 1093, 1095, 1096, 1098, 1104, 1108,
                    1111, 1112, 1113, 1117, 1119, 1123, 1124, 1125, 1132, 1139, 1140, 1143, 1144, 1147, 1148, 1149,
                    1152, 1155, 1158, 1159, 1170, 1171, 1172, 1173, 1176, 1183, 1184, 1192, 1194, 1197, 1199, 1200,
                    1202, 1203, 1209, 1215, 1219, 1221, 1224, 1227, 1229, 1232, 1238, 1242, 1243, 1245, 1246, 1250,
                    1251, 1256, 1258, 1260, 1261, 1262, 1271, 1272, 1275, 1280, 1282, 1284, 1287, 1292, 1293, 1294,
                    1295, 1299, 1301, 1303, 1307, 1312, 1315, 1317, 1319, 1326, 1327, 1328, 1329, 1330, 1331, 1332,
                    1333, 1341, 1343, 1345, 1347, 1349, 1351, 1355, 1356, 1357, 1358, 1368, 1369, 1370, 1371, 1373,
                    1374, 1382, 1384, 1385, 1388, 1392, 1395, 1400, 1403, 1405, 1407, 1408, 1411, 1414, 1415, 1416,
                    1420, 1422, 1423, 1426, 1427, 1436, 1438, 1440, 1448, 1451, 1456, 1460, 1461, 1468, 1472, 1478,
                    1486, 1488, 1493, 1496, 1501, 1503, 1505, 1508, 1511, 1512, 1515, 1519, 1520, 1521, 1522, 1524,
                    1526, 1527, 1530, 1532, 1533, 1535, 1536, 1544, 1545, 1546, 1555, 1557, 1562, 1565, 1572, 1578,
                    1580, 1581, 1587, 1592, 1593, 1605, 1606, 1608, 1609, 1611, 1618, 1620, 1621, 1622, 1625, 1626,
                    1628, 1629, 1631, 1633, 1635, 1636]
    final_indexes = np.array(final_indexes) -1
    # Given the final index, finding the word freq and the appropriate bin #
    final_freq = pool_freq(word_freq, final_indexes)
    bins = (bin(final_freq))
    print(bins)


    # Creating a word dictionary with the final indices of words and their corresponding word frequency values
    word_dict = {}
    for i in range(len(final_indexes)):
        word_dict[final_indexes[i]] = bins[i]
    return word_dict



#    print((bin(final_freq)).count(0))
#    print((bin(final_freq)).count(1))
#    print((bin(final_freq)).count(2))
#    print((bin(final_freq)).count(3))
#    print((bin(final_freq)).count(4))
#     print((bin(final_freq)).count(5))
#     print((bin(final_freq)).count(6))
#     print((bin(final_freq)).count(7))
#     print((bin(final_freq)).count(8))
#     print((bin(final_freq)).count(9))
#     print(len(bin(final_freq)))



def pool_freq(word_freq, final_indexes):

    # Finding the corresponding word frequencies when the word index is given
    frequency = word_freq['F'].astype('int')
    print("First word", frequency[final_indexes[1]])
    print(len(frequency))
    final_freq = []
    for i in final_indexes:
        final_freq.append(frequency[i])
    #print(final_freq)
    #print("LENGTH IS:", len(final_freq))
    return(final_freq)

def bin(final_meaningfulness):
    bins = []

    # Finding the bin number that corresponds to word's frequency value. For example, a word with a freq value of 50 would be in bin 2


    # [0.066735214413799995, 0.083195649134599994, 0.094137637353199993, 0.10206963121900001, 0.110626921751,
    #  0.119491107842, 0.12765822482399999, 0.13569146022, 0.14670560746399999, 0.18063925375000001]

    for i in range(len(final_meaningfulness)):
        if final_meaningfulness[i] >= 0 and final_meaningfulness[i] <= bin_edges[0]:
            bins.append(0)
        elif final_meaningfulness[i] > bin_edges[0] and final_meaningfulness[i] <= bin_edges[1]:
            bins.append(1)
        elif final_meaningfulness[i] >  bin_edges[1] and final_meaningfulness[i] <= bin_edges[2]:
            bins.append(2)
        elif final_meaningfulness[i] > bin_edges[2] and final_meaningfulness[i] <= bin_edges[3]:
            bins.append(3)
        elif final_meaningfulness[i] >bin_edges[3] and final_meaningfulness[i] <= bin_edges[4]:
            bins.append(4)
        elif final_meaningfulness[i] > bin_edges[4] and final_meaningfulness[i] <= bin_edges[5]:
            bins.append(5)
        elif final_meaningfulness[i] > bin_edges[5] and final_meaningfulness[i] <=  bin_edges[6]:
            bins.append(6)
        elif final_meaningfulness[i] > bin_edges[6] and final_meaningfulness[i] <= bin_edges[7]:
            bins.append(7)
        elif final_meaningfulness[i] > bin_edges[7] and final_meaningfulness[i] <=  bin_edges[8]:
            bins.append(8)
        elif final_meaningfulness[i] >  bin_edges[8]and final_meaningfulness[i] <= bin_edges[9]:
            bins.append(9)
        elif final_meaningfulness[i] > bin_edges[9] and final_meaningfulness[i] <= bin_edges[10] :
            bins.append(10)
        elif final_meaningfulness[i] > bin_edges[10] and final_meaningfulness[i] <= bin_edges[11] :
            bins.append(11)
        elif final_meaningfulness[i] > bin_edges[11] and final_meaningfulness[i] <=  bin_edges[12]:
            bins.append(12)
        elif final_meaningfulness[i] > bin_edges[12] and final_meaningfulness[i] <= bin_edges[13] :
            bins.append(13)
        elif final_meaningfulness[i] > bin_edges[13] and final_meaningfulness[i] <= bin_edges[14]:
            bins.append(14)
        elif final_meaningfulness[i] > bin_edges[14] and final_meaningfulness[i] <= bin_edges[15] :
            bins.append(15)
        elif final_meaningfulness[i] > bin_edges[15] and final_meaningfulness[i] <= bin_edges[16] :
            bins.append(16)
        elif final_meaningfulness[i] > bin_edges[16] and final_meaningfulness[i] <= bin_edges[17] :
            bins.append(17)
        elif final_meaningfulness[i] > bin_edges[17] and final_meaningfulness[i] <= bin_edges[18] :
            bins.append(18)
        elif final_meaningfulness[i] > bin_edges[18] :
            bins.append(19)
    print(bins)
    print(len(bins))
    return(bins)





if __name__ == "__main__":
    get_bins()

