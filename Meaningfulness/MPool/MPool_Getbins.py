import scipy.io
import numpy as np
import copy
import numpy.ma as ma
from Meaningfulness.MPool.MPool_Excel import bin_edges
import w2vec.Final.ltpFR2_word_to_allwords as m_pool

def get_bins():

    final_indexes = np.array([5, 6, 10, 12, 13, 24, 29, 33, 35, 38, 39, 46, 53, 55, 61, 63, 64, 66, 67, 68, 75, 82, 83, 84, 86,
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
                    1628, 1629, 1631, 1633, 1635, 1636])

    concreteness = np.array([4.57, 4.54, 3.61, 4.96, 4.87, 4.81, 4.86, 5, 4.87, 4.96, 4.7, np.nan, 4.79, 3.34, 4.26, 4.14, 5,
                    4.19, 4.9, 4.93, 4.92, 4.9, 4.78, 4.43, 4, 4.59, 4.86, 4.89, 4.92, 4.72, 4.63, 4.68, 4.74, 4.8, 5,
                    4.89, 4.68, 4.72, 4.93, 5, 4.25, 4.96, 4.77, 4.57, 4.79, 4.74, 4.9, 4.59, 5, 4.44, 4.9, 4.81, 4.89,
                    4.83, 4.86, 4.43, 4.43, 4.6, 4.96, 5, 4.18, 4.83, 4.97, 4.04, 4.44, 4.65, 4.75, 4.92, 4.96, 4.93,
                    4.68, 4.83, 4.81, 3.03, 4.86, 5, 4.92, 4.89, 4.86, 4.44, 4.64, 4.85, 4.68, 4.82, 4.6, 4.43, 4.24,
                    4.93, 4.78, 4.97, 4.9, 4.93, 4.21, 4.89, 4.53, 4.5, 5, 4.76, 4.54, 5, 4.4, 4.83, 4.61, 4.81, 4.89,
                    4.62, 3.89, 4.67, 4.66, 4.35, 4.15, 4.11, 4.32, 4.9, 4.4, 4.57, 4.85, 4.71, 4.17, 4.04, 3.82, 3.7,
                    4.72, 4.9, 4.61, 4.87, 4.07, 4.43, 4.86, 4.77, 3.55, 4.44, 4.81, 4.5, 4.79, 4.82, 4.85, 4.48, 5,
                    4.75, 4.79, 4.61, 4.79, 4.77, 4.93, 4.85, 4.82, 4.69, 4.96, 5, 3.54, 4.41, 4.96, 4.39, 4.6, 4.93,
                    4.96, 4.4, 4.76, 4.71, 4.48, 4.4, 5, 5, np.nan, 5, 3, np.nan, 2.85, 5, 4.54, 4.57, 4.81, 4.71, 5,
                    4.8, 4.68, 4.79, 5, 4.79, 3.81, 4.59, 4.26, 5, 5, 4.9, 4.73, 4.9, 4.76, 4.97, 4.03, 4.3, np.nan,
                    4.56, 4.87, 3.88, 3.07, 4.81, 4.59, 3.92, 3.93, 4.69, 4.73, 4.89, 4.88, 4.72, 3.82, 4.56, 4.85,
                    4.82, 4.59, 4.97, 4.38, 5, 4.56, 5, 4.61, 4.86, 3.17, 4.77, 4.04, 4.9, 4.85, 4.21, 4.72, 4.93, 4.85,
                    4.93, 4.93, 5, 4.52, 4.54, 4.92, 3.07, 4.72, 4.53, 4.88, 4.88, 4.79, 4.96, 5, 4.12, 4.48, 4.93,
                    4.11, 4.96, 4.66, 3.63, 4.73, 4.93, 4.19, 4.96, 4.41, 4.82, 5, np.nan, 4.93, 4.63, 3.75, 4.5, 4.66,
                    4.64, 3, 5, 4.96, 4.97, 4.92, 4.9, 4.9, 4.46, 4.85, 5, 4.33, 4.5, 4.88, 4.97, 4.56, 4.5, 4.82, 3.89,
                    4.83, 5, 4.97, 4.59, 4.69, 4.96, 4.9, 4.68, 4.68, 4, 4.32, np.nan, 3.68, 4.83, 4.56, 4.31, 4.25, 5,
                    4.57, 4.59, 4.46, 4.25, 4.62, 4.7, 4.48, 4.48, 4.85, 4.96, 4.14, 5, 4.9, 4.89, 3.97, 4.57, 4.92,
                    4.83, 3.15, 4.54, 3.72, 4.97, 4.93, 4.84, 4.78, 4.72, 4.93, 4.93, 4.96, 2.69, 4.9, 4.1, 4.92, 4.21,
                    4.39, 4.93, 4.5, 4.93, 4.86, 4.66, 4.92, 4.61, 4.12, 3.61, 4.86, 4.85, 3.5, 4.72, 4.52, 4.8, 4.93,
                    4.57, 4.93, 4.93, 4.56, 5, 4.77, 3.53, 3.8, 4.86, 4.97, 2.5, 3.86, 4.9, 4.89, 4.86, 4.87, 4.44, 5,
                    4.59, 4.1, 4.66, 3.43, 4.9, 4.83, 4.52, 4.71, 4.4, 4.67, 4.77, 4.89, 4.81, 5, 4.23, 4.59, 4.77,
                    4.77, 4.44, 4.93, np.nan, 4.68, 4.36, 4.27, 4.79, 5, 4.79, 4.9, 4.73, 4.37, 4.76, 4.7, 4.5, 4.44,
                    4.72, 3, 4.9, 4.67, 4.55, 4.78, 4.65, 4.43, 4.45, 4.93, 4.26, 4.87, 5, 4.07, 4.9, 3.07, 4.86, 4.15,
                    3.3, 3.92, 4.85, 4.89, 4.31, 4.61, 4.65, 4.73, 4.43, 4.75, 4.79, 4.52, 4.85, 4.97, 4.81, 4.89, 4.9,
                    4.88, 4.61, 4.86, 4.68, 4.97, 4.85, 4.55, 4.1, 4.79, 4.83, 5, 4.63, 4.64, 4.55, 4.93, 4.96, 4.64,
                    4.5, 4.94, 4.41, 4.82, 4.93, 4.97, 4.92, 4.37, 4.96, 4.7, 4, 4.56, 4.73, 4.82, 4.48, 4.48, 4.07,
                    4.64, 4.14, 4.36, 4.93, 5, 4.97, 4.89, 3.54, 4.85, 4.83, 4.97, 5, 4.62, 4.96, 3.85, 4.21, 4.72, 4.7,
                    4.34, 4.69, 4.93, 4.67, 4.89, 4.72, 4.96, 4.5, 4.92, 4.86, 4.97, 4.21, 4.69, 4.54, 4.63, 4.08, 2.59,
                    4.96, 4.77, 4.07, 4.93, 4.9, 4.82, 3.27, 4.93, 4.52, 4.53, 4.7, 4.37, 4.83, 4.64, 4.68, 4.9, 4.71,
                    4.87, 4.59, 5, 3.77, 3.79, 4.46, 4.14, 4.9, 4.72, 4.84, 4.86, 4.82, 4.89, 4.79, 5, 4.68, 4.96, 4.75,
                    4.44, 4.41, 4.69, 4.27, 4.24, 3.46, 4.72, 4.83, 4.44, 4.54, 3.59, np.nan, 3.48, 4.89, 4.67, 4.56,
                    4.67, 4.7, 4.96, 4.89, 4.42, 4.33, 4.13, 4.86, 4.07, 4.46, 4.59, 4.36, 4.93, 4.93, 3.96, 4.97, 4.93,
                    4.78, 4.86, 4.83])
    import w2vec.Final.ltpFR2_word_to_allwords as m_pool
    m_pool = m_pool.w2v_ltpFR2
    mpool = np.array(m_pool)

    final_indexes_masked = final_indexes[np.logical_not(np.isnan(concreteness))]
    final_mpool = mpool[np.logical_not(np.isnan(concreteness))]
    print("Final valence", len(final_mpool))
    bins = bin(final_mpool)
    print(bins)

    '''   final_indexes_masked = final_indexes[np.logical_not(np.isnan(concreteness))]

    # Given the final index, finding the word freq and the appropriate bin #
    # final_conc = pool_freq(word_freq, final_indexes_masked)
    final_conc = concreteness[np.logical_not(np.isnan(concreteness))]
    bins = bin(final_conc)
    print(bins)
'''
    # Creating a word dictionary with the final indices of words and their corresponding word frequency values
    word_dict = {}
    print(final_indexes_masked)
    for i in range(len(final_indexes_masked)):
        print(final_indexes_masked[i], bins[i])
        word_dict[final_indexes_masked[i]] = bins[i]
    return word_dict

def bin(final_meaningfulness):
    bins = []

    # Finding the bin number that corresponds to word's frequency value. For example, a word with a freq value of 50 would be in bin 2


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
        elif final_meaningfulness[i] > bin_edges[11] and final_meaningfulness[i] <= bin_edges[12]:
            bins.append(12)
        elif final_meaningfulness[i] > bin_edges[12] and final_meaningfulness[i] <= bin_edges[13]:
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
        elif final_meaningfulness[i] > bin_edges[18]  :
            bins.append(19)
    print(bins)
    print(len(bins))
    return(bins)






if __name__ == "__main__":
    get_bins()

