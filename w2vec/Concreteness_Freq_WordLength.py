import numpy as np
import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Static lists for ltpFR2 word IDs, concreteness, and word frequencies.

word_id = [5, 6, 10, 12, 13, 24, 29, 33, 35, 38, 39, 46, 53, 55, 61, 63, 64, 66, 67, 68, 75, 82, 83, 84, 86, 88, 91, 94, 100, 104, 108, 109, 113, 119, 125, 127, 131, 133, 134, 136, 137, 139, 142, 143, 145, 157, 159, 162, 164, 166, 168, 169, 171, 172, 175, 177, 179, 183, 184, 187, 188, 194, 198, 199, 200, 203, 210, 211, 216, 223, 226, 230, 235, 240, 248, 249, 253, 254, 255, 258, 261, 265, 267, 278, 280, 283, 292, 295, 299, 304, 307, 308, 313, 315, 316, 323, 325, 327, 328, 336, 338, 339, 340, 341, 343, 346, 348, 353, 358, 360, 368, 371, 372, 373, 378, 383, 384, 386, 388, 389, 390, 392, 395, 397, 400, 401, 403, 405, 407, 408, 410, 413, 415, 417, 425, 427, 429, 431, 435, 437, 439, 441, 442, 446, 447, 466, 467, 476, 483, 484, 485, 488, 490, 492, 494, 495, 496, 497, 498, 499, 501, 508, 509, 512, 519, 520, 530, 541, 543, 546, 553, 559, 561, 565, 567, 569, 570, 573, 575, 576, 578, 579, 580, 584, 585, 590, 591, 592, 593, 598, 599, 600, 601, 602, 604, 605, 606, 610, 614, 618, 620, 622, 623, 624, 627, 628, 633, 635, 637, 639, 641, 642, 645, 654, 658, 659, 660, 661, 662, 666, 667, 670, 675, 682, 683, 684, 689, 691, 692, 696, 697, 699, 702, 706, 707, 708, 713, 714, 715, 718, 719, 723, 724, 726, 728, 731, 732, 733, 734, 737, 738, 745, 746, 749, 752, 754, 755, 760, 761, 763, 764, 766, 769, 771, 776, 778, 782, 783, 785, 790, 792, 793, 794, 795, 796, 798, 800, 801, 804, 807, 809, 812, 814, 820, 821, 827, 829, 833, 834, 839, 840, 843, 846, 847, 848, 849, 852, 857, 858, 861, 866, 869, 870, 871, 872, 873, 874, 876, 878, 879, 883, 885, 888, 893, 895, 900, 901, 908, 912, 916, 919, 922, 929, 932, 935, 937, 940, 947, 953, 955, 959, 960, 963, 972, 974, 976, 977, 980, 982, 987, 988, 989, 994, 995, 996, 997, 999, 1000, 1002, 1006, 1009, 1011, 1013, 1015, 1016, 1017, 1019, 1021, 1022, 1024, 1027, 1031, 1032, 1035, 1036, 1037, 1044, 1047, 1048, 1049, 1051, 1057, 1060, 1061, 1065, 1066, 1070, 1071, 1075, 1076, 1079, 1080, 1085, 1086, 1087, 1088, 1089, 1092, 1093, 1095, 1096, 1098, 1104, 1108, 1111, 1112, 1113, 1117, 1119, 1123, 1124, 1125, 1132, 1139, 1140, 1143, 1144, 1147, 1148, 1149, 1152, 1155, 1158, 1159, 1170, 1171, 1172, 1173, 1176, 1183, 1184, 1192, 1194, 1197, 1199, 1200, 1202, 1203, 1209, 1215, 1219, 1221, 1224, 1227, 1229, 1232, 1238, 1242, 1243, 1245, 1246, 1250, 1251, 1256, 1258, 1260, 1261, 1262, 1271, 1272, 1275, 1280, 1282, 1284, 1287, 1292, 1293, 1294, 1295, 1299, 1301, 1303, 1307, 1312, 1315, 1317, 1319, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1341, 1343, 1345, 1347, 1349, 1351, 1355, 1356, 1357, 1358, 1368, 1369, 1370, 1371, 1373, 1374, 1382, 1384, 1385, 1388, 1392, 1395, 1400, 1403, 1405, 1407, 1408, 1411, 1414, 1415, 1416, 1420, 1422, 1423, 1426, 1427, 1436, 1438, 1440, 1448, 1451, 1456, 1460, 1461, 1468, 1472, 1478, 1486, 1488, 1493, 1496, 1501, 1503, 1505, 1508, 1511, 1512, 1515, 1519, 1520, 1521, 1522, 1524, 1526, 1527, 1530, 1532, 1533, 1535, 1536, 1544, 1545, 1546, 1555, 1557, 1562, 1565, 1572, 1578, 1580, 1581, 1587, 1592, 1593, 1605, 1606, 1608, 1609, 1611, 1618, 1620, 1621, 1622, 1625, 1626, 1628, 1629, 1631, 1633, 1635, 1636]
# print("Word_id", word_id)
# print("Length of word_id", len(word_id))

concreteness =  [4.57, 4.54, 3.61, 4.96, 4.87, 4.81, 4.86, 5, 4.87, 4.96, 4.7, 3.32, 4.79, 3.34, 4.26, 4.14, 5, 4.19, 4.9, 4.93, 4.92, 4.9, 4.78, 4.43, 4, 4.59, 4.86, 4.89, 4.92, 4.72, 4.63, 4.68, 4.74, 4.8, 5, 4.89, 4.68, 4.72, 4.93, 5, 4.25, 4.96, 4.77, 4.57, 4.79, 4.74, 4.9, 4.59, 5, 4.44, 4.9, 4.81, 4.89, 4.83, 4.86, 4.43, 4.43, 4.6, 4.96, 5, 4.18, 4.83, 4.97, 4.04, 4.44, 4.65, 4.75, 4.92, 4.96, 4.93, 4.68, 4.83, 4.81, 3.03, 4.86, 5, 4.92, 4.89, 4.86, 4.44, 4.64, 4.85, 4.68, 4.82, 4.6, 4.43, 4.24, 4.93, 4.78, 4.97, 4.9, 4.93, 4.21, 4.89, 4.53, 4.5, 5, 4.76, 4.54, 5, 4.4, 4.83, 4.61, 4.81, 4.89, 4.62, 3.89, 4.67, 4.66, 4.35, 4.15, 4.11, 4.32, 4.9, 4.4, 4.57, 4.85, 4.71, 4.17, 4.04, 3.82, 3.7, 4.72, 4.9, 4.61, 4.87, 4.07, 4.43, 4.86, 4.77, 3.55, 4.44, 4.81, 4.5, 4.79, 4.82, 4.85, 4.48, 5, 4.75, 4.79, 4.61, 4.79, 4.77, 4.93, 4.85, 4.82, 4.69, 4.96, 5, 3.54, 4.41, 4.96, 4.39, 4.6, 4.93, 4.96, 4.4, 4.76, 4.71, 4.48, 4.4, 5, 5, 1.58, 5, 3, 4.3, 2.85, 5, 4.54, 4.57, 4.81, 4.71, 5, 4.8, 4.68, 4.79, 5, 4.79, 3.81, 4.59, 4.26, 5, 5, 4.9, 4.73, 4.9, 4.76, 4.97, 4.03, 4.3, 4.04, 4.56, 4.87, 3.88, 3.07, 4.81, 4.59, 3.92, 3.93, 4.69, 4.73, 4.89, 4.88, 4.72, 3.82, 4.56, 4.85, 4.82, 4.59, 4.97, 4.38, 5, 4.56, 5, 4.61, 4.86, 3.17, 4.77, 4.04, 4.9, 4.85, 4.21, 4.72, 4.93, 4.85, 4.93, 4.93, 5, 4.52, 4.54, 4.92, 3.07, 4.72, 4.53, 4.88, 4.88, 4.79, 4.96, 5, 4.12, 4.48, 4.93, 4.11, 4.96, 4.66, 3.63, 4.73, 4.93, 4.19, 4.96, 4.41, 4.82, 5, 3, 4.93, 4.63, 3.75, 4.5, 4.66, 4.64, 3, 5, 4.96, 4.97, 4.92, 4.9, 4.9, 4.46, 4.85, 5, 4.33, 4.5, 4.88, 4.97, 4.56, 4.5, 4.82, 3.89, 4.83, 5, 4.97, 4.59, 4.69, 4.96, 4.9, 4.68, 4.68, 4, 4.32, 5, 3.68, 4.83, 4.56, 4.31, 4.25, 5, 4.57, 4.59, 4.46, 4.25, 4.62, 4.7, 4.48, 4.48, 4.85, 4.96, 4.14, 5, 4.9, 4.89, 3.97, 4.57, 4.92, 4.83, 3.15, 4.54, 3.72, 4.97, 4.93, 4.84, 4.78, 4.72, 4.93, 4.93, 4.96, 2.69, 4.9, 4.1, 4.92, 4.21, 4.39, 4.93, 4.5, 4.93, 4.86, 4.66, 4.92, 4.61, 4.12, 3.61, 4.86, 4.85, 3.5, 4.72, 4.52, 4.8, 4.93, 4.57, 4.93, 4.93, 4.56, 5, 4.77, 3.53, 3.8, 4.86, 4.97, 2.5, 3.86, 4.9, 4.89, 4.86, 4.87, 4.44, 5, 4.59, 4.1, 4.66, 3.43, 4.9, 4.83, 4.52, 4.71, 4.4, 4.67, 4.77, 4.89, 4.81, 5, 4.23, 4.59, 4.77, 4.77, 4.44, 4.93, 3.3, 4.68, 4.36, 4.27, 4.79, 5, 4.79, 4.9, 4.73, 4.37, 4.76, 4.7, 4.5, 4.44, 4.72, 3, 4.9, 4.67, 4.55, 4.78, 4.65, 4.43, 4.45, 4.93, 4.26, 4.87, 5, 4.07, 4.9, 3.07, 4.86, 4.15, 3.3, 3.92, 4.85, 4.89, 4.31, 4.61, 4.65, 4.73, 4.43, 4.75, 4.79, 4.52, 4.85, 4.97, 4.81, 4.89, 4.9, 4.88, 4.61, 4.86, 4.68, 4.97, 4.85, 4.55, 4.1, 4.79, 4.83, 5, 4.63, 4.64, 4.55, 4.93, 4.96, 4.64, 4.5, 4.94, 4.41, 4.82, 4.93, 4.97, 4.92, 4.37, 4.96, 4.7, 4, 4.56, 4.73, 4.82, 4.48, 4.48, 4.07, 4.64, 4.14, 4.36, 4.93, 5, 4.97, 4.89, 3.54, 4.85, 4.83, 4.97, 5, 4.62, 4.96, 3.85, 4.21, 4.72, 4.7, 4.34, 4.69, 4.93, 4.67, 4.89, 4.72, 4.96, 4.5, 4.92, 4.86, 4.97, 4.21, 4.69, 4.54, 4.63, 4.08, 2.59, 4.96, 4.77, 4.07, 4.93, 4.9, 4.82, 3.27, 4.93, 4.52, 4.53, 4.7, 4.37, 4.83, 4.64, 4.68, 4.9, 4.71, 4.87, 4.59, 5, 3.77, 3.79, 4.46, 4.14, 4.9, 4.72, 4.84, 4.86, 4.82, 4.89, 4.79, 5, 4.68, 4.96, 4.75, 4.44, 4.41, 4.69, 4.27, 4.24, 3.46, 4.72, 4.83, 4.44, 4.54, 3.59, 2.12, 3.48, 4.89, 4.67, 4.56, 4.67, 4.7, 4.96, 4.89, 4.42, 4.33, 4.13, 4.86, 4.07, 4.46, 4.59, 4.36, 4.93, 4.71, 4.93, 4.93, 4.11, 4.75, 4.52, 3.77]
# print("Concreteness", concreteness)
# print("Length of concreteness", len(concreteness))

word_freq = [785, 254, 766, 73, 932, 185, 35, 315, 131, 1860, 1933, 475, 40, 187, 522, 169, 3281, 78, 288, 117, 58, 6, 2333, 129, 67, 174, 253, 224, 31, 21, 298, 37, 273, 305, 149, 16, 26, 96, 205, 22, 48, 156, 53, 1525, 5243, 75, 704, 80, 33, 65, 961, 291, 774, 498, 144, 74, 1531, 67, 237, 36, 15, 229, 19, 289, 44, 87, 144, 485, 342, 146, 211, 113, 149, 81, 231, 99, 8, 33, 35, 21, 568, 463, 183, 285, 372, 83, 238, 778, 7645, 2, 2844, 882, 1, 13, 19, 94, 637, 2225, 536, 35, 147, 31, 87, 1644, 132, 1380, 1610, 26, 82, 272, 538, 30, 266, 8, 39, 367, 555, 161, 6036, 785, 11914, 400, 84, 80, 85, 13, 619, 33, 19, 520, 786, 400, 412, 50, 270, 346, 54, 10, 531, 146, 6, 37, 1797, 50, 115, 184, 47, 16, 30, 162, 48, 72, 20, 135, 485, 1332, 83, 109, 1414, 746, 344, 762, 8, 129, 584, 280, 381, 2784, 331, 52, 1017, 1044, 28, 1546, 867, 20, 148, 168, 77, 76, 240, 930, 11, 476, 44, 1753, 573, 438, 1219, 177, 46, 442, 1779, 1, 61, 19, 3087, 513, 94, 117, 50, 166, 1954, 115, 12, 13, 205, 560, 4944, 2246, 184, 82, 6, 37, 270, 185, 106, 49, 3, 3084, 658, 102, 10, 29, 7889, 137, 43, 28, 67, 13, 2597, 178, 166, 545, 285, 12, 371, 77, 487, 10, 1518, 164, 33, 5113, 2405, 41, 42, 86, 8, 372, 106, 1209, 350, 541, 229, 8, 172, 318, 732, 6, 254, 521, 96, 26, 84, 1890, 42, 15, 635, 223, 186, 238, 1219, 56, 718, 381, 50, 31, 63, 1227, 1137, 118, 115, 253, 120, 152, 122, 242, 47, 222, 46, 5368, 430, 201, 25, 1381, 1469, 24, 3, 116, 56, 183, 37, 2374, 50, 166, 213, 237, 775, 170, 1280, 99, 1214, 34, 1792, 465, 68, 7226, 262, 38, 125, 465, 6, 198, 84, 84, 41, 168, 45, 46, 137, 495, 556, 4460, 46, 1, 170, 258, 44, 68, 182, 8, 79, 64, 62, 281, 31, 11, 48, 761, 50, 3128, 1266, 45, 118, 459, 655, 36, 129, 707, 161, 47, 49, 85, 7, 22, 69, 117, 34, 116, 70, 466, 184, 1905, 38, 31, 235, 11, 254, 56, 28, 49, 209, 656, 211, 61, 26, 20, 1003, 305, 204, 3694, 14, 150, 335, 20, 32, 327, 58, 76, 592, 217, 55, 198, 37, 617, 87, 7, 911, 889, 189, 139, 11, 57, 47, 139, 74, 122, 1077, 99, 1860, 286, 1938, 42, 207, 67, 142, 503, 10, 321, 67, 148, 281, 108, 629, 184, 103, 17, 18, 14, 142, 79, 153, 15, 373, 9, 18, 88, 892, 12, 246, 246, 358, 49, 812, 2, 195, 1196, 57, 66, 26, 104, 440, 1464, 122, 2, 351, 153, 35, 330, 38, 32, 58, 46, 251, 148, 8, 2259, 47, 15, 73, 107, 56, 201, 50, 231, 24, 183, 73, 952, 273, 35, 705, 737, 294, 661, 3587, 37, 220, 241, 51, 165, 480, 543, 133, 76, 41, 315, 237, 3645, 48, 39, 509, 1490, 416, 42, 120, 182, 175, 58, 11, 29, 71, 333, 123, 49, 145, 94, 416, 11, 280, 444, 5, 237, 173, 49, 20, 0, 29, 44, 8, 38, 29, 25, 1061, 9, 885, 87, 206, 108, 497, 17, 167, 143, 299, 138, 160, 30, 43, 41, 140, 254, 3777, 2372, 278, 6072, 3020, 13345, 47, 358, 2, 83, 24, 47, 12, 24]
# print("Word Freq", word_freq)
# print("Length of Word Freq", len(word_freq))

word_length = [5, 7, 5, 8, 7, 5, 6, 5, 5, 3, 4, 4, 5, 4, 6, 5, 4, 8, 5, 5, 7, 5, 4, 6, 7, 6, 6, 8, 7, 6, 5, 6, 4, 5, 4, 6, 5, 10, 5, 7, 8, 6, 9, 5, 4, 7, 3, 9, 6, 5, 6, 6, 6, 5, 9, 5, 7, 6, 6, 3, 5, 6, 5, 6, 7, 7, 7, 5, 4, 5, 5, 5, 6, 7, 8, 6, 6, 7, 6, 7, 6, 7, 6, 9, 6, 9, 7, 5, 5, 8, 6, 5, 6, 4, 5, 7, 5, 7, 5, 5, 8, 6, 3, 6, 4, 7, 7, 5, 7, 7, 8, 7, 4, 8, 5, 7, 7, 5, 7, 6, 6, 6, 6, 4, 6, 6, 8, 7, 4, 7, 6, 5, 5, 6, 8, 7, 7, 7, 5, 6, 8, 9, 8, 5, 7, 4, 5, 5, 7, 6, 5, 4, 8, 6, 7, 5, 7, 5, 5, 6, 4, 4, 7, 5, 5, 5, 6, 6, 6, 7, 6, 6, 6, 4, 6, 7, 9, 4, 10, 5, 5, 5, 7, 6, 5, 4, 8, 8, 6, 3, 9, 5, 6, 7, 7, 5, 6, 5, 6, 6, 8, 7, 6, 6, 5, 7, 6, 4, 4, 5, 5, 5, 6, 5, 5, 6, 6, 5, 7, 6, 5, 6, 7, 6, 4, 7, 4, 7, 4, 8, 5, 5, 6, 4, 7, 5, 5, 4, 4, 6, 5, 7, 5, 5, 7, 7, 5, 4, 5, 6, 6, 6, 4, 5, 5, 5, 5, 7, 5, 7, 6, 4, 6, 7, 6, 7, 7, 8, 5, 5, 4, 6, 4, 6, 4, 4, 5, 5, 4, 6, 3, 7, 7, 9, 4, 4, 8, 5, 6, 5, 4, 6, 5, 7, 6, 5, 7, 7, 7, 6, 5, 6, 6, 6, 6, 4, 5, 4, 5, 8, 4, 5, 7, 6, 4, 7, 6, 5, 7, 3, 5, 5, 6, 5, 7, 6, 8, 7, 9, 5, 8, 5, 5, 6, 8, 6, 5, 6, 6, 8, 6, 6, 2, 6, 5, 7, 7, 6, 4, 6, 7, 5, 6, 6, 7, 7, 7, 5, 6, 7, 6, 5, 6, 6, 5, 5, 7, 6, 5, 7, 6, 5, 6, 7, 6, 7, 5, 6, 6, 6, 5, 5, 7, 5, 10, 5, 6, 5, 6, 4, 6, 6, 7, 4, 8, 6, 7, 6, 8, 7, 6, 8, 6, 7, 6, 5, 5, 5, 7, 5, 6, 6, 6, 4, 6, 5, 5, 7, 6, 6, 6, 5, 5, 6, 5, 5, 6, 3, 7, 3, 4, 6, 5, 6, 4, 8, 7, 7, 7, 9, 5, 8, 6, 8, 9, 7, 7, 4, 7, 6, 5, 5, 7, 7, 5, 9, 6, 8, 6, 5, 7, 8, 4, 6, 6, 7, 5, 5, 5, 5, 4, 4, 5, 5, 5, 4, 8, 5, 7, 6, 6, 6, 5, 5, 6, 5, 8, 5, 6, 4, 6, 7, 7, 5, 5, 6, 7, 6, 8, 6, 7, 6, 6, 6, 7, 5, 7, 6, 5, 5, 6, 4, 4, 7, 6, 7, 5, 6, 6, 4, 7, 9, 8, 7, 7, 10, 5, 6, 5, 8, 5, 5, 4, 6, 6, 6, 6, 4, 8, 4, 7, 6, 5, 6, 5, 7, 6, 5, 6, 5, 6, 6, 5, 5, 6, 8, 8, 6, 4, 8, 7, 5, 4, 6, 7, 5, 6, 5, 6, 5, 5, 5, 4, 4, 5, 6]
# print("Word length", word_length)
# print("Length of Word Length", len(word_length))

# Dictionaries to match word id's with their corresponding concreteness, word_freq, and word_length values (e.g., keys are the word ids)
concreteness_dict = {}
for i in range(len(word_id)):
    concreteness_dict[word_id[i]] = concreteness[i]
# print("Length of conc dic", len(concreteness_dict))

word_freq_dict = {}
for i in range(len(word_id)):
    word_freq_dict[word_id[i]] = word_freq[i]
# print("Length of word freq dic", len(word_freq_dict))

word_len_dict = {}
for i in range(len(word_id)):
    word_len_dict[word_id[i]] = word_length[i]
# print("Length of word len dic", len(word_len_dict))

# Printing dictionaries to see the matches
# print("concreteness_dict", concreteness_dict)
# print("word_freq_dict", word_freq_dict)
# print("word_len_dict", word_len_dict)

def one_participant_concreteness_list(presented_items):
    """This is the loop to find one participant's average
    concreteness values per each list:
    Output will be N participants * 552 lists"""

    mean_conc_participant = []

    # For each list in the presented items
    for single_list in presented_items:
        this_list_conc = []
        # print("Single list:", single_list)
        # For each word in the list
        for item in single_list:
            # Find each word's concreteness value
            item_conc = concreteness_dict.get(item)
            # print("Item conc", item_conc)
            # Apeend the values to a new this_list_corr list
            this_list_conc.append(item_conc)
        # print("This list conc", this_list_conc)
        # print("length of this_list_conc", len(this_list_conc))
        # Find the average conc value in each list
        mean_this_list = np.mean(this_list_conc)
        #print(mean_this_list)
        # Get the average of the values in each list to find aver. similarity value for each list
        # Append them to a mean_lists_participant list
        mean_conc_participant.append(mean_this_list)
    # print("mean conc list", mean_conc_participant)
    # print("Length of this part conc of lists", len(mean_conc_participant))
    return mean_conc_participant


def one_participant_wordfreq_list(presented_items):
    """This is the loop to find one participant's average
    word frequency values per each list:
    Output will be N participants * 552 lists"""

    mean_wordfreq_participant = []

    # For each list in the presented items
    for single_list in presented_items:
        this_list_wordfreq = []
        # print("Single list:", single_list)
        # For each word in the list
        for item in single_list:
            # Find each word's frequency value
            item_wordfreq = word_freq_dict.get(item)
            # print("Item conc", item_conc)
            # Apeend the values to a new this_list_corr list
            this_list_wordfreq.append(item_wordfreq)
        # print("This list conc", this_list_conc)
        # print("length of this_list_conc", len(this_list_conc))
        # Find the average word freq value in each list
        mean_this_list = np.mean(this_list_wordfreq)
        #print(mean_this_list)
        # Get the average of the values in each list to find aver. similarity value for each list
        # Append them to a mean_lists_participant list
        mean_wordfreq_participant.append(mean_this_list)
    # print("mean wordfreq list", mean_wordfreq_participant)
    # print("Length of this part wordfreq of lists", len(mean_wordfreq_participant))
    return mean_wordfreq_participant


def one_participant_wordlen_list(presented_items):
    """This is the loop to find one participant's average
    word length per each list:
    Output will be N participants * 552 lists"""

    mean_wordlen_participant = []

    # For each list in the presented items
    for single_list in presented_items:
        this_list_wordlen = []
        # print("Single list:", single_list)
        # For each word in the list
        for item in single_list:
            # Find each word's len
            item_wordlen = word_len_dict.get(item)
            # print("Item conc", item_conc)
            # Apeend the values to a new this_list_corr list
            this_list_wordlen.append(item_wordlen)
        # print("This list word len", this_list_wordlen)
        # print("length of this_list_wordlen", len(this_list_wordlen))
        # Find the average word len in each list
        mean_this_list = np.mean(this_list_wordlen)
        #print(mean_this_list)
        # Get the average of the values in each list to find aver. similarity value for each list
        # Append them to a mean_lists_participant list
        mean_wordlen_participant.append(mean_this_list)
    # print("mean word len list", mean_wordlen_participant)
    # print("Length of this part word len of lists", len(mean_wordlen_participant))
    return mean_wordlen_participant

def measures_by_list(files_ltpFR2):
    """This is the loop to find all participants' average concreteness, word freq, word len's per list"""
    all_mean_conc_participants = []
    all_mean_wordfreq_participants = []
    all_mean_wordlen_participants = []

    for f in files_ltpFR2:
        # Read in data
        test_mat_file = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)

        # Skip if there is no data for a participant
        if isinstance(test_mat_file['data'], np.ndarray):
            # print('Skipping...')
            continue

        # Get the session matrix for the participant
        session_mat = test_mat_file['data'].session
        # print(len(session_mat))
        # print(np.bincount(session_mat))

        # Skip if the participant did not finish
        if len(session_mat) < 576:
            # print('Skipping because participant did not finish...')
            continue

        # Get the presented items matrix for the participant
        else:
            print(f)
            pres_mat = test_mat_file['data'].pres_itemnos.astype('int16')

        # Get just the lists in sessions 1-23rd sessions
        pres_mat = pres_mat[np.where(session_mat != 24)]

        # Add each participants' values
        mean_conc_participant = one_participant_concreteness_list(pres_mat)
        mean_wordfreq_participant = one_participant_wordfreq_list(pres_mat)
        mean_wordlen_participant = one_participant_wordlen_list(pres_mat)

        # Add each participants' values to correct all_mean lists
        all_mean_conc_participants.append(mean_conc_participant)
        all_mean_wordfreq_participants.append(mean_wordfreq_participant)
        all_mean_wordlen_participants.append(mean_wordlen_participant)

    return all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants


if __name__ == "__main__":
    # Get the files
    files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP*.mat')
    all_mean_conc_participants, all_mean_wordfreq_participants, all_mean_wordlen_participants = measures_by_list(files_ltpFR2)
    print("ALl mean conc participants:", all_mean_conc_participants)
    print("Shape of all_mean_conc", np.shape(np.array(all_mean_conc_participants)))
    print(" ")
    print("All mean word freq participants:", all_mean_wordfreq_participants)
    print("Shape of all_mean_word freq", np.shape(np.array(all_mean_wordfreq_participants)))
    print(" ")
    print("All mean word len participants:", all_mean_wordlen_participants)
    print("Shape of all_mean_wordlen", np.shape(np.array(all_mean_wordlen_participants)))
