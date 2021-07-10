# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:53:26 2020

@author: pariya pourmohammadi
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('Data')
elev = np.genfromtxt('elev.txt', delimiter=' ', skip_header=6)

elev_patches = np.array(
    [elev[i:i + 32, j:j + 32] for i in range(0, elev.shape[0], int(32)) for j
     in range(0, elev.shape[1], int(32))])
exc_ind = []
for i in range(elev_patches.shape[0]):
    if (np.sum(elev_patches[i]) == -10238976.0):
        exc_ind.append(i)

np.random.seed(1364)
# generate random indices for the partitioned data
range_ind = np.arange(52455)

range_ind = [x for x in range_ind if x not in exc_ind]

np.random.shuffle(range_ind)
# ind_train= range_ind[:int(len(range_ind)*.66)]
# ind_test= range_ind[int(len(range_ind)*.66):]
train = np.load('y_patches.npy')
y_train = train[range_ind[0:int(len(range_ind) * .66)], :, :]
y_test = train[range_ind[int(len(range_ind) * .66):], :, :]

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('wr', ["w", "r"], N=256)
cmap1 = LinearSegmentedColormap.from_list('wg', ["w", "g"], N=256)


def texmaker(target, name, fmt, flag_train, flag_file):
    mask = np.zeros(shape=(8608, 6240), dtype=float)
    blocks_in_row = 6240 / 32

    count = 0
    for i in range(y_train.shape[0]):
        #    print(i)
        ind = range_ind[i]
        row = (int(ind / blocks_in_row)) * 32
        col = (int(ind % blocks_in_row)) * 32
        if flag_train:
            if len(target.shape) == 4:
                mask[row:(row + 32), col:(col + 32)] = y_train[i, :, :]
                count += 1

            if len(target.shape) == 5:
                mask[row:(row + 32), col:(col + 32)] = y_train[i, :, :]

        else:
            mask[row:(row + 32), col:(col + 32)] = np.zeros((32, 32))

    #    print(count)

    for i in range(y_test.shape[0]):
        #    print(i)
        ind = range_ind[i + y_train.shape[0]]
        row = (int(ind / blocks_in_row)) * 32
        col = (int(ind % blocks_in_row)) * 32

        if len(target.shape) == 4:
            mask[row:(row + 32), col:(col + 32)] = target[i, 0, :, :]

        if len(target.shape) == 5:
            mask[row:(row + 32), col:(col + 32)] = target[i, 0, :, :, 0]

        if len(target.shape) == 3:
            mask[row:(row + 32), col:(col + 32)] = target[i, :, :]
        #        print(i)

    mask = mask[:8581, :6220]
    # np.savetxt("mask.txt",mask, fmt='%d', delimiter=' ')
    if flag_file:
        mask[elev == -9999] = int(-9999)
        all_lines = open("Lines.txt", "r")
        lines = all_lines.readlines()
        all_lines.close()

        header = ''
        for line in lines: header = header + line
        np.savetxt(name, mask, fmt=fmt, delimiter=' ', header=header,
                   comments='')

    return mask


def visualize(t):
    blocks_row = 100
    shape = np.zeros(shape=(2400, 3200), dtype=float)
    for i in range(y_test.shape[0]):
        #    print(i)
        row = (int(i / blocks_row)) * 32
        col = (int(i % blocks_row)) * 32

        if len(t.shape) == 4:
            shape[row:(row + 32), col:(col + 32)] = t[i, 0, :, :]
            print(np.sum(t[i, 0, :, :]))

        if len(t.shape) == 5:
            shape[row:(row + 32), col:(col + 32)] = t[i, 0, :, :, 0]

        if len(t.shape) == 3:
            shape[row:(row + 32), col:(col + 32)] = t[i, :, :]

        #            print(i)

    plt.imshow(shape)

    return shape


def moving_average(shape, n):
    from scipy import ndimage
    density = ndimage.uniform_filter(shape, size=n, mode='constant')
    return density


# this function finds the misses (FN/Type II Error)
def miss(res, ground_t):
    shape = np.zeros(shape=(8581, 6220), dtype=float)
    shape[(res == 0) & (ground_t == 1)] = 1.0

    return (shape)


# this function finds the hits (TP)
def hit(res, ground_t):
    shape = np.zeros(shape=(8581, 6220), dtype=float)
    shape[(res == 1) & (ground_t == 1)] = 1.0
    return (shape)


# this function finds the false alarms (FP/ Type I error)
def false_alarm(res, ground_t):
    shape = np.zeros(shape=(8581, 6220), dtype=float)
    shape[(res == 1) & (ground_t == 0)] = 1.0
    return (shape)


def dice_coef(input1, input2):
    a = hit(input1, input2)
    b = false_alarm(input1, input2)
    c = miss(input1, input2)
    dice_coef = 2 * np.sum(a) / (2 * np.sum(a) + np.sum(b) + np.sum(c))

    return (dice_coef)


def wrt_file(mask, name, fmt):
    mask[elev == -9999] = int(-9999)
    all_lines = open("Lines.txt", "r")
    lines = all_lines.readlines()
    all_lines.close()

    header = ''
    for line in lines: header = header + line
    np.savetxt(name, mask, fmt=fmt, delimiter=' ', header=header, comments='')


def auc(inpu1, inpu2):
    TP = np.sum(hit(inpu1, inpu2))
    FN = np.sum(miss(inpu1, inpu2))
    FP = np.sum(false_alarm(inpu1, inpu2))
    TN = 7407 * 32 * 32 - (TP + FP + FN)

    Recall = TP / (TP + FN)
    Percision = TP / (TP + FP)
    Specificity = TN / (TN + FP)
    FPR = 1 - Specificity

    outAUC = {'Recall': Recall, 'Percision': Percision,
              'Specificity': Specificity, 'FPR': FPR}

    return (outAUC)


labeledData = ['y_classdeepLandS.npy','y_classdeepLandU.npy',
               'y_classUnet.npy', 'y_class3DdeepLand.npy',
               'y_class3DdeepLandv.npy',
               'y_classdeepLandF.npy', 'y_classdeepLandF2.npy',
               'y_classdeepLandF3.npy', 'y_classdeepLand3Df.npy',
               'yclass_Total_RF.npy']

scoreData = ['y_predictdeepLandS.npy',
             'y_predictdeepLandU.npy','y_predictUnet.npy',
             'y_predict3DdeepLand.npy', 'y_predict3DdeepLandv.npy',
             'y_predictdeepLandF.npy', 'y_predictdeepLandF2.npy',
             'y_predictdeepLandF3.npy','y_predictdeepLand3Df.npy',
             'yPredict_Total_RF.npy']

for file1 in scoreData:
    res1 = np.load(file1)
    texmaker(res1, '../results/' + file1[2:-4] + '.txt', '%.6f', False, True)

for file in labeledData:
    res = np.load(file)
    texmaker(res, '../results/' + file[2:-4] + '.txt', '%d', False, True)

groundTruth = texmaker(y_test, 'y_test.txt', '%d', False, False)
L1 = texmaker(np.load(labeledData[0]),
              '../results/' + labeledData[0][2:-4] + '.txt', '%d', False,
              False)
L2 = texmaker(np.load(labeledData[1]),
              '../results/' + labeledData[1][2:-4] + '.txt', '%d', False,
              False)
L3 = texmaker(np.load(labeledData[2]),
              '../results/' + labeledData[2][2:-4] + '.txt', '%d', False,
              False)
L4 = texmaker(np.load(labeledData[3]),
              '../results/' + labeledData[3][2:-4] + '.txt', '%d', False,
              False)
L5 = texmaker(np.load(labeledData[4]),
              '../results/' + labeledData[4][2:-4] + '.txt', '%d', False,
              False)
L6 = texmaker(np.load(labeledData[5]),
              '../results/' + labeledData[5][2:-4] + '.txt', '%d', False,
              False)
L7 = texmaker(np.load(labeledData[6]),
              '../results/' + labeledData[6][2:-4] + '.txt', '%d', False,
              False)
L8 = texmaker(np.load(labeledData[7]),
              '../results/' + labeledData[7][2:-4] + '.txt', '%d', False,
              False)
L9 = texmaker(np.load(labeledData[8]),
              '../results/' + labeledData[8][2:-4] + '.txt', '%d', False,
              False)

L10 = texmaker(np.load(labeledData[9]),
               '../results/' + labeledData[9][1:-4] + '.txt', '%d', False,
               False)
# L11 = texmaker(np.load(labeledData[10]),'../results/'+labeledData[10][2:-4]+'.txt', '%d', False, False)
sumLab = L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8 + L9

# sumLab[sumLab < 0] = 0
sumLab[sumLab <= 4.5] = 0
sumLab[sumLab > 4.5] = 1

sumLab_NoL = L1 + L2 + L3 +  L7 + L8 + L9
sumLab_NoL[sumLab_NoL <= 3] = 0
sumLab_NoL[sumLab_NoL > 3] = 1


dice_voting = dice_coef(sumLab, groundTruth)
dice_voting_noL = dice_coef(sumLab_NoL, groundTruth)
auc_Sum_noL = auc(sumLab_NoL, groundTruth)

groundTruth = texmaker(y_test, 'y_test.txt', '%d', False, False)
S1 = texmaker(np.load(scoreData[0]),
              '../results/' + scoreData[0][2:-4] + '.txt', '%.6f', False,
              False)
S2 = texmaker(np.load(scoreData[1]),
              '../results/' + scoreData[1][2:-4] + '.txt', '%.6f', False,
              False)
S3 = texmaker(np.load(scoreData[2]),
              '../results/' + scoreData[2][2:-4] + '.txt', '%.6f', False,
              False)
S4 = texmaker(np.load(scoreData[3]),
              '../results/' + scoreData[3][2:-4] + '.txt', '%.6f', False,
              False)
S5 = texmaker(np.load(scoreData[4]),
              '../results/' + scoreData[4][2:-4] + '.txt', '%.6f', False,
              False)
S6 = texmaker(np.load(scoreData[5]),
              '../results/' + scoreData[5][2:-4] + '.txt', '%.6f', False,
              False)
S7 = texmaker(np.load(scoreData[6]),
              '../results/' + scoreData[6][2:-4] + '.txt', '%.6f', False,
              False)
S8 = texmaker(np.load(scoreData[7]),
              '../results/' + scoreData[7][2:-4] + '.txt', '%.6f', False,
              False)
S9 = texmaker(np.load(scoreData[8]),
              '../results/' + scoreData[8][2:-4] + '.txt', '%.6f', False,
              False)

S10 = texmaker(np.load(scoreData[9]),
               '../results/' + scoreData[9][2:-4] + '.txt', '%.6f', False,
               False)


avgScore = (S1 + S2 + S3 + S4 + S5 + S6 + S7 + S8 + S9) / 9

avgScoreL = np.zeros(shape=avgScore.shape)
avgScoreL[avgScore < .5] = 0
avgScoreL[avgScore > .5] = 1

dice_Mean = dice_coef(avgScoreL, groundTruth)

avgScore_no3D = (S1 + S2 + S3 +S6 + S7 + S8 ) / 6

avgScore_no3DL = np.zeros(shape=avgScore.shape)
avgScore_no3DL[avgScore_no3D < .5] = 0
avgScore_no3DL[avgScore_no3D > .5] = 1

dice_Mean_no3DL = dice_coef(avgScore_no3DL, groundTruth)
auc_no3DL = auc(avgScore_no3DL, groundTruth)


wrt_file(sumLab, '../results/voting.txt', '%d')
wrt_file(avgScore, '../results/avg_scr.txt', '%.6f')
wrt_file(avgScoreL, '../results/avg_scrLab.txt', '%d')

deepLandS_AUC = auc(L1, groundTruth)
deepLandU_AUC = auc(L2, groundTruth)
deepLandF_AUC = auc(L3, groundTruth)
deepLandF2_AUC = auc(L4, groundTruth)
deepLandF3_AUC = auc(L5, groundTruth)
Unet_AUC = auc(L6, groundTruth)
deepLand3D_AUC = auc(L7, groundTruth)
deepLand3Dv_AUC = auc(L8, groundTruth)
deepLand3Df_AUC = auc(L9, groundTruth)
RF_AUC = auc(L10, groundTruth)

sumLab_AUC = auc(sumLab, groundTruth)
avgScoreL_AUC = auc(avgScoreL, groundTruth)

dice_matrix = np.zeros((9, 9), dtype=float)

for i in range(9):
    for j in range(9):
        dice_matrix[i][j] = dice_coef(globals()['L' + str(i + 1)],
                                      globals()['L' + str(j + 1)])
        print(str(j) + ' ' + str(i))

# import seaborn as sns
import pandas as pd
df = pd.DataFrame(dice_matrix, columns=["M1", "M2", "M3", "M4", "M5", "M6",
                                        "M7", "M8", "M9"],
                                index=["M1", "M2", "M3", "M4", "M5", "M6",
                                        "M7", "M8", "M9"])

sns.set_theme()
sns.heatmap(df, annot=True, fmt='.3', cmap='YlGnBu', linewidths=.5)








