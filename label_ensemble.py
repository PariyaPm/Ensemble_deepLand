'''
this is an ensemble model to train the ultimate output based on probability
 output of deeplandS deeplandU and deepLandFusion
in this model I used 5-fold cross validation and I fif parameter fine-tuning
to find the best treedepth and #of estimators
'''

import numpy as np
import sklearn
import os
from numpy import interp

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import cohen_kappa_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, cohen_kappa_score

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

os.chdir('../Data')

w1_train = np.load('y_predictdeepLandF2_train.npy')
w2_train = np.load('y_predictdeepLandF3_train.npy')
w3_train = np.load('y_class3DdeepLand_train.npy')
w4_train = np.load('y_classdeepLand3Df_train.npy')
w5_train = np.load('y_class3DdeepLandv_train.npy')
w6_train = np.load('y_predictUnet_train.npy')
w7_train = np.load('y_classdeepLandS_train.npy')
w8_train = np.load('y_classdeepLandU_train.npy')
w9_train = np.load('y_classdeepLandF_train.npy')

w1 = np.load('y_predictdeepLandF2.npy')
w2 = np.load('y_predictdeepLandF3.npy')
w3 = np.load('y_class3DdeepLand.npy')
w4 = np.load('y_classdeepLand3Df.npy')
w5 = np.load('y_class3DdeepLandv.npy')
w6 = np.load('y_predictUne.npy')
w7 = np.load('y_classdeepLandS.npy')
w8 = np.load('y_classdeepLandU.npy')
w9 = np.load('y_classdeepLandF.npy')

labels = np.load('y_patches.npy')

elev = np.genfromtxt('elev.txt', delimiter=' ', skip_header=6)

elev_patches = np.array(
    [elev[i:i + 32, j:j + 32] for i in range(0, elev.shape[0], int(32)) for j in
     range(0, elev.shape[1], int(32))])
exc_ind = []
for i in range(elev_patches.shape[0]):
    if np.sum(elev_patches[i]) == -10238976.0:
        exc_ind.append(i)

np.random.seed(1364)
# generate random indices for the partitioned data
range_ind = np.arange(52455)

range_ind = [x for x in range_ind if x not in exc_ind]

np.random.shuffle(range_ind)

data_train = np.column_stack([
    w1_train.flatten(),
    w2_train.flatten(),
    w3_train.flatten(),
    w4_train.flatten(),
    w5_train.flatten(),
    w6_train.flatten(),
    w7_train.flatten(),
    w8_train.flatten(),
    w9_train.flatten()])

dat_test = np.column_stack([
    w1.flatten(),
    w2.flatten(),
    w3.flatten(),
    w4.flatten(),
    w5.flatten(),
    w6.flatten(),
    w7.flatten(),
    w8.flatten(),
    w9.flatten()])

y_train = labels[range_ind[0:int(len(range_ind) * .66)], :, :].flatten()
y_test = labels[range_ind[int(len(range_ind) * .66):], :, :].flatten()
label_train = y_train

xTrainValid = data_train
yTrainValid = y_train
data_test = dat_test
label_test = y_test

def auc(inpu1, inpu2):
    TP = np.sum(hit(inpu1, inpu2))
    FN = np.sum(miss(inpu1, inpu2))
    FP = np.sum(false_alarm(inpu1, inpu2))
    TN = inpu1.shape[0] - (TP + FP + FN)

    Recall = TP / (TP + FN)
    Percision = TP / (TP + FP)
    Specificity = TN / (TN + FP)
    FPR = 1 - Specificity

    outAUC = {'Recall': Recall, 'Percision': Percision,
              'Specificity': Specificity, 'FPR': FPR}

    return (outAUC)


# this function finds the misses (FN/Type II Error)
def miss(res, ground_t):
    shape = np.zeros(shape=res.shape, dtype=float)
    shape[(res == 0) & (ground_t == 1)] = 1.0

    return shape


# this function finds the hits (TP)
def hit(res, ground_t):
    shape = np.zeros(shape=res.shape, dtype=float)
    shape[(res == 1) & (ground_t == 1)] = 1.0
    return shape


# this function finds the false alarms (FP/ Type I error)
def false_alarm(res, ground_t):
    shape = np.zeros(shape=res.shape, dtype=float)
    shape[(res == 1) & (ground_t == 0)] = 1.0
    return shape


def dice_coef(input1, input2):
    a = hit(input1, input2)
    b = false_alarm(input1, input2)
    c = miss(input1, input2)
    dice_coef = 2 * np.sum(a) / (2 * np.sum(a) + np.sum(b) + np.sum(c))

    return dice_coef


estimators = np.arange(400, 1001, 200)
depths = np.arange(35, 50, 3)

k = 5


def rf(est, dpt, X, Y, Xtest, Ytest):
    RF = ExtraTreesClassifier(n_estimators=est,
                              max_depth=dpt,
                              class_weight='balanced')

    RF.fit(X, Y.ravel())
    yPred = RF.predict(Xtest)
    dice = dice_coef(Ytest, yPred)
    print(dice)
    return dice


resultsRF = [(depth, estimator,
              rf(depth, estimator, data_train, label_train, data_test,
                 label_test))
             for depth in depths for estimator in estimators]

diceVals = list(list(zip(*resultsRF))[2])
index = diceVals.index(max(diceVals))
best_depth, best_est = resultsRF[index][0], resultsRF[index][1]

clfRF = ExtraTreesClassifier(n_estimators=best_est, max_depth=best_depth,
                             class_weight='balanced')

cv = StratifiedKFold(n_splits=k, shuffle=False)

tprsRF = []
aucsRF = []
mean_fpr_RF = np.linspace(0, 5, 100)

i = 0

for train, test in cv.split(xTrainValid, yTrainValid):
    # fit RF model and get OOB decision function
    clfRF.fit(xTrainValid[train], yTrainValid[train].ravel())
    clfRF_probs = clfRF.predict_proba(xTrainValid[test])

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(yTrainValid[test], clfRF_probs[:, 1])
    tprsRF.append(np.interp(mean_fpr_RF, fpr, tpr))
    tprsRF[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucsRF.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    print(i)
    i += 1

mean_tpr_RF = np.mean(tprsRF, axis=0)
mean_tpr_RF[-1] = 1.0
mean_auc_RF = auc(mean_tpr_RF, mean_tpr_RF)
std_auc_RF = np.std(aucsRF)
plt.plot(mean_fpr_RF, mean_tpr_RF, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)'
               % (mean_auc_RF, std_auc_RF),
         lw=2,
         alpha=.8)

std_tpr = np.std(tprsRF, axis=0)
tprs_upper_RF = np.minimum(mean_tpr_RF + std_tpr, 1)
tprs_lower_RF = np.maximum(mean_tpr_RF - std_tpr, 0)
plt.fill_between(mean_fpr_RF, tprs_lower_RF, tprs_upper_RF, color='grey',
                 alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate of Ensemble Classifier')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic of RF Classifier')
plt.legend(loc="lower right")
plt.show()
plt.savefig('RF_5fold_pr_lab')

yPredict_Total_RF = clfRF.predict(data_test)
yPredict_train_RF = clfRF.predict(data_train)

RF_dice = dice_coef(label_test, yPredict_Total_RF)
RF_auc = auc(label_test, yPredict_Total_RF)
RF_train_dice = max(diceVals)
RF_train_auc = auc(yPredict_train_RF, y_train)

w1 = np.load('y_predictdeepLandS.npy')
yPredict_Total_RF = yPredict_Total_RF.reshape(w1.shape)

y_class = yPredict_Total_RF
y_class[y_class>=0.5] = 1
y_class[y_class<0.5] = 0


np.save('clf_RF_9models_lab', clfRF)

np.save("yPredict_Total_RF_lab", yPredict_Total_RF)
np.save("yclass_Total_RF_lab", y_class)
