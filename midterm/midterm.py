# %%
import json
import gzip
import math
from collections import defaultdict
import numpy as np
from sklearn import linear_model
import random
import statistics

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
answers = {}

# %%
z = gzip.open("data/train.json.gz")

# %%
dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)

# %%
z.close()

# %%
### Question 1

# %%
def MSE(y, ypred):
    return np.sum((y-ypred)**2) / len(y)

# %%
def MAE(y, ypred):
    return np.sum(abs(y-ypred)) / len(y)

# %%
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['userID'],d['gameID']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
    
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['date'])
    
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['date'])

# %%
for d in dataset:
    print(d)
    break

# %%
def feat1(d):
    feat = [1]
    hour = d['hours']
    return feat + [hour]

# %%
X = [feat1(datum) for datum in dataset]
y = [len(datum['text']) for datum in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)

# %%
theta_1 = mod.coef_[1]
mse_q1 = MSE(np.array(y), np.array(predictions))

# %%
answers['Q1'] = [theta_1, mse_q1]

# %%
assertFloatList(answers['Q1'], 2)

# %%
answers

# %%
### Question 2

# %%
medianHours = np.median([datum['hours'] for datum in dataset])

# %%
dataset[0]

# %%
def feat2(d):
    feat = [1]
    hour = d['hours']
    hourTransformed = d['hours_transformed']
    sqrtHour = np.sqrt(hour)
    aboveMedian = 1 if hour > medianHours else 0
    return feat + [hour, hourTransformed, sqrtHour, aboveMedian]

# %%
X = [feat2(d) for d in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)

# %%
mse_q2 = MSE(np.array(y), np.array(predictions))

# %%
answers['Q2'] = mse_q2

# %%
assertFloat(answers['Q2'])

# %%
answers

# %%
### Question 3

# %%
def feat3(d):
    feat = [1]
    hour = d['hours']
    aboveOne = 1 if hour > 1 else 0
    aboveFive = 1 if hour > 5 else 0
    aboveTen = 1 if hour > 10 else 0
    aboveHundred = 1 if hour > 100 else 0
    aboveThousand = 1 if hour > 1000 else 0
    return feat + [aboveOne, aboveFive, aboveTen, aboveHundred, aboveThousand]

# %%
X = [feat3(d) for d in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)

# %%
mse_q3 = MSE(np.array(y), np.array(predictions))

# %%
answers['Q3'] = mse_q3

# %%
assertFloat(answers['Q3'])

# %%
answers

# %%
### Question 4

# %%
def feat4(d):
    feat = [1]
    reviewLength = len(d['text'])
    return feat + [reviewLength]

# %%
X = [feat4(d) for d in dataset]
y = [datum['hours'] for datum in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)

# %%
mse = MSE(np.array(y), np.array(predictions))
mae = MAE(np.array(y), np.array(predictions))

# %%
print(f'Min error: {np.min(y-predictions)}')
print(f'Max error: {np.max(y-predictions)}')
print(f'Diff: {np.max(y-predictions) - np.min(y-predictions)}')

# %%
answers['Q4'] = [mse, mae, "Since we have a sizable range of error (16548), the MAE is better as it uses the absolute differences of errors, whereas the MSE squares errors, which exacerbates our errors, especially when the difference is large."]

# %%
assertFloatList(answers['Q4'][:2], 2)

# %%
answers

# %%
### Question 5

# %%
y_trans = [datum['hours_transformed'] for datum in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)

# %%
mse_trans = MSE(np.array(y_trans), np.array(predictions_trans))

# %%
predictions_untrans = (2**(np.array(predictions_trans)))-1

# %%
predictions_trans

# %%
predictions_untrans

# %%
mse_untrans = MSE(np.array(y), np.array(predictions_untrans))

# %%
answers['Q5'] = [mse_trans, mse_untrans]

# %%
assertFloatList(answers['Q5'], 2)

# %%
answers

# %%
### Question 6

# %%
def feat6(d):
    feat = [1]
    hourEncode = [0]*100
    hourRoundDown = math.floor(d['hours'])
    if hourRoundDown <= 99:
        hourEncode[hourRoundDown] = 1
    else: 
        hourEncode[-1] = 1
    return feat + hourEncode

# %%
X = [feat6(d) for d in dataset]
y = [len(d['text']) for d in dataset]

# %%
dataset[88]

# %%
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

# %%
models = {}
mses = {}
bestC = None
bestMSE = None

for c in [1, 10, 100, 1000, 10000]:
    mdl = linear_model.Ridge(alpha=c, fit_intercept=False)
    mdl.fit(Xtrain, ytrain)
    preds = mdl.predict(Xvalid)
    mse = MSE(np.array(yvalid), np.array(preds))
    models[c] = mdl
    mses[c] = mse 
    if bestMSE is None:
        bestMSE = mse
        bestC = c
    if mse < bestMSE:
        bestMSE = mse
        bestC = c

# %%
predictions_test = models[bestC].predict(Xtest)

# %%
mse_valid = mses[bestC]

# %%
mse_test = MSE(np.array(ytest), np.array(predictions_test))

# %%
answers['Q6'] = [bestC, mse_valid, mse_test]

# %%
assertFloatList(answers['Q6'], 3)

# %%
answers

# %%
### Question 7

# %%
times = [d['hours_transformed'] for d in dataset]
median = statistics.median(times)

# %%
notPlayed = [datum['hours'] for datum in dataset]
nNotPlayed = np.sum([1 for h in notPlayed if h < 1])

# %%
answers['Q7'] = [median, nNotPlayed]

# %%
assertFloatList(answers['Q7'], 2)

# %%
answers

# %%
### Question 8

# %%
def feat8(d):
    return [len(d['text'])]


# %%
X = [feat8(d) for d in dataset]
y = [d['hours_transformed'] > median for d in dataset]

# %%
mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X) # Binary vector of predictions

# %%
def rates(predictions, y):
    truePos = 0
    numPos = 0
    trueNeg = 0
    numNeg = 0
    for i in range(len(y)):
        if y[i]:
            numPos += 1
        if predictions[i] and y[i]:
            truePos += 1
        if not y[i]:
            numNeg += 1
        if not predictions[i] and not y[i]:
            trueNeg += 1
    TP = truePos
    FP = numNeg-trueNeg
    TN = trueNeg
    FN = numPos-truePos
    return TP, TN, FP, FN

# %%
TP, TN, FP, FN = rates(predictions, y)

# %%
SEN = TP / (TP + FN)  # sensitivity / TPR 
FPR = FP / (FP + TN)  # FPR
SPE = TN / (TN + FP)  # specificity / TNR
BER = 0.5*(FPR + (1-SEN))

# %%
answers['Q8'] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q8'], 5)

# %%
answers

# %%
### Question 9

# %%
# confidences = reg_q5.decision_function(X)

# sortedByConfidence = list(zip(confidences, y))
# sortedByConfidence.sort(reverse=True)

# sortedByConfidence

# precs = []

# for k in [1,100,1000,10000]:
#     topK = sortedByConfidence[:k]
#     prec = np.sum([1 if pred[1] else 0 for pred in topK]) / k
#     precs.append(prec)

# %%
precision = lambda label: np.sum([1 if label[1] else 0 for label in topK]) / len(topK)
recall = lambda label: np.sum([1 if label[1] else 0 for label in topK]) / np.sum(y)

# %%
confidences = mod.decision_function(X)
sortedByConfidence = list(zip(confidences, y))
sortedByConfidence.sort(reverse=True)

# %%
precs = []
recs = []

for i in [5, 10, 100, 1000]:
    topK = sortedByConfidence[:i]
    threshold = topK[-1][0]
    k = i
    nextVal = sortedByConfidence[k][0]
    while nextVal == threshold:
        topK.append(sortedByConfidence[k])
        k += 1
        nextVal = sortedByConfidence[k][0]
    prec = precision(topK)
    rec = recall(topK)
    precs.append(prec)
    recs.append(rec)


# %%
answers['Q9'] = precs

# %%
assertFloatList(answers['Q9'], 4)

# %%
answers

# %%
### Question 10

# %%
y_trans = [d['hours_transformed'] for d in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)

# %%
predictions_thresh

# %%
predictions_thresh = np.median(predictions_trans)+0.31

# %%
binaryPreds = [True if pred > predictions_thresh else False for pred in predictions_trans]

# %%
TP, TN, FP, FN = rates(binaryPreds, y)

# %%
SEN = TP / (TP + FN)  # sensitivity / TPR 
FPR = FP / (FP + TN)  # FPR
SPE = TN / (TN + FP)  # specificity / TNR
BER = 0.5*(FPR + (1-SEN))

# %%
BER

# %%
your_threshold = predictions_thresh

# %%
answers['Q10'] = [your_threshold, BER]

# %%
assertFloatList(answers['Q10'], 2)

# %%
answers

# %%
### Question 11

# %%
dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]

# %%
userMedian = defaultdict(list)
itemMedian = defaultdict(list)

# Compute medians on training data

# %%
dataTrain[0]

# %%
for datum in dataTrain:
    u, i = datum['userID'], datum['gameID']
    userMedian[u].append(datum['hours'])
    itemMedian[i].append(datum['hours'])

for u in userMedian:
    userMedian[u] = np.median(userMedian[u])

for i in itemMedian:
    itemMedian[i] = np.median(itemMedian[i])

# %%
answers['Q11'] = [itemMedian['g35322304'], userMedian['u55351001']]

# %%
assertFloatList(answers['Q11'], 2)

# %%
answers

# %%
### Question 12

# %%
globalMedian = np.median([datum['hours'] for datum in dataTrain])

# %%
def f12(u,i):
    # Function returns a single value (0 or 1)
    if i in list(itemMedian.keys()):
        return 1 if itemMedian[i] > globalMedian else 0
    return 1 if userMedian[u] > globalMedian else 0

# %%
preds = [f12(d['userID'], d['gameID']) for d in dataTest]

# %%
testMedian = np.median([datum['hours'] for datum in dataTest])
y = [1 if datum['hours'] > testMedian else 0 for datum in dataTest]

# %%
accuracy = np.sum(np.array(preds)==np.array(y)) / len(y)

# %%
answers['Q12'] = accuracy

# %%
assertFloat(answers['Q12'])

# %%
answers

# %%
### Question 13

# %%
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}

for d in dataset:
    user,item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)

# %%
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom

# %%
def mostSimilar(i, func, N):
    similarities = []
    for i2 in usersPerItem:
        if i2 == i:
            continue
        simmilarity = func(usersPerItem[i], usersPerItem[i2])
        similarities.append((simmilarity, i2))
    similarities.sort(reverse=True)
    return similarities[:N]

# %%
ms = mostSimilar(dataset[0]['gameID'], Jaccard, 10)

# %%
answers['Q13'] = [ms[0][0], ms[-1][0]]

# %%
assertFloatList(answers['Q13'], 2)

# %%
answers

# %%
### Question 14

# %%
def mostSimilar14(i, func, N):
    similarities = []
    for i2 in usersPerItem:
        if i2 == i:
            continue
        simmilarity = func(i, i2)
        similarities.append((simmilarity, i2))
    similarities.sort(reverse=True)
    return similarities[:N]

# %%
median = np.median([datum['hours'] for datum in dataset])

# %%
ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = 1 if d['hours'] > median else -1
    ratingDict[(u,i)] = lab

# %%
def Cosine(i1, i2):
    # Between two items
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer = np.sum([ratingDict[(u, i1)]*ratingDict[(u, i2)] for u in inter])
    norm1 = np.sum([ratingDict[(u, i1)]**2 for u in usersPerItem[i1]])
    norm2 = np.sum([ratingDict[(u, i2)]**2 for u in usersPerItem[i2]])
    denom = math.sqrt(norm1) * math.sqrt(norm2)
    if denom == 0:
        return 0
    return numer / denom

# %%
ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)

# %%
answers['Q14'] = [ms[0][0], ms[-1][0]]

# %%
assertFloatList(answers['Q14'], 2)

# %%
answers

# %%
### Question 15

# %%
ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = d['hours_transformed']
    ratingDict[(u,i)] = lab

# %%
ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)

# %%
answers['Q15'] = [ms[0][0], ms[-1][0]]

# %%
assertFloatList(answers['Q15'], 2)

# %%
answers

# %%
f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



