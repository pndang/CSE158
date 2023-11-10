# %%
import json
from collections import defaultdict
from sklearn import linear_model
import numpy as np
import random
import gzip
import dateutil.parser
import math

from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import matplotlib.pyplot as plt

# %%
answers = {}

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
### Question 1 - Regression

# %%
f = gzip.open("data/fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

# %%
def feature(datum):
    review_lengths = []
    review_ratings = []

    for review in datum:
        review_lengths.append([len(review['review_text'])])
        review_ratings.append(review['rating'])
        
    # Scale review lengths
    review_lengths = review_lengths / np.max(review_lengths)

    return review_lengths, review_ratings

# %%
X = feature(dataset)[0]
Y = feature(dataset)[1]

# %%
# Instantiate model

reg = linear_model.LinearRegression().fit(X, Y)

# %%
theta = (reg.intercept_, reg.coef_[0])

# %%
# Get MSE

preds = reg.predict(X)
MSE = np.sum((Y - preds)**2) / len(preds)

# %%
# Sanity check

from sklearn import metrics
metrics.mean_squared_error(Y, preds)

# %%
answers['Q1'] = [theta[0], theta[1], MSE]

# %%
assertFloatList(answers['Q1'], 3)

# %%
### Question 2

# %%
for d in dataset:
    t = dateutil.parser.parse(d['date_added'])
    d['parsed_date'] = t

# %%
# Manually write out feature vector for first two reviews

# [1, 0.14581295, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# [1, 0.10631903, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

# %%
def feature(datum):
    review_lengths = []
    review_ratings = []
    weekdays = []
    months = []
    offset_term = [[1]]*len(datum)

    for review in datum:
        review_lengths.append([len(review['review_text'])])
        review_ratings.append(review['rating'])
        weekdays.append([review['parsed_date'].weekday()])
        months.append([review['parsed_date'].month])
    
    # Scale review lengths
    review_lengths = review_lengths / np.max(review_lengths)

    # One-hot encode weekdays
    weekday_enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    weekday_enc.fit(weekdays)
    weekdays_ohe = weekday_enc.transform(weekdays).toarray()

    # One-hot encode months
    month_enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    month_enc.fit(months)
    month_ohe = month_enc.transform(months).toarray()

    return np.hstack((offset_term, review_lengths, weekdays_ohe, month_ohe)), \
        review_ratings

# %%
X_q2 = feature(dataset)[0]
Y_q2 = feature(dataset)[1]

# %%
answers['Q2'] = [list(X_q2[0]), list(X_q2[1])]

# %%
assertFloatList(answers['Q2'][0], 19)
assertFloatList(answers['Q2'][1], 19)

# %%
### Question 3

# %%
def feature3(datum):
    review_lengths = []
    review_ratings = []
    weekdays = []
    months = []

    for review in datum:
        review_lengths.append([len(review['review_text'])])
        review_ratings.append(review['rating'])
        weekdays.append([review['parsed_date'].weekday()])
        months.append([review['parsed_date'].month])
        
    # Scale review lengths
    review_lengths = review_lengths / np.max(review_lengths)

    return np.hstack((review_lengths, weekdays, months)), review_ratings

# %%
X3 = feature3(dataset)[0]
Y3 = feature3(dataset)[1]

# %%
# Use weekday and month values directly

reg_raw = linear_model.LinearRegression(fit_intercept=True).fit(X3, Y3)
preds_raw = reg_raw.predict(X3)
mse3 = np.sum((Y3 - preds_raw)**2) / len(preds_raw)

# Use one-hot encoded features

reg_q2 = linear_model.LinearRegression(fit_intercept=False).fit(X_q2, Y_q2)
preds_q2 = reg_q2.predict(X_q2)
mse2 = np.sum((Y_q2 - preds_q2)**2) / len(preds_q2)

# %%
answers['Q3'] = [mse2, mse3]

# %%
assertFloatList(answers['Q3'], 2)

# %%
### Question 4

# %%
random.seed(0)
random.shuffle(dataset)

# %%
# X2 = [feature(d) for d in dataset]
X2 = feature(dataset)[0]
# X3 = [feature3(d) for d in dataset]
X3 = feature3(dataset)[0]
Y = [d['rating'] for d in dataset]

# %%
train2, test2 = X2[:len(X2)//2], X2[len(X2)//2:]
train3, test3 = X3[:len(X3)//2], X3[len(X3)//2:]
trainY, testY = Y[:len(Y)//2], Y[len(Y)//2:]

# %%
# Train and evaluate one-hot encoding model (q2)

mdl_q2 = linear_model.LinearRegression(fit_intercept=False).fit(train2, trainY)
mdl_q2_preds = mdl_q2.predict(test2)
test_mse2 = np.sum((testY - mdl_q2_preds)**2) / len(mdl_q2_preds)

# Train and evaluate direct encoding model (q3)

mdl_q3 = linear_model.LinearRegression().fit(train3, trainY)
mdl_q3_preds = mdl_q3.predict(test3)
test_mse3 = np.sum((testY - mdl_q3_preds)**2) / len(mdl_q3_preds)

# %%
answers['Q4'] = [test_mse2, test_mse3]

# %%
assertFloatList(answers['Q4'], 2)

# %%
answers

# %%
### Question 5

# %%
f = open("data/beer_50000.json")
dataset = []
for l in f:
    dataset.append(eval(l))

# %%
X = [[len(r['review/text'])] for r in dataset]
y = [r['review/overall'] >= 4 for r in dataset]

# %%
reg_q5 = linear_model.LogisticRegression(class_weight='balanced').fit(X, y)

# %%
preds_q5 = reg_q5.predict(X)

# %%
tn, fp, fn, tp = metrics.confusion_matrix(y, preds_q5).ravel()

# %%
TP = tp
TN = tn 
FP = fp
FN = fn

# %%
SEN = TP / (TP + FN)  # sensitivity / TPR 
FPR = FP / (FP + TN)  # FPR
SPE = TN / (TN + FP)  # specificity / TNR
BER = 0.5*(FPR + (1-SEN))

# %%
answers['Q5'] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q5'], 5)

# %%
answers

# %%
### Question 6

# %%
confidences = reg_q5.decision_function(X)

# %%
sortedByConfidence = list(zip(confidences, y))
sortedByConfidence.sort(reverse=True)

# %%
sortedByConfidence

# %%
precs = []

# %%
for k in [1,100,1000,10000]:
    topK = sortedByConfidence[:k]
    prec = np.sum([1 if pred[1] else 0 for pred in topK]) / k
    precs.append(prec)

# %%
# Plot the precision@K 

plt.plot([1,100,1000,10000], precs)
plt.xlabel('K')
plt.ylabel('Precision')
plt.title('precision@K');

# %%
answers['Q6'] = precs

# %%
assertFloatList(answers['Q6'], 4)

# %%
answers

# %%
### Question 7

# %%
dataset[0]

# %%
type(dataset[0]['review/timeStruct']['year'])

# %%
# r['review/timeStruct']['hour']

# %%
X_q7 = [[len(r['review/text']), r['review/taste']] for r in dataset]
y_q7 = [r['review/overall'] >= 4 for r in dataset]

# %%
# Examine for linearity

text_len = []
taste_review = []

for datum in X_q7:
    text_len.append(datum[0])
    taste_review.append(datum[1])

# text_len.sort()
# taste_review.sort()

# %%
plt.scatter(text_len, taste_review)

# %%
reg_q7 = linear_model.LogisticRegression(class_weight='balanced').fit(X_q7, y_q7)

# %%
preds_q7 = reg_q7.predict(X_q7)
tn_q7, fp_q7, fn_q7, tp_q7 = metrics.confusion_matrix(y_q7, preds_q7).ravel()
SEN_q7 = tp_q7 / (tp_q7 + fn_q7)  # sensitivity / TPR 
FPR_q7 = fp_q7 / (fp_q7 + tn_q7)  # FPR
SPE_q7 = tn_q7 / (tn_q7 + fp_q7)  # specificity / TNR
BER_q7 = 0.5*(FPR_q7 + (1-SEN_q7))

# %%
BER_q7

# %%
its_test_BER = BER_q7

# %%
reasoning = "I added the taste review as a second learning feature. My rationale"\
    +" is people's assessment/perception of the beer's taste will have a strong"\
    +" influence on the overall review score. I also plotted the two learning"\
    +" feature to check for feature association and there isn't seem to be any."

# %%
answers['Q7'] = [reasoning, its_test_BER]

# %%
answers

# %%
f = open("answers_hw1.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



