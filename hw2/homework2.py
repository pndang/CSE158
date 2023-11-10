# %%
import random
from sklearn import linear_model
from matplotlib import pyplot as plt
from collections import defaultdict
import gzip
import numpy as np
from sklearn import metrics
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
answers = {}

# %%
def parseData(fname):
    for l in open(fname):
        yield eval(l)

# %%
data = list(parseData("data/beer_50000.json"))

# %%
random.seed(0)
random.shuffle(data)

# %%
dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]

# %%
yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]

# %%
categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1

# %%
categories = [c for c in categoryCounts if categoryCounts[c] > 1000]

# %%
catID = dict(zip(list(categories),range(len(categories))))

# %%
max_len = 0
for datum in dataTrain:
    if len(datum['review/text']) > max_len:
        max_len = len(datum['review/text'])

# %%
def feat(d, includeCat = True, includeReview = True, includeLength = True):
    # In my solution, I wrote a reusable function that takes parameters to generate features for each question
    # Feel free to keep or discard

    output_feat = []

    if includeCat:
        cat_feat = [0]*len(catID)
        cat = d['beer/style']
        if cat in categories:
            cat_feat[catID[cat]] = 1
        output_feat += cat_feat
    
    if includeReview:
        rev_feat = [d['review/aroma'], d['review/overall'], d['review/appearance'], \
            d['review/taste'], d['review/palate']]
        output_feat += rev_feat

    if includeLength:
        len_feat = len(d['review/text']) / max_len
        output_feat += [len_feat]
    
    return [1] + output_feat

# %%
# sanity check

test = np.array([1, 1, 2])
test1 = np.array([1, 1, 3])
sum(test==test1)

# %%
# sanity check

[feat(datum, True, True, True) for datum in dataTrain[:5]]

# %%
def pipeline(reg, includeCat = True, includeReview = True, includeLength = True):
    # ...

    features = [feat(datum, includeCat, includeReview, includeLength) for datum in dataTrain]
    reg = linear_model.LogisticRegression(C=reg, fit_intercept=False, class_weight='balanced').fit(features, yTrain)

    # validation
    valid_feat = [feat(datum, includeCat, includeReview, includeLength) for datum in dataValid]
    valid_preds = reg.predict(valid_feat)
    valid_acc = sum(valid_preds==yValid) / len(valid_preds)
    tn, fp, fn, tp = metrics.confusion_matrix(yValid, valid_preds).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn) 
    valid_ber = 0.5*(fpr + (1 - tpr))

    # test
    test_feat = [feat(datum, includeCat, includeReview, includeLength) for datum in dataTest]
    test_preds = reg.predict(test_feat)
    test_acc = sum(test_preds==yTest) / len(test_preds)
    tn, fp, fn, tp = metrics.confusion_matrix(yTest, test_preds).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn) 
    test_ber = 0.5*(fpr + (1 - tpr))

    return reg, valid_ber, test_ber

# %%
### Question 1

# %%
mod, validBER, testBER = pipeline(10, True, False, False)

# %%
answers['Q1'] = [validBER, testBER]

# %%
assertFloatList(answers['Q1'], 2)

# %%
### Question 2

# %%
mod, validBER, testBER = pipeline(10, True, True, True)

# %%
answers['Q2'] = [validBER, testBER]

# %%
assertFloatList(answers['Q2'], 2)

# %%
### Question 3

# %%
for c in [0.001, 0.01, 0.1, 1, 10]:
    validation_ber = pipeline(c, True, True, True)[1]
    print(f'c = {c} - valid BER = {validation_ber}\n')

# %%
bestC = 10

# %%
mod, validBER, testBER = pipeline(10, True, True, True)

# %%
answers['Q3'] = [bestC, validBER, testBER]

# %%
assertFloatList(answers['Q3'], 3)

# %%
### Question 4

# %%
# sanity check

[feat(datum, True, False, True) for datum in dataTrain[:5]]

# %%
mod, validBER, testBER_noCat = pipeline(1, False, True, True)

# %%
mod, validBER, testBER_noReview = pipeline(1, True, False, True)

# %%
mod, validBER, testBER_noLength = pipeline(1, True, True, False)

# %%
answers['Q4'] = [testBER_noCat, testBER_noReview, testBER_noLength]

# %%
assertFloatList(answers['Q4'], 3)

# %%
### Question 5

# %%
path = "data/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')

# %%
header

# %%
dataset = []

pairsSeen = set()

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    ui = (d['customer_id'], d['product_id'])
    if ui in pairsSeen:
        print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)

# %%
dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]

# %%
dataset[0]

# %%
# Feel free to keep or discard

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}
ratingDict = {} # To retrieve a rating for a specific user/item pair
reviewsPerUser = defaultdict(list)

# for d in dataTrain:
#     user, item = d['customer_id'], d['product_id']
#     usersPerItem[item].add(user)
#     itemsPerUser[user].add(item)
#     itemNames[item] = d['product_title']
#     ratingDict[(user, item)] = d['star_rating']
#     reviewsPerUser[user].append(d)

for d in dataset:
    user, item = d['customer_id'], d['product_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    itemNames[item] = d['product_title']
    ratingDict[(user, item)] = d['star_rating']
    reviewsPerUser[user].append(d)

# %%
userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    ratings = [ratingDict[(u, i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(ratings) / len(ratings)
    
for i in usersPerItem:
    ratings = [ratingDict[(u, i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(ratings) / len(ratings)

ratingMean = np.mean(list(ratingDict.values()))

# %%
# sanity check

sum([d['star_rating'] for d in dataset]) / len(dataset)

# %%
ratingMean

# %%
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom

# %%
# sanity check

test = set([1, 2, 3])
test1 = set([1, 5 ,3])
len(test.union(test1))

# %%
def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i:
            continue
        similarity = Jaccard(users, usersPerItem[i2])
        similarities.append((similarity, i2))
    similarities.sort(key = lambda x: x[0], reverse=True)
    return similarities[:N]


# %%
query = 'B00KCHRKD6'

# %%
ms = mostSimilar(query, 10)

# %%
answers['Q5'] = ms

# %%
assertFloatList([m[0] for m in ms], 10)

# %%
### Question 6

# %%
def MSE(y, ypred):
    errors = [(x-y)**2 for x,y in zip(y, ypred)]
    return sum(errors) / len(errors)

# %%
'B00KCHRKD6' in itemNames.keys()

# %%
dataTrain[0]

# %%
reviewsPerUser = defaultdict(list)
usersPerItem = defaultdict(set)
itemAverages = {}

for d in dataTrain:
    user, item = d['customer_id'], d['product_id']
    reviewsPerUser[user].append(d)
    usersPerItem[item].add(user)

for i in usersPerItem:
    itemAverages[i] = np.mean([ratingDict[(u, i)] for u in usersPerItem[i]])


# %%
averageRating = np.mean([d['star_rating'] for d in dataTrain])

# %%
def predictRating(user,item):

    if item not in itemAverages.keys(): return averageRating

    ratings = []
    similarities = []
    for r in reviewsPerUser[user]:
        item2 = r['product_id']
        if item2 == item:
            continue
        ratings.append(r['star_rating'] - itemAverages[item2])
        
        # calculating similarity between item and item2
        similarity = Jaccard(usersPerItem[item], usersPerItem[item2])
        similarities.append(similarity)
    
    if sum(similarities) > 0:
        weightedRatings = [(x*y) for x,y in zip(ratings, similarities)]
        return itemAverages[item] + (sum(weightedRatings) / sum(similarities))
    else:
        return itemAverages[item]
    

# %%
# test = set([1])
# test1 = set([])
# ans = test.union(test1)
# len(ans) is 0

# %%
alwaysPredictMean = [averageRating]*len(dataTest)

# %%
# sanity check

len(alwaysPredictMean) / len(dataset)

# %%
simPredictions = [predictRating(d['customer_id'], d['product_id']) for d in dataTest]

# %%
labels = [d['star_rating'] for d in dataTest]

# %%
answers['Q6'] = MSE(simPredictions, labels)

# %%
assertFloat(answers['Q6'])

# %%
answers

# %%
### Question 7 - incorporate time-weight collaborative filtering
### (simple temporal dynamics feature for recommender systems)

# %%
pd.to_datetime(dataset[9999]['review_date']) 

# %%
delta = pd.to_datetime(dataset[0]['review_date']) - pd.to_datetime(dataset[9999]['review_date'])

# %%
delta.days

# %%
reviewDates = {}
for d in dataset:
    pair = (d['customer_id'], d['product_id'])
    reviewDates[pair] = d['review_date']

# %%
# test = [pd.to_datetime(d['review_date']) for d in dataset]

# %%
# min(test)

# %%
# max(test) 

# %%
# abs(min(test)- max(test)).days / 365

# %%
def predictRating_q7(user,item):

    if item not in itemAverages.keys(): return averageRating

    itemTimestamp = pd.to_datetime(reviewDates[(user, item)])

    ratings = []
    similarities = []
    recency = []
    for r in reviewsPerUser[user]:
        item2 = r['product_id']
        if item2 == item:
            continue
        ratings.append(r['star_rating'] - itemAverages[item2])
        
        # calculating similarity between item and item2
        similarity = Jaccard(usersPerItem[item], usersPerItem[item2])
        similarities.append(similarity)

        # calculating time delta between review user,item2 and review user,item
        # and determine weight/influence using decay function depending on recency
        item2Timestamp = pd.to_datetime(reviewDates[(user, item2)])
        timeDelta = abs(itemTimestamp - item2Timestamp)
        recency.append(np.exp(-1/8*(timeDelta.days/365)))
    
    if sum([(x*y) for x,y in zip(similarities, recency)]) > 0:
        weightedRatings = [(x*y) for x,y in zip(ratings, similarities)]
        ratingsWithRecency = [(x*y) for x,y in zip(weightedRatings, recency)]
        return itemAverages[item] + (sum(ratingsWithRecency) / \
            sum([(x*y) for x,y in zip(similarities, recency)]))
    else:
        return itemAverages[item]

# %%
preds_q7 = [predictRating_q7(d['customer_id'], d['product_id']) for d in dataTest]

# %%
MSE(preds_q7, simPredictions)

# %%
actual = [d['star_rating'] for d in dataTest]

# %%
itsMSE = MSE(preds_q7, actual)

# %%
answers['Q7'] = ["I chose t(u, j) to represent the duration between when user u"+\
    " reviewed item i and when they reviewed item j, in years, which is then"+\
    " plugged into a decay function with lambda = -1/8. Graphically, this lambda"+\
    " parameter choice adequately captures the range of duration, or doesn't converge"+\
    " to zero too early or late.", itsMSE]

# %%
assertFloat(answers['Q7'][1])

# %%
answers

# %%
f = open("test_script_answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



