# %%
import gzip
import math
import numpy as np
import random
import sklearn
import string
from collections import defaultdict
from nltk.stem.porter import *
from sklearn import linear_model
from gensim.models import Word2Vec
import dateutil
from scipy.sparse import lil_matrix # To build sparse feature matrices, if you like
import re

import warnings
warnings.filterwarnings('ignore')

# %%
answers = {}

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
### Question 1

# %%
dataset = []

f = gzip.open("data/steam_category.json.gz")
for l in f:
    d = eval(l)
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()

# %%
Ntrain = 10000
Ntest = 10000

dataTrain = dataset[:Ntrain]
dataTest = dataset[Ntrain:Ntrain + Ntest]

# %%
sp = set(string.punctuation)

# %%
dataTrain[0]

# %%
allTrainText = [datum['text'] for datum in dataTrain]

# %%
# Add \ before certain punctuations as they represent special characters in regex

specialChars = ['.', '*', '?', '+', '^', '$', '[', ']', '(', ')']
spGrouped = '|'.join(sp)

saw = []
for _ in range(len(spGrouped)):
    if (spGrouped[_] in specialChars) and (spGrouped[_] not in saw):
        saw.append(spGrouped[_])
        spGrouped = spGrouped[:_]+'\\'+spGrouped[_:]

# %%
pattern = rf'{spGrouped}'

# %%
cleanTrainText = []

for _ in range(len(allTrainText)):
    newString = re.sub(pattern, '', allTrainText[_])
    cleanTrainText.append(newString.lower())

# %%
# Find most common words in train text corpus

commonWords = defaultdict(int)

for t in cleanTrainText:
    words = t.split()
    for w in words:
        if w == 'i':
            allIs.append(w)
        commonWords[w] += 1

# %%
commonWords = sorted(commonWords.items(), key=lambda x:x[1], reverse=True)

# %%
commonWords[:10]

# %%
counts = commonWords[:10]

# %%
counts

# %%
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in dataTrain:
  r = ''.join([c for c in d['text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

# %%
answers['Q1'] = counts[:10]

# %%
assertFloatList([x[0] for x in answers['Q1']], 10)

# %%
### Question 2

# %%
NW = 1000 # dictionary size

# %%
words = [_[1] for _ in countsClassCode[:1000]]

# %%
# Build X...

wordID = dict(zip(words, range(len(words))))

def feat_q2(datum):
    feat = [0]*len(words)
    r = ''.join([c for c in datum['text'].lower() if not c in punctuation])
    for w in r.split():
        if w in words:
            feat[wordID[w]] += 1
    return feat

X = [feat_q2(datum) for datum in dataset]

# %%
y = [datum['genreID'] for datum in dataset]

# %%
Xtrain = X[:Ntrain]
ytrain = y[:Ntrain]
Xtest = X[Ntrain:]
ytest = y[Ntrain:]

# %%
mod = linear_model.LogisticRegression(C=1)

# %%
mod.fit(Xtrain, ytrain)

# %%
preds = mod.predict(Xtest)

# %%
correct = preds == ytest

# %%
answers['Q2'] = sum(correct) / len(correct)

# %%
assertFloat(answers['Q2'])

# %%
answers

# %%
### Question 3

# %%
targetWords = ['character', 'game', 'length', 'a', 'it']

# %%
docFrequency = defaultdict(int)
for datum in dataTrain:
    r = ''.join([c for c in datum['text'].lower() if not c in punctuation])
    for w in set(r.split()):
        docFrequency[w] += 1

# %%
def clean(text):
    r = ''.join([c for c in text.lower() if not c in punctuation])
    return r

# %%
# Term frequency for words in the first text corpus in train set

termFreq = defaultdict(int)
t = clean(dataTrain[0]['text'])
for w in t.split():
    termFreq[w] += 1

# %%
q3Answer = []
for w in targetWords:
    idf = math.log10(len(dataTrain) / docFrequency[w])
    tfidf = termFreq[w] * math.log10(len(dataTrain) / docFrequency[w])
    q3Answer.append((idf, tfidf))

# %%
answers['Q3'] = q3Answer

# %%
assertFloatList([x[0] for x in answers['Q3']], 5)
assertFloatList([x[1] for x in answers['Q3']], 5)

# %%
answers

# %%
### Question 4

# %%
topWords = [_[1] for _ in counts[:1000]]

# %%
wordID = dict(zip(topWords, range(len(topWords))))

# %%
df = defaultdict(int)

for datum in dataTrain:
    text = clean(datum['text'])
    for w in set(text.split()):
        if w in topWords:
            df[w] += 1

# %%
def feat_q4(text):
    feat = []
    text = clean(text)
    tf = defaultdict(int)

    for w in text.split():
        tf[w] += 1
    for w in topWords:
        tfidf = tf[w] * math.log10(len(dataTrain) / df[w])
        feat.append(tfidf)

    return feat

# %%
# Build X and y...

X = [feat_q4(datum['text']) for datum in dataset]
y = [datum['genreID'] for datum in dataset]

# %%
Xtrain = X[:Ntrain]
ytrain = y[:Ntrain]
Xtest = X[Ntrain:]
ytest = y[Ntrain:]

# %%
mod = linear_model.LogisticRegression(C=1)

# %%
mod.fit(Xtrain, ytrain)

# %%
preds = mod.predict(Xtest)
correct = preds == ytest

# %%
answers['Q4'] = sum(correct) / len(correct)

# %%
assertFloat(answers['Q4'])

# %%
answers

# %%
### Question 5

# %%
def Cosine(x1,x2):
    numer, norm1, norm2 = 0, 0, 0
    for a1,a2 in zip(x1,x2):
        numer += a1*a2
        norm1 += a1**2
        norm2 += a2**2
    if norm1*norm2:
        return numer / math.sqrt(norm1*norm2)
    return 0

# %%
similarities = {}
firstInTest = feat_q4(dataTrain[0]['text'])

for datum in dataTest:
    feat = feat_q4(datum['text'])
    sim = Cosine(firstInTest, feat)
    similarities[datum['reviewID']] = sim

# %%
# similarities.sort(reverse=True)

# %%
similarities = sorted(similarities.items(), key=lambda x:x[1], reverse=True)

# %%
answers['Q5'] = (similarities[0][1], similarities[0][0])

# %%
assertFloat(answers['Q5'][0])

# %%
answers

# %%
### Question 6

# %%
cValues = np.linspace(0.1, 1, 5)
bestAcc = None

for c in cValues:
    mod = linear_model.LogisticRegression(C=c)
    mod.fit(Xtrain, ytrain)
    preds = mod.predict(Xtest)
    correct = preds == ytest
    acc = sum(correct) / len(correct)
    print(acc)
    if acc > max(answers['Q2'], answers['Q4']):
        print(f"Found better c: c = {c}, acc = {acc}")
        if bestAcc is None: 
            bestAcc = acc
        elif bestAcc < acc:
            bestAcc = acc

# %%
answers['Q6'] = bestAcc

# %%
assertFloat(answers['Q6'])

# %%
answers

# %%
### Question 7

# %%
import dateutil.parser

# %%
dataset = []

f = gzip.open("data/young_adult_20000.json.gz")
for l in f:
    d = eval(l)
    # print(d['date_added'])
    # print(type(d['date_added']))
    d['datetime'] = dateutil.parser.parse(d['date_added'])
    # print(d['datetime'])
    # print(type(d['datetime']))
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()

# %%
dataset[0]

# %%
reviewsPerUser = defaultdict(list)
for r in dataset:
    uid, bid = r['user_id'], r['book_id']
    reviewsPerUser[uid].append(bid)

# %%
reviewLists = []
for u in reviewsPerUser:
    rl = list(reviewsPerUser[u])
    rl.sort()
    reviewLists.append(rl)

# %%
model5 = Word2Vec(reviewLists,
                  min_count=1, # Words/items with fewer instances are discarded
                  vector_size=5, # Model dimensionality
                  window=3, # Window size
                  sg=1) # Skip-gram model

# %%
firstRev = dataset[0]['book_id']

# %%
firstRev

# %%
res = model5.wv.similar_by_word(firstRev)

# %%
answers['Q7'] = res[:5]

# %%
assertFloatList([x[1] for x in answers['Q7']], 5)

# %%
f = open("answers_hw4.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



