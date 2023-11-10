# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
from sklearn import linear_model
import os
import matplotlib.pyplot as plt
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
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readJSON(path):
    f = gzip.open(path, 'rt', encoding='utf8')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d

# %%
answers = {}

# %%
# Some data structures that will be useful

gamesPerUser = defaultdict(set)
allGames = set()

# %%
allHours = []
for l in readJSON("data/train.json.gz"):
    allHours.append(l)

# %%
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

# %%
##################################################
# Play prediction                                #
##################################################

# %%
# Any other preprocessing...

hoursValid[5]

# %%
### Question 1

for h in allHours:
    uid = h[0]
    gid = h[1]
    gamesPerUser[uid].add(gid)
    allGames.add(gid)

# %%
# Random sampling

newValid = []
for h in hoursValid:
    uid = h[0]
    haveNotPlayed = []
    for g in allGames:
        if g not in gamesPerUser[uid]:
            haveNotPlayed.append(g)
    newValid.append((uid, h[1]))
    newValid.append((uid, random.choice(haveNotPlayed)))

# %%
# Evaluate baseline strategy

gameCount = defaultdict(int)
totalPlayed = 0

for user,game,_ in readJSON("data/train.json.gz"):
  gameCount[game] += 1
  totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

# %%
return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalPlayed/2: break

# %%
newValid

# %%
'g61913894' in gamesPerUser['u00914251']

# %%
actual_q1 = []
pred_q1 = []

for pair in newValid:
    uid = pair[0]
    gid = pair[1]

    if gid in gamesPerUser[uid]:
        actual_q1.append(1)
    else:
        actual_q1.append(0)

    if gid in return1:
        pred_q1.append(1)
    else:
        pred_q1.append(0)

# %%
matches = 0

for i in range(len(actual_q1)):
    if actual_q1[i] == pred_q1[i]:
        matches += 1

acc_q1 = matches / len(actual_q1)

# %%
answers['Q1'] = acc_q1

# %%
assertFloat(answers['Q1'])

# %%
answers

# %%
### Question 2

# %%
# Improved strategy

def filterMostPopular(factor=1/2):
    output = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        output.add(i)
        if count > totalPlayed*factor: 
            return output

factors = [0.35, 0.40, 0.45, 0.5, 0.55]
for f in factors:
    return1 = filterMostPopular(f)

    pred = []
    for pair in newValid:
        uid = pair[0]
        gid = pair[1]

        if gid in return1:
            pred.append(1)
        else:
            pred.append(0)

    matches = 0
    for i in range(len(actual_q1)):
        if actual_q1[i] == pred[i]:
            matches += 1

    acc = matches / len(actual_q1)
    print(f"factor: {f}, acc: {acc}")

# %%
# Evaluate baseline strategy

return1 = filterMostPopular(0.55)

pred_q2 = []
for pair in newValid:
    uid = pair[0]
    gid = pair[1]

    if gid in return1:
        pred_q2.append(1)
    else:
        pred_q2.append(0)

matches = 0
for i in range(len(actual_q1)):
    if actual_q1[i] == pred_q2[i]:
        matches += 1

acc_q2 = matches / len(actual_q1)

# %%
answers['Q2'] = [totalPlayed*0.55, acc_q2]

# %%
answers

# %%
assertFloatList(answers['Q2'], 2)

# %%
### Question 3/4

# %%
def Jaccard(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union

# %%
# Useful data structures

playersPerGame = defaultdict(set)
gamesPerPlayer = defaultdict(set)

for h in hoursTrain:
    uid = h[0]
    gid = h[1]
    playersPerGame[gid].add(uid)
    gamesPerPlayer[uid].add(gid)

# %%
pred_q3 = []
all_sim = []

for h in newValid:
    uid = h[0]
    gid = h[1]
    played = gamesPerPlayer[uid]
    similarities = []
    for gid2 in played:
        sim = Jaccard(playersPerGame[gid], playersPerGame[gid2])
        similarities.append(sim)
        all_sim.append(sim)
    if max(similarities) > 0.030:
        pred_q3.append(1)
    else: 
        pred_q3.append(0)


# %%
plt.boxplot(all_sim);

# %%
dummy = pd.DataFrame(data={'similarities': all_sim})
dummy.describe()

# %%
# Q3 accuracy

matches = 0
for i in range(len(actual_q1)):
    if actual_q1[i] == pred_q3[i]:
        matches += 1

acc_q3 = matches / len(actual_q1)

# %%
acc_q3

# %%
pred_q4 = []

return1 = filterMostPopular(0.55)

for h in newValid:
    uid = h[0]
    gid = h[1]
    played = gamesPerPlayer[uid]
    similarities = []
    for gid2 in played:
        sim = Jaccard(playersPerGame[gid], playersPerGame[gid2])
        similarities.append(sim)
    if max(similarities) > 0.03 and gid in return1:
        pred_q4.append(1)
    else: 
        pred_q4.append(0)

# %%
# Q4 accuracy

matches = 0
for i in range(len(actual_q1)):
    if actual_q1[i] == pred_q4[i]:
        matches += 1

acc_q4 = matches / len(actual_q1)

# %%
answers['Q3'] = acc_q3
answers['Q4'] = acc_q4

# %%
assertFloat(answers['Q3'])
assertFloat(answers['Q4'])

# %%
answers

# %%
return1 = filterMostPopular(0.55)
predictions = open("HWpredictions_Played.csv", 'w')
for l in open("data/pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    played = gamesPerPlayer[u]
    similarities = [0]
    for g2 in played:
        sim = Jaccard(playersPerGame[g], playersPerGame[g2])
        similarities.append(sim)
    if max(similarities) > 0.03 and g in return1:
        pred = 1
    else: 
        pred = 0
    
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()

# %%
answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"

# %%
answers

# %%
##################################################
# Hours played prediction                        #
##################################################

# %%
trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)

# %%
### Question 6

# %%
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)

for h in hoursTrain:
    uid = h[0]
    gid = h[1]
    hoursTransformed = h[2]['hours_transformed']
    hoursPerUser[uid].append(hoursTransformed)
    hoursPerItem[gid].append(hoursTransformed)

# %%
betaU['u91746794']

# %%
alpha = globalAverage # Could initialize anywhere, this is a guess

# %%
list(betaU.values())[0]

# %%
def MSE(actual, pred):
    return np.sum((actual-pred)**2) / len(actual)

# %%
alpha = globalAverage # Could initialize anywhere, this is a guess

# %%
betaU = {}
betaI = {}
for u in hoursPerUser:
    betaU[u] = 0

for g in hoursPerItem:
    betaI[g] = 0

# %%
trainHoursByPair = defaultdict(float)
itemsPerUser = defaultdict(set)
usersPerItem = defaultdict(set)

for h in hoursTrain:
    uid, gid, hours = h[0], h[1], h[2]['hours_transformed']
    trainHoursByPair[(uid, gid)] = hours
    itemsPerUser[uid].add(gid)
    usersPerItem[gid].add(uid)

# %%
# Alpha function
def calculate_alpha():
    numer = 0
    for pair in trainHoursByPair:
        u, g = pair[0], pair[1]
        numer += (trainHoursByPair[pair]-(betaU[u]+betaI[g]))
    denom = len(trainHours)
    return numer / denom

# BetaU function
def calculate_betaU(u, alpha, lamb):
    numer = 0
    for i in itemsPerUser[u]:
        numer += (trainHoursByPair[(u, i)]-(alpha+betaI[i]))
    denom = lamb + len(itemsPerUser[u])
    return numer / denom

# BetaI function
def calculate_betaI(i, alpha, lamb):
    numer = 0
    for u in usersPerItem[i]:
        numer += (trainHoursByPair[(u, i)]-(alpha+betaU[u]))
    denom = lamb + len(usersPerItem[i])
    return numer / denom

# Objective function
def calculate_objective(alpha, lamb):
    totalError = 0

    for pair in trainHoursByPair:
        pred = alpha + betaU[pair[0]] + betaI[pair[1]]
        actual = trainHoursByPair[pair]
        totalError += ((pred-actual)**2)

    regularizer = lamb*\
        (np.sum(np.array(list(betaU.values()))**2) + \
        np.sum(np.array(list(betaI.values()))**2))
        
    return totalError + regularizer

# %%
def iterate(lamb):

    # calculate alpha
    alpha_cd = calculate_alpha()

    # loop over users, calculate betaU, store in dictionary
    for p in hoursPerUser:
        bU_cd = calculate_betaU(p, alpha_cd, lamb)
        betaU[p] = bU_cd

    # loop over items, calculate betaI, store in dictionary
    for g in hoursPerItem:
        bI_cd = calculate_betaI(g, alpha_cd, lamb)
        betaI[g] = bI_cd

    return alpha_cd
        

# %%
# Coordinate Descent

objLog = []
lastObjective = None
bestObjective = None
iterations = 1000
tol = 1e-5
lamb = 1

for iter in range(iterations):
    alpha_ = iterate(lamb)
    currObjective = calculate_objective(alpha_, lamb)
    print(f'Iteration {iter+1}: Loss = {currObjective}')
    if lastObjective and abs(lastObjective-currObjective) < tol:
        bestObjective = currObjective
        break
    lastObjective = currObjective

# %%
# MSE q6

actual_q6 = [h[2]['hours_transformed'] for h in hoursValid]
preds_q6 = []

for h in hoursValid:
    user, game = h[0], h[1]
    pred = alpha_ + betaU[user] + betaI[game]
    preds_q6.append(pred)

validMSE = MSE(np.array(actual_q6), np.array(preds_q6))

# %%
answers['Q6'] = validMSE

# %%
assertFloat(answers['Q6'])

# %%
answers

# %%
### Question 7

# %%
betaUs = [(betaU[u], u) for u in betaU]
betaIs = [(betaI[i], i) for i in betaI]
betaUs.sort()
betaIs.sort()

print("Maximum betaU = " + str(betaUs[-1][1]) + ' (' + str(betaUs[-1][0]) + ')')
print("Maximum betaI = " + str(betaIs[-1][1]) + ' (' + str(betaIs[-1][0]) + ')')
print("Minimum betaU = " + str(betaUs[0][1]) + ' (' + str(betaUs[0][0]) + ')')
print("Minimum betaI = " + str(betaIs[0][1]) + ' (' + str(betaIs[0][0]) + ')')

# %%
answers['Q7'] = [betaUs[-1][0], betaUs[0][0], betaIs[-1][0], betaIs[0][0]]

# %%
answers['Q7']

# %%
assertFloatList(answers['Q7'], 4)

# %%
answers

# %%
### Question 8

# %%
# Coordinate Descent

lambdas = [0.5, 1.5, 2]
mseDict = defaultdict(float)
iterations = 1000
tol = 1e-3

actual_q8 = [h[2]['hours_transformed'] for h in hoursValid]

for lamb in lambdas:
    lastObjective = None
    bestObjective = None

    betaU = {}
    betaI = {}
    for u in hoursPerUser:
        betaU[u] = 0
    for g in hoursPerItem:
        betaI[g] = 0

    for iter in range(iterations):
        alpha_ = iterate(lamb)
        currObjective = calculate_objective(alpha_, lamb)
        print(f'Iteration {iter+1}: Loss = {currObjective}')
        if lastObjective and abs(lastObjective-currObjective) < tol:
            bestObjective = currObjective
            break
        lastObjective = currObjective

    preds = []
    for h in hoursValid:
        user, game = h[0], h[1]
        pred = alpha_ + betaU[user] + betaI[game]
        preds.append(pred)

    mse = MSE(np.array(actual_q8), np.array(preds))
    mseDict[lamb] = mse
    if mse < validMSE:
        print("Found lamb with lower MSE!!!")
        break

# %%
# Better lambda...

mseDict

# %%
validMSE = list(mseDict.values())[1]

# %%
answers['Q8'] = (list(mseDict.keys())[1], validMSE)

# %%
assertFloatList(answers['Q8'], 2)

# %%
predictions = open("HWpredictions_Hours.csv", 'w')
for l in open("data/pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    alpha = alpha_
    bu, bi = betaU[u], betaI[g]
    
    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')

predictions.close()

# %%
answers

# %%
f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %% [markdown]
# ### Unused Code Archive

# %%
# trainHoursByPair = defaultdict(float)

# for h in hoursTrain:
#     uid, gid, hours = h[0], h[1], h[2]['hours_transformed']
#     trainHoursByPair[(uid, gid)] = hours

# lamb = 1

# # BetaU
# betaUs = defaultdict(float)
# for u in gamesPerPlayer:
#     numer = []
#     for g in gamesPerPlayer[u]:
#         diff = trainHoursByPair[(u, g)] - (a + betaI[g])
#         numer.append(diff)
#     oneBetaU = sum(numer) / (lamb + len(gamesPerPlayer[u]))
#     betaUs[u] = oneBetaU

# # BetaI
# betaIs = defaultdict(float)
# for g in playersPerGame:
#     numer = []
#     for u in playersPerGame[g]:
#         diff = trainHoursByPair[(u, g)] - (a + betaU[u])
#         numer.append(diff)
#     oneBetaI = sum(numer) / (lamb + len(playersPerGame[g]))
#     betaIs[g] = oneBetaI

# regularizer = lamb*(sum(np.array(list(betaUs.values()))**2) \
#     + sum(np.array(list(betaIs.values()))**2))

# optimBetas = None
# minError = None

# for bu in betaUs.values():
#     for bi in betaIs.values():
#         se = []
#         for pair in trainHoursByPair:
#             se.append((a + bu + bi - trainHoursByPair[pair])**2)
#         error = sum(se) + regularizer
#         if minError is None or minError > error:
#             minError = error
#             optimBetas = (bu, bi)

# u91746794

        # def gradient_descent(iter=1000, lr=0.0001, tol=1e-8, lamb=1):

        #     # Initializing
        #     curr_alpha = alpha
        #     # curr_betaU = np.mean(list(betaU.values()))
        #     curr_betaU = random.choice(list(betaU.values()))
        #     # curr_betaI = np.mean(list(betaI.values()))
        #     curr_betaI = random.choice(list(betaI.values()))

        #     losses = []
        #     prev_loss = None

        #     # Estimating optimal params
        #     for i in range(iter):
        #         # iter_loss = []
        #         for u in hoursPerUser:
        #             length = len(hoursPerUser[u])
        #             actualHours = np.array(hoursPerUser[u])

        #             pred = [curr_alpha + curr_betaU + curr_betaI]*length
        #             # loss_bu = MSE(actualHours, np.array(pred))
        #             # loss.append(loss_bu)

        #             betaU_derivative = \
        #                 2*np.sum(np.array(pred)-actualHours) + 2*lamb*curr_betaU
                    
        #             next_betaU = curr_betaU - (lr * betaU_derivative)

        #             if abs(next_betaU - curr_betaU) < tol:
        #                 break

        #             curr_betaU = next_betaU

        #         for g in hoursPerItem:
        #             length = len(hoursPerItem[g])
        #             actualHours = np.array(hoursPerItem[g])

        #             pred = [curr_alpha + curr_betaU + curr_betaI]*length
        #             # loss_bi = MSE(actualHours, np.array(pred))
        #             # loss.append(loss_bi)

        #             betaI_derivative = \
        #                 2*np.sum(np.array(pred)-actualHours) + 2*lamb*curr_betaI
                    
        #             next_betaI = curr_betaI - (lr * betaI_derivative)

        #             if abs(next_betaI - curr_betaI) < tol:
        #                 break

        #             curr_betaI = next_betaI
                
        #         # Updating alpha
        #         length = len(trainHours)
        #         pred = [curr_alpha + curr_betaU + curr_betaI]*length
        #         # curr_loss = MSE(np.array(trainHours), np.array(pred))
        #         # iter_loss.append(loss)

        #         curr_loss = np.sum((np.array(pred)-np.array(trainHours))**2) + \
        #             lamb*(np.sum(np.array(list(betaU.values()))**2) + np.sum(np.array(list(betaI.values()))**2))

        #         alpha_derivative = \
        #             2*np.sum(np.array(pred)-np.array(trainHours))

        #         next_alpha = curr_alpha - (lr * alpha_derivative)

        #         # if np.sum(np.isnan(iter_loss)) > 0:
        #         #     nonNullLosses = []
        #         #     for l in loss:
        #         #         if not np.isnan(l):
        #         #             nonNullLosses.append(l)
                
        #         # curr_loss = np.mean(nonNullLosses)

        #         print(f"Iteration {i+1}: Loss {curr_loss}")

        #         # if prev_loss and abs(prev_loss-curr_loss) <= tol:
        #         if abs(next_alpha - curr_alpha) < tol:
        #             break

        #         prev_loss = curr_loss
        #         losses.append(curr_loss)
        #         curr_alpha = next_alpha

        #         # if i == 55: break

        #     return curr_alpha, curr_betaU, curr_betaI


# # Coordinate Descent

# objLog = []
# lastObjective = None
# bestObjective = None
# iterations = 100
# tol = 1e-5
# lamb = 1
# bu = 0
# bi = 0

# for iter in range(iterations):
#     alpha_cd = calculate_alpha(bu, bi)

#     if bestObjective is None:
#         bestObjective = calculate_objective(alpha_cd, bu, bi, lamb)
#         print(bestObjective)

#     for p in hoursPerUser:
#         bU_cd = calculate_betaU(p, alpha_cd, bi, lamb)
#         newObjective = calculate_objective(alpha_cd, bU_cd, bi, lamb)

#         if newObjective < bestObjective:
#             bu = bU_cd
#             bestObjective = newObjective
    
#     for g in hoursPerItem:
#         bI_cd = calculate_betaI(g, alpha_cd, bu, lamb)
#         newObjective = calculate_objective(alpha_cd, bu, bI_cd, lamb)

#         if newObjective < bestObjective:
#             bi = bI_cd
#             bestObjective = newObjective

#     # if lastObjective and abs(lastObjective-bestObjective) < tol:
#     #     break

#     lastObjective = bestObjective
#     objLog.append(lastObjective)

#     print(f'Iteration {iter+1}: Objective = {bestObjective}')

# userEncoder = {}

# counter = 0
# for u in hoursPerUser:
#     encoder = [0]*(len(hoursPerUser)-1)
#     if counter > 0:
#         encoder[counter-1] = 1
#     userEncoder[u] = encoder
#     counter += 1

# itemEncoder = {}

# counter = 0
# for g in hoursPerItem:
#     encoder = [0]*(len(hoursPerItem)-1)
#     if counter > 0:
#         encoder[counter-1] = 1
#     itemEncoder[g] = encoder
#     counter += 1

# X = [[1]+userEncoder[u[0]]+itemEncoder[u[1]] for u in trainHoursByPair]
# y = trainHours
# mdl = linear_model.SGDRegressor(fit_intercept=False, alpha=1)
# mdl.fit(X, y)
# preds_sklearn = mdl.predict(X[:5])
# mse_sklearn = MSE(np.array(trainHours[:5]), preds_sklearn)

# x2 = [[1, 1, 1]]*len(trainHours)
# mdl2 = linear_model.SGDRegressor(fit_intercept=False)
# mdl2.fit(x2, y)
# preds2 = mdl2.predict(x2)
# mse_sklearn2 = MSE(np.array(trainHours), preds2)


