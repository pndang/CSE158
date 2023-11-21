# %%
import gzip
from collections import defaultdict
import math
import numpy as np
import string
import random
import os

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
  if count > totalPlayed*0.72: break

# %%
def accuracy(actual, pred):
    matches = 0

    for i in range(len(actual)):
        if actual[i] == pred[i]:
            matches += 1

    return matches / len(actual)

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
allGamesTrain = []

for h in hoursTrain:
    uid = h[0]
    gid = h[1]
    playersPerGame[gid].add(uid)
    gamesPerPlayer[uid].add(gid)
    if gid not in allGamesTrain:
        allGamesTrain.append(gid)

actualPlay = []

for pair in newValid:
    uid = pair[0]
    gid = pair[1]

    if gid in gamesPerUser[uid]:
        actualPlay.append(1)
    else:
        actualPlay.append(0)

# %%
hoursTrain[0]

# %%
return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalPlayed*0.76: break

# %%
# Played prediction

predsMod = []
rankingByUser = defaultdict(dict)

for h in newValid:

    if len(rankingByUser) == 0:
        for h in newValid:
            u, i = h[0], h[1]
            points = 0
            played = gamesPerPlayer[u]
            similarities = [0]
            for i2 in played:
                if i == i2:
                    continue
                sim = Jaccard(playersPerGame[i], playersPerGame[i2])
                similarities.append(sim)
            if max(similarities) > 0.021:
                points += 0.5
            if i in return1:
                points += 0.5
            rankingByUser[u][i] = points

    u, i = h[0], h[1]

    if (u not in rankingByUser.keys()) and (i not in playersPerGame.keys()):
        pred = random.choice([0, 1])
    elif u not in rankingByUser.keys():
        pred = 1 if i in return1 else 0
    elif i not in playersPerGame.keys():
        pred = random.choice([0, 1])
    else:
        if isinstance(rankingByUser[u], dict):
            rankingByUser[u] = sorted(rankingByUser[u].items(), key=lambda x: x[1], reverse=True)
            rankingByUser[u] = [_[0] for _ in rankingByUser[u]]
        
        if i in rankingByUser[u][:len(rankingByUser[u])//2]:
            pred = 1
        else:
            pred = 0
    
    predsMod.append(pred)
    

# %%
accuracyMod = accuracy(actualPlay, predsMod)

# %%
rankingByUser = defaultdict(dict)
itemPopularity = defaultdict(int)

for h in open("data/pairs_Played.csv"):
    if h.startswith("userID"):
        continue
    u,i = h.strip().split(',')
    itemPopularity[i] += 1

itemPopularity = \
    sorted(itemPopularity.items(), key=lambda x: x[1], reverse=True)
itemPopularity = [_[0] for _ in itemPopularity]

for h in open("data/pairs_Played.csv"):
    if h.startswith("userID"):
        continue
    u,i = h.strip().split(',')
    points = 0
    itemSimilarities = [0]
    userSimilarities = [0]
    played = gamesPerPlayer[u]
    for i2 in played:
        if i == i2:
            continue
        sim = Jaccard(playersPerGame[i], playersPerGame[i2])
        itemSimilarities.append(sim)
    played = playersPerGame[i]
    for u2 in played:
        if u == u2:
            continue
        sim = Jaccard(gamesPerPlayer[u], gamesPerPlayer[u2])
        userSimilarities.append(sim)
    if max(itemSimilarities) > 0.032:
        points += 0.5
    if max(userSimilarities) > 0.05:
        points += 0.5
    if i in return1:
        points += 0.77
    if i in itemPopularity[:len(itemPopularity)//8]:
        points += 1.2
    elif i in itemPopularity[len(itemPopularity)//8: len(itemPopularity)//6]:
        points += 1.0
    elif i in itemPopularity[len(itemPopularity)//6: len(itemPopularity)//4]:
        points += 0.8
    elif i in itemPopularity[len(itemPopularity)//4: len(itemPopularity)//2]:
        points += 0.6
    elif i in itemPopularity[len(itemPopularity)//2: int(len(itemPopularity)*0.85)]:
        points += 0.4
    elif i in itemPopularity[int(len(itemPopularity)*0.85):]:
        points += 0.2
    rankingByUser[u][i] = points

# %%
predictions = open("predictions_Played.csv", 'w')
for l in open("data/pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,i = l.strip().split(',')
    
    # Logic...
    if (u not in gamesPerPlayer.keys()) and (i not in playersPerGame.keys()):
        pred = random.choice([0, 1])
    elif u not in gamesPerPlayer.keys():
        pred = 1 if i in return1 else 0
    elif i not in playersPerGame.keys():
        pred = random.choice([0, 1])
    else:
        if isinstance(rankingByUser[u], dict):
            rankingByUser[u] = sorted(rankingByUser[u].items(), key=lambda x: x[1], reverse=True)
            rankingByUser[u] = [datum[0] for datum in rankingByUser[u]]
        
        if i in rankingByUser[u][:len(rankingByUser[u])//2]:
            pred = 1
        else:
            pred = 0
            
    _ = predictions.write(u + ',' + i + ',' + str(pred) + '\n')

predictions.close()

# %%
##################################################
# Hours played prediction                        #
##################################################

# %%
def MSE(actual, pred):
    return np.sum((actual-pred)**2)/len(actual)

# %%
trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)

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
alpha = globalAverage # Could initialize anywhere, this is a guess

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
betaU = {}
betaI = {}
for u in hoursPerUser:
    # betaU[u] = 1.1666666666666665
    betaU[u] = 0.6

for g in hoursPerItem:
    # betaI[g] = 2.811111111111111
    betaI[g] = 0.9

# %%
# Coordinate Descent

lossLog = {}
lastLoss = None
bestLoss = None
iterations = 1000
tol = 1e-8
lamb = 5

for iter in range(iterations):
    alpha_ = iterate(lamb)
    preds = []
    for h in hoursValid:
        user, game = h[0], h[1]
        pred = alpha_+betaU[user]+betaI[game]
        preds.append(pred)

    currLoss = MSE(np.array(actuals), np.array(preds))
    print(f"Iteration {iter+1}: Loss {currLoss}")

    if (lastLoss and abs(lastLoss-currLoss) < tol) or \
        (lastLoss and (lastLoss <= currLoss)):
        lossLog[iter+1] = currLoss
        break

    lastLoss = bestLoss = currLoss
    lossLog[iter+1] = currLoss

# %%
# actuals = [h[2]['hours_transformed'] for h in hoursValid]

# # # initBus = np.linspace(0.5, 1.5, 10)
# # # initBis = np.linspace(2.3, 3.3, 10)
# initBus = [1.1666666666666665]
# initBis = [2.811111111111111]
# lambdas = np.linspace(4.5, 5.5, 10)
# hyperParams = defaultdict(float)
# lowestValidMSE = None

# for init_bu in initBus:
#     for init_bi in initBis:
#         for lamb in lambdas:
#             betaU = {}
#             betaI = {}
#             for u in hoursPerUser:
#                 betaU[u] = init_bu

#             for g in hoursPerItem:
#                 betaI[g] = init_bi

#             # Coordinate Descent
#             lastLoss = None
#             bestLoss = None
#             iterations = 1000
#             tol = 1e-8
#             lamb = lamb

#             for iter in range(1000):
#                 alpha_ = iterate(lamb)
#                 preds = []
#                 for h in hoursValid:
#                     user, game = h[0], h[1]
#                     pred = alpha_+betaU[user]+betaI[game]
#                     preds.append(pred)

#                 currLoss = MSE(np.array(actuals), np.array(preds))

#                 if (lastLoss and abs(lastLoss-currLoss) < tol) or \
#                     (lastLoss and (lastLoss <= currLoss)):
#                     # hyperParams[(init_bu, init_bi)] = currLoss
#                     hyperParams[lamb] = currLoss
#                     if lowestValidMSE is None:
#                         lowestValidMSE = currLoss
#                     else:
#                         if currLoss < lowestValidMSE:
#                             lowestValidMSE = currLoss
#                             # print(f'Found lower: bu = {init_bu}, bi = {init_bi}, MSE = {currLoss}')
#                             print(f'Found lower: lamb: {lamb}, MSE = {currLoss}')
#                     break

#                 lastLoss = bestLoss = currLoss

# %%
# sorted(hyperParams.items(), key=lambda x:x[1])

# %%
# Get MSE

actuals = [h[2]['hours_transformed'] for h in hoursValid]
preds = []

for h in hoursValid:
    user, game = h[0], h[1]
    pred = alpha_ + betaU[user] + betaI[game]
    preds.append(pred)

validMSE = MSE(np.array(actuals), np.array(preds))

# %%
alpha = globalAverage

# Take betaUs and betaIs from coordinate descent base model
gdBetaU = betaU.copy()
gdBetaI = betaI.copy()

# Initialize gammaUs and gammaIs
gammaU = {}
gammaI = {}
for u in hoursPerUser:
    gammaU[u] = np.array([-0.155])
for g in hoursPerItem:
    gammaI[g] = np.array([0.155])

# %%
def betaUDerivative(u, alpha, lamb):
    
    error = 0
    for i in itemsPerUser[u]:
        gamma = np.sum(gammaU[u]*gammaI[i])
        error += (alpha+gdBetaU[u]+gdBetaI[i]+gamma - trainHoursByPair[(u,i)])
    regularizer = 2*lamb*gdBetaU[u]

    return 2*error + regularizer

def betaIDerivative(i, alpha, lamb):

    error = 0
    for u in usersPerItem[i]:
        gamma = np.sum(gammaU[u]*gammaI[i])
        error += (alpha+gdBetaU[u]+gdBetaI[i]+gamma - trainHoursByPair[(u,i)])
    regularizer = 2*lamb*betaI[i]

    return 2*error + regularizer

def alphaDerivative(alpha):

    error = 0
    for pair in trainHoursByPair:
        u,i = pair[0], pair[1]
        gamma = np.sum(gammaU[u]*gammaI[i])
        error += (alpha+gdBetaU[u]+gdBetaI[i]+gamma - trainHoursByPair[pair])
    
    return 2*error

def gammaUDerivative(u, idx, alpha, lamb):

    error = 0
    for item in itemsPerUser[u]:
        error += (gammaI[item][idx] * \
            (alpha+gdBetaU[u]+gdBetaI[item]+np.sum(gammaU[u]*gammaI[item])) - trainHoursByPair[(u,item)]) 
    regularizer = 2*lamb*gammaU[u][idx]

    return 2*error + regularizer

def gammaIDerivative(i, idx, alpha, lamb):

    error = 0
    for user in usersPerItem[i]:
        error += (gammaU[user][idx] * \
            (alpha+gdBetaU[user]+gdBetaI[i]+np.sum(gammaU[user]*gammaI[i])) - trainHoursByPair[(user,i)]) 
    regularizer = 2*lamb*gammaI[i][idx]

    return 2*error + regularizer

def objectiveFunction(alpha, lamb):
    totalError = 0

    for pair in trainHoursByPair:
        gamma = np.sum(gammaU[pair[0]]*gammaI[pair[1]])
        pred = alpha+gdBetaU[pair[0]]+gdBetaI[pair[1]]+gamma
        actual = trainHoursByPair[pair]
        totalError += ((pred-actual)**2)

    betaRegularizer = \
        (np.sum(np.array(list(betaU.values()))**2) + \
         np.sum(np.array(list(betaI.values()))**2))
    
    lambRegularizer = \
        np.sum([np.sum(val**2) for val in list(gammaU.values())]) + \
        np.sum([np.sum(val**2) for val in list(gammaI.values())])

    return totalError + lamb*(betaRegularizer+lambRegularizer)

# %%
actuals = [h[2]['hours_transformed'] for h in hoursValid]

# %%
def gradient_descent(lr=0.000001, tol=0.0001, lamb=1, k=1):
    
    # Initializing
    ite = 0
    currAlpha = alpha_
    losses = []
    prevLoss = None

    # Estimating optimal params
    while True:
        
        # alpha_derivative = alphaDerivative(currAlpha)
        # nextAlpha = currAlpha-(lr*alpha_derivative)

        for u in hoursPerUser:
            # betaU_derivative = betaUDerivative(u, currAlpha, lamb)
            # nextBetaU = gdBetaU[u]-(lr*betaU_derivative)
            # gdBetaU[u] = nextBetaU

            for idx in range(k):
                gammaU_derivative = gammaUDerivative(u, idx, currAlpha, lamb)
                nextGammaU = gammaU[u][idx]-(lr*gammaU_derivative)
                gammaU[u][idx] = nextGammaU

        for i in hoursPerItem:
            # betaI_derivative = betaIDerivative(i, currAlpha, lamb)
            # nextBetaI = gdBetaI[i]-(lr*betaI_derivative)
            # gdBetaI[i] = nextBetaI

            for idx in range(k):
                gammaI_derivative = gammaIDerivative(i, idx, currAlpha, lamb)
                nextGammaI = gammaI[i][idx]-(lr*gammaI_derivative)
                gammaI[i][idx] = nextGammaI

        preds = []
        for h in hoursValid:
            user, game = h[0], h[1]
            gamma = np.sum(gammaU[user]*gammaI[game])
            pred = currAlpha+gdBetaU[user]+gdBetaI[game]+gamma
            preds.append(pred)

        currLoss = np.sum((np.array(actuals)-np.array(preds))**2)/len(actuals)
        print(f"Iteration {ite+1}: Loss {currLoss}")

        if (prevLoss and abs(currLoss-prevLoss) < tol) or \
            (prevLoss and prevLoss < currLoss):
            break

        prevLoss = currLoss
        losses.append(currLoss)
        # currAlpha = nextAlpha
        ite += 1
        
        # if ite == 5: break

    return currAlpha, currLoss

# %%
optimAlpha = gradient_descent(lamb=0.2, k=1)

# %%
# 2.9978457265036615 -1.0  -1.0     3.0587508633547
# 2.991672624529245  -0.5  -0.5     3.056969245065207
# 2.989965524554382  -0.22  -0.22   3.057170372406158

# 2.997774547040354   -0.77  0.77   3.055212314924871

# 2.9964264348542344  -0.7  0.7     3.054955090634088
# 2.995398728408632   -0.63  0.63   3.054961197744413
# 2.9943585372209474  -0.55  0.55   3.055019273278388
# 2.9917683990644663  -0.3  0.3     3.055258034133135
# 2.991065330904339   -0.2  0.2     3.055437138287312
# 2.9909620023556527  -0.18  0.18
# 2.990900247877377  -0.17  0.17
# 2.9908411275415325  -0.16   0.16  3.05547485237682
# 2.9907845516814255  -0.15  0.15   
# 2.9907668494931734  -0.14  0.14   3.055452604188862
# 2.9906022911685444   -0.11  0.11  3.055551376120019
# 2.990434912056805   -0.05   0.05  3.055578947029252
# 2.9903220093292826  -0.01   0.01  3.05565491385999
# 2.9903063186285  0.0  0.0         3.055661490842236
# 2.9902911461248585  0.01   0.01   3.055668765103901
# 2.9902578584770048  0.05   0.05
# 2.9900533611526776  0.09   0.09
# 2.989699330120003  0.15  0.15
# 2.989535854391708  0.2  0.2
# 2.989916741531046  1.0  1.0       3.057206512701083

# %%
lambdas = np.linspace(3, 6, 33)

# %%
lambLog = defaultdict(float)
bestLoss = None

for lamb in lambdas:

    # Take betaUs and betaIs from coordinate descent base model
    gdBetaU = betaU.copy()
    gdBetaI = betaI.copy()

    # Initialize gammaUs and gammaIs
    gammaU = {}
    gammaI = {}
    for u in hoursPerUser:
        gammaU[u] = np.array([0.0])
    for g in hoursPerItem:
        gammaI[g] = np.array([0.0])

    a, loss = gradient_descent(lamb=lamb, k=1)

    if bestLoss is None:
        bestLoss = loss
    elif loss < bestLoss:
        bestLoss = loss
        print(f'Lower MSE found: lambda = {lamb}, MSE = {loss}')

    lambLog[lamb] = loss
    

# %%
threeToSix = lambLog.copy()

# %%

#         0.04  1  1   0    3.060395   0.05
#         0.08  1  1   0    3.065877   0.08
#         0.01 0  4    0    3.056916   0.02
#         0.5 0  0.5   0    3.132520   0.2
#         -1 0.1 0.1  0.1   3.063503   -0.1
#         -1  0  0.1   0    3.065779   -0.1
#         1  0  0.1   0     3.069938828333021    0.01
#         -1  1  0.01  0    3.055440340866809   -0.02
#         -1  0  0.1  0     3.065779707448912   -0.1


# -0.0001 0 1  0    3.055402507271341   -0.0001
# -0.001 0  1  0    3.055359970029504   -0.001
# -1  0  0.01  0    3.0554273115071    -0.02
# -0.1  0  0  0     3.055401558027065   -0.1
         
#                   3.055533306202107   -0.04
#                   3.054807047932062   -0.03 (-0.17*0.17 = -0.0289)
#                   3.054755589797452   -0.02
#                   3.054762917313528   -0.019
#                   3.054826318614454   -0.015
# -1  0  0  0       3.055268852878444   -0.01
# -1  1  0  0       3.055269832312217   -0.01
# -0.0001 0 1  0    3.055402507271341   -0.0001
# 0   0   0   0     3.055412186381949   0
# 0.0001 0 1  0     3.055412186381949   0.0001
# 0.001 0  1  0     3.055456761064895   0.001
# -0.1 -0  -0.1  0  3.055793725275343   0.01
# 0.01  0  1  0   3.055994157358141   0.01
# 0.1  0  4  0  3.206790051752321   0.4
# 1  0.4  1  2  6.343127679944726     1.8
# 5.933941182781103   1.6
# 1  0.1  2  0.1  7.138106418044735   2.01


# -0.1 -0.1 3.056309659631276
# -0.2  0 3.058233876032432


# 0.03  3.100596221721324
# 0.05  3.341260986368183

# 2.988886301738373  0  0.5 l=0.02 k=2
# 2.9897519900660687  0.2
# 2.9897548967990804  1
# 2.9897522272849266  0.1    0 gamma
# 2.99073011929595   2.5

# %%
# Get MSE

actuals = [h[2]['hours_transformed'] for h in hoursTrain]
preds = []

for h in hoursTrain:
    user, game = h[0], h[1]
    gamma = np.sum(gammaU[user]*gammaI[game])
    pred = optimAlpha + gdBetaU[user] + gdBetaI[game] + gamma
    preds.append(pred)

trainMSE = np.sum((np.array(actuals)-np.array(preds))**2)/len(actuals)
trainMSE

# %%
# Get MSE

actuals = [h[2]['hours_transformed'] for h in hoursValid]
preds = []

for h in hoursValid:
    user, game = h[0], h[1]
    gamma = np.sum(gammaU[user]*gammaI[game])
    pred = optimAlpha + gdBetaU[user] + gdBetaI[game] + gamma
    preds.append(pred)

validMSE = np.sum((np.array(actuals)-np.array(preds))**2)/len(actuals)
validMSE

# %%
predictions = open("predictions_Hours.csv", 'w')
for l in open("data/pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    alpha = alpha_
    gamma = np.sum(gammaU[u]*gammaI[g])
    bu, bi = betaU[u], betaI[g]
    
    _ = predictions.write(u + ',' + g + ',' + str(alpha+bu+bi+gamma) + '\n')

predictions.close()

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

# ranks = defaultdict(dict)

# for u in gamesPerPlayer:
#     pointsDict = defaultdict(dict)
#     points = 0
#     for i in allGames:
#         # Item-item similarity (Jaccard)
#         similarities = []
#         played = gamesPerPlayer[u]
#         for i2 in played:
#             sim = Jaccard(playersPerGame[i], playersPerGame[i2])
#             similarities.append(sim)
#         points += max(similarities)
#         if i in return1:
#             points += (points*1/2)
#         pointsDict[i] = points
#     ranks[u] = pointsDict      

# def makePlayPrediction(u, i):
#     pointsDict = defaultdict(dict)
#     points = 0

#     for g in allGames:
#         # Item-item similarity (Jaccard)
#         similarities = []
#         played = gamesPerPlayer[u]
#         for g2 in played:
#             sim = Jaccard(playersPerGame[g], playersPerGame[g2])
#             similarities.append(sim)
#         points += max(similarities)
#         if g in return1:
#             points += (points*1/2)
#         pointsDict[g] = points
    
#     sortedItems = sorted(pointsDict, reverse=True)
#     if i in sortedItems[:len(sortedItems)//2]:
#         return 1
#     return 0

# # Assign ids 0 to [number of users]-1 to users in train set (to be used as one-hot encoding indices)

# userIndex = 0
# userIdDict = defaultdict(dict)
# for u in gamesPerPlayer:
#     userIdDict[u] = userIndex
#     userIndex += 1

# # Assign ids 0 to [number of games]-1 to games in train set (to be used as one-hot encoding indices)

# gameIndex = 0
# gameIdDict = defaultdict(dict)
# for g in playersPerGame:
#     gameIdDict[g] = gameIndex
#     gameIndex += 1

# def featPlay(datum):
#     feature = [1]

#     # One-hot encode user id
#     uid = datum[0]
#     uidEncode = [0]*(len(gamesPerPlayer)-1)
#     try:
#         uidEncode[userIdDict[uid]] = 1
#     except:
#         pass

#     # One-hot encode game id
#     gid = datum[1]
#     gidEncode = [0]*(len(playersPerGame)-1)
#     try:
#         gidEncode[gameIdDict[gid]] = 1
#     except:
#         pass

#     # Get max Jaccard similarity 
#     played = gamesPerPlayer[uid]
#     similarities = []
#     for gid2 in played:
#         if gid == gid2:
#             continue
#         sim = Jaccard(playersPerGame[gid], playersPerGame[gid2])
#         similarities.append(sim)
#     try:
#         similarity = [np.average(similarities)]
#     except:
#         similarity = [0]

#     # Whether game is among most popular games
#     popular = 0
#     if gid in return1:
#         popular = 1
    
#     return feature+uidEncode+gidEncode+similarity+[popular]

# # Create training set with negative samples

# newTrain = []
# for h in hoursTrain[:len(hoursTrain)//4]:
#     uid = h[0]
#     haveNotPlayed = []
#     for g in allGames:
#         if g not in gamesPerUser[uid]:
#             haveNotPlayed.append(g)
#     newTrain.append((uid, h[1]))
#     newTrain.append((uid, random.choice(haveNotPlayed)))

# # From new train set created above, get corresponding play status 

# newTrainPlayStatus = []
# for h in newTrain:
#     uid, gid = h[0], h[1]
#     if gid in gamesPerPlayer[uid]:
#         newTrainPlayStatus.append(1)
#     else:
#         newTrainPlayStatus.append(0)

# # Transform train set into features

# XTrain = []
# for h in newTrain:
#     feat = featPlay(h)
#     XTrain.append(feat)

# # Fit model

# mdlPlay = linear_model.LogisticRegression(fit_intercept=False, class_weight='balanced')
# mdlPlay.fit(XTrain, newTrainPlayStatus)

# # Transform validation set into features

# XValid = []
# for h in newValid:
#     feat = featPlay(h)
#     XValid.append(feat)

# # Make predictions

# preds = mdlPlay.predict(XValid)

# # Get accuracy

# accuracyPlay = accuracy(actualPlay, preds)

# accuracyPlay

# predsOriginal = []

# for h in newValid:
#     u, i = h[0], h[1]

#     if u not in list(gamesPerUser.keys()):
#         print('test')
#     if i not in list(playersPerGame.keys()):
#         print('test1')

#     played = gamesPerPlayer[u]
#     similarities = [0]
#     for i2 in played:
#         if i == i2:
#             continue
#         sim = Jaccard(playersPerGame[i], playersPerGame[i2])
#         similarities.append(sim)
#     if max(similarities) > 0.021 and i in return1:
#         pred = 1
#     else: 
#         pred = 0
#     predsOriginal.append(pred)

# Coordinate Descent

# lambdas = [0.5, 1.5, 2]
# mseDict = defaultdict(float)
# iterations = 1000
# tol = 1e-3

# actual_q8 = [h[2]['hours_transformed'] for h in hoursValid]

# for lamb in lambdas:
#     lastObjective = None
#     bestObjective = None

#     betaU = {}
#     betaI = {}
#     for u in hoursPerUser:
#         betaU[u] = 0
#     for g in hoursPerItem:
#         betaI[g] = 0

#     for iter in range(iterations):
#         alpha_ = iterate(lamb)
#         currObjective = calculate_objective(alpha_, lamb)
#         print(f'Iteration {iter+1}: Loss = {currObjective}')
#         if lastObjective and abs(lastObjective-currObjective) < tol:
#             bestObjective = currObjective
#             break
#         lastObjective = currObjective

#     preds = []
#     for h in hoursValid:
#         user, game = h[0], h[1]
#         pred = alpha_ + betaU[user] + betaI[game]
#         preds.append(pred)

#     mse = MSE(np.array(actual_q8), np.array(preds))
#     mseDict[lamb] = mse
#     if mse < validMSE:
#         print("Found lamb with lower MSE!!!")
#         break

# %%
# Working gradient descent code for base model

# def betaUDerivative(u, alpha, lamb):
    
#     error = 0
#     for i in itemsPerUser[u]:
#         error += (alpha+gdBetaU[u]+gdBetaI[i] - trainHoursByPair[(u,i)])
#     regularizer = 2*lamb*gdBetaU[u]

#     return 2*error + regularizer

# def betaIDerivative(i, alpha, lamb):

#     error = 0
#     for u in usersPerItem[i]:
#         error += (alpha+gdBetaU[u]+gdBetaI[i] - trainHoursByPair[(u,i)])
#     regularizer = 2*lamb*betaI[i]

#     return 2*error + regularizer

# def alphaDerivative(alpha):

#     error = 0
#     for pair in trainHoursByPair:
#         u,i = pair[0], pair[1]
#         error += (alpha+gdBetaU[u]+gdBetaI[i] - trainHoursByPair[pair])
    
#     return 2*error

# def gradient_descent(lr=0.000001, tol=0.00001, lamb=1):

#     # Initializing
#     ite = 0
#     currAlpha = alpha
#     losses = []
#     prevLoss = None

#     # Estimating optimal params
#     while True:
        
#         alpha_derivative = alphaDerivative(currAlpha)
#         nextAlpha = currAlpha-(lr*alpha_derivative)

#         for u in hoursPerUser:
#             betaU_derivative = betaUDerivative(u, currAlpha, lamb)
#             # if u == 'u70666506':
#             #     print('test')
#             #     print(betaU_derivative)
#             # return ''
#             nextBetaU = betaU[u]-(lr*betaU_derivative)
#             betaU[u] = nextBetaU

#         for i in hoursPerItem:
#             betaI_derivative = betaIDerivative(i, currAlpha, lamb)
#             nextBetaI = betaI[i]-(lr*betaI_derivative)
#             betaI[i] = nextBetaI

#         # currLoss = calculate_objective(currAlpha, lamb)
#         preds = []
#         for h in hoursValid:
#             user, game = h[0], h[1]
#             pred = currAlpha + betaU[user] + betaI[game]
#             preds.append(pred)

#         currLoss = np.sum((np.array(actuals)-np.array(preds))**2)/len(actuals)
#         print(f"Iteration {ite+1}: Loss {currLoss}")

#         if prevLoss and abs(currLoss-prevLoss) < tol:
#             break

#         prevLoss = currLoss
#         losses.append(currLoss)
#         currAlpha = nextAlpha
#         ite += 1
        
#         if ite == 22: break

#     return currAlpha


