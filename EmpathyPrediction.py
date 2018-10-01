import pandas as pd
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import KFold

import csv

recordList= []

with open('responses.csv') as csvFile:
    readCSV= csv.reader(csvFile, delimiter=',' )
    for row in readCSV:
        recordList.append(row)

data = pd.read_csv("responses.csv")

## Converting categorical Data to distinct value
print("!!!!!-----Conversion of Categorical Data to Numerical Data Starts-----!!!!!")
dataExtract = data.loc[:, ['Punctuality']]   
dataUniq = dataExtract["Punctuality"].unique()
for i in data["Punctuality"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1
dataExtract = data.loc[:, ['House - block of flats']]
dataUniq = dataExtract["House - block of flats"].unique()
for i in data["House - block of flats"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1   
dataExtract = data.loc[:, ['Village - town']]
dataUniq = dataExtract["Village - town"].unique()
for i in data["Village - town"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1
dataExtract = data.loc[:, ['Only child']]
dataUniq = dataExtract["Only child"].unique()
for i in data["Only child"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1
dataExtract = data.loc[:, ['Education']]
dataUniq = dataExtract["Education"].unique()
for i in data["Education"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1
dataExtract = data.loc[:, ['Left - right handed']]
dataUniq = dataExtract["Left - right handed"].unique()
for i in data["Left - right handed"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1
dataExtract = data.loc[:, ['Gender']]
dataUniq = dataExtract["Gender"].unique()
for i in data["Gender"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1
dataExtract = data.loc[:, ['Internet usage']]
dataUniq = dataExtract["Internet usage"].unique()
for i in data["Internet usage"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1
dataExtract = data.loc[:, ['Lying']]
dataUniq = dataExtract["Lying"].unique()
for i in data["Lying"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1
dataExtract = data.loc[:, ['Alcohol']]
dataUniq = dataExtract["Alcohol"].unique()
for i in data["Alcohol"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1
dataExtract = data.loc[:, ['Smoking']]
dataUniq = dataExtract["Smoking"].unique()
for i in data["Smoking"]:
    num = 1
    for j in dataUniq:
        if i == j:
            data.replace(i, num, inplace=True)
            break
        else:
            num = num + 1
            

print("!!!!!-----Conversion of Categorical Data to Numerical Data Ends-----!!!!!")

#Handling nan and NaN
print("!!!!!-----Conversion of ~NaN and ~nan to Most Frequent Data-----!!!!!")
data = data.replace("nan", np.nan)
data = data.replace("NaN", np.nan)
data_new = data
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp.fit(data_new)
data_new = imp.transform(data_new)

data = pd.DataFrame(data=data_new[:,:],index=[i for i in range(len(data_new))],columns=data.columns.tolist())

#Feature Engineering
print("!!!!!-----Feature Engineering Starts-----!!!!!")
counta = 0
countb = 0
countc = 0
countd = 0
counte = 0
countt = 0
dataExtract = data.loc[:, ['Height']]
dataUniq = dataExtract["Height"].unique()
for i in data["Height"]:
    countt = countt + 1
    if i > 130 and i <= 170:
        data.replace(i, 1, inplace=True)
        
    if i > 170 and i<= 185:
        data.replace(i, 2, inplace=True)
        countc = countb + 1
    if i > 185:
        data.replace(i, 3, inplace=True)
        countd = countc + 1

dataExtract = data.loc[:, ['Weight']]
dataUniq = dataExtract["Weight"].unique()
for i in data["Weight"]:
    countt = countt + 1
    if i > 40 and i <= 65:
        data.replace(i, 1, inplace=True)
       
    if i > 65 and i<= 80:
        data.replace(i, 2, inplace=True)
        countc = countb + 1
    if i > 80:
        data.replace(i, 3, inplace=True)
        countd = countc + 1

dataExtract = data.loc[:, ['Age']]
dataUniq = dataExtract["Age"].unique()
for i in data["Age"]:
    countt = countt + 1
    if i > 10 and i <= 23:
        data.replace(i, 1, inplace=True)
        counta = counta + 1
    
    if i > 23 and i <= 27:
        data.replace(i, 2, inplace=True)
        countd = countc + 1
    if i > 27:
        data.replace(i, 3   , inplace=True)
        countd = countc + 1

print("!!!!!-----Feature Engineering Ends-----!!!!!")

index = data.columns.get_loc("Empathy")
#y = [i for i in data.iloc[:, index]]
y = data.iloc[:, index].values
#y_train = y[:int(len(y)*0.9)]
#y_test = y[int(len(y)*0.9):]

data_train = data.drop(["Empathy"], axis=1)
x = data_train.iloc[:, :].values
#x_train = x[:int(len(x)*0.9)]
#x_test = x[int(len(x)*0.9):]

m = KFold(n_splits=10)
m.get_n_splits(x)
print(m)
gnb = GaussianNB()
clf = svm.SVC()
AccuracyGNB = 0
AccuracySVC = 0
for train_index, test_index in m.split(x):
    x_train = x[train_index]
    x_test = x[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    #print(train_index)
    #print(test_index)
    x_train, x_test  = x[train_index], x[test_index]
    y_train, y_test  = y[train_index], y[test_index]
    model = gnb.fit(x_train, y_train)
    preds = gnb.predict(x_test)
    AccuracyGNB = AccuracyGNB + accuracy_score(y_test, preds)
    #print("Accuracy with Gaussian Naive Baise- ",accuracy_score(y_test, preds))
    clf.fit(x_train, y_train)
    preds2 = clf.predict(x_test)
    AccuracySVC = AccuracySVC + accuracy_score(y_test, preds2)
    #print("Accuracy with SVM- ",accuracy_score(y_test, preds2))
print("Accuracy with Gaussian Naive Baise- ",AccuracyGNB/10)
print("Accuracy with SVM- ",AccuracySVC/10)


df1 = data[['team','x','y','outcome']]
for a, b, c, d in zip[df1.team, df1.x, df1.y, df1.outcome]:
    if b > 0 and b < 35 and c > 0 and c < 17:
        dictClub[a]["RB"] += 1



