#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:24:40 2021

@author: johann
"""

#imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
#user created function to calculate the 95 confidence interval of data set
def conf(data):
  n = len(data)-1
  s = np.std(data)
  z = 1.96
  ci = ((z * s) / np.sqrt(n))
  return ci

#%% get data
columnNames = ["timestamp", "company", "level", "title", "yearly_compensation", "location", "years_of_experience", "years_at_company", "tag", "base_salary", "stock_grant_value", "bonus", "gender", "other_details", "city_id", "dmaid", "row_number", "masters", "bachelors", "doctorate", "high_school", "some_college", "asian", "white", "two_or_more", "black", "hispanic", "race", "education"]
faangNames = ["Facebook", "Amazon", "Apple", "Netflix", "Google"]
dataframe = pd.read_csv('data.csv',header=None).dropna()
dataframe.drop(index=dataframe.index[0], 
        axis=0, 
        inplace=True)

dataframe.columns = columnNames
d = dataframe
#%% data peeking
#print(dataframe[["yearly_compensation","education"]])

#%% Total yearly compensation histogram
dataset = pd.read_csv("data.csv").dropna()
dataset.head()

#dataset['totalyearlycompensation'] = np.log(dataset['totalyearlycompensation'])

plt.figure()
plt.hist(x='totalyearlycompensation', data = dataset, bins=np.arange(10000, 1500000, 25000))
plt.title("Histogram of total yearly compensation")
plt.xlabel("total yearly compensation")
plt.ylabel("Number of people")
plt.show()
#%% t_testing the mean


faang = d[d["company"].isin(faangNames)][["yearly_compensation","years_at_company"]]
#print(type(faang))
nonFaang = d[~d["company"].isin(faangNames)][["yearly_compensation","years_at_company"]]
faang = np.array(faang)
faang = faang.astype(float)
#print(faang.mean(axis=0))
nonFaang = np.array(nonFaang)
nonFaang = nonFaang.astype(float)
print(nonFaang.mean(axis=0))

print('\n')
print(stats.ttest_ind(faang[:,0],nonFaang[:,0]))
print(stats.ttest_ind(faang[:,1],nonFaang[:,1]))
#%% correlation between salary and years of exp vs salary and years at company
salaries = dataframe[["yearly_compensation","years_of_experience","years_at_company"]]
#print(salaries.head(20))
d = np.array(salaries)
d = d.astype(float)
#print(d)
x, trash1 = stats.pearsonr(d[:,1], d[:,0])
y, trash2 = stats.pearsonr(d[:,2], d[:,0])
x = round(x, 4)
y = round(y, 4)
stringX = "Pearson corr. with total years: " + str(x)
stringY = "Pearson corr. with years at company: " + str(y)
plt.scatter(d[:,1], d[:,0], marker='.', color='b')
plt.scatter(d[:,2], d[:,0], marker='.', color='r')
plt.xlabel("Years")
plt.ylabel("Salary")
plt.title('Correlation between salary and years of experience vs salary and years at company')
plt.text(30,60000,stringX)
plt.text(30,40000,stringY)
plt.legend(["Salary against Total Years","Salary against Years at Company"])
plt.yscale('log')
plt.figure()
plt.show()
#%% correlation between education level and yearly compensation
x = dataframe[["yearly_compensation","education"]]
hSals = np.array(x[x["education"] == "Highschool"][["yearly_compensation"]]).astype(int)
scSals = np.array(x[x["education"] == "Some College"][["yearly_compensation"]]).astype(int)
bSals = np.array(x[x["education"] == "Bachelor's Degree"][["yearly_compensation"]]).astype(int)
mSals = np.array(x[x["education"] == "Master's Degree"][["yearly_compensation"]]).astype(int)
dSals = np.array(x[x["education"] == "PhD"][["yearly_compensation"]]).astype(int)
hMean = hSals.mean()
scMean = scSals.mean()
bMean = bSals.mean()
mMean = mSals.mean()
dMean = dSals.mean()

hCI = conf(hSals.flatten())
scCI = conf(scSals.flatten())
bCI = conf(bSals.flatten())
mCI = conf(mSals.flatten())
dCI = conf(dSals.flatten())

#creating plot
plt.errorbar(["High School","Some College","Bachelor's","Master's","PhD"],[hMean,scMean,bMean,mMean,dMean],[hCI,scCI,bCI,mCI,dCI], marker='d', capsize=5,linestyle='')
plt.xlabel("Education Level")
plt.ylabel("Compensation")
plt.xticks()
#plt.ylim(0,25000)
plt.title('Fig: Yearly Compensation by Level of Education (mean & 95% CI)')
plt.figure()
plt.show()


