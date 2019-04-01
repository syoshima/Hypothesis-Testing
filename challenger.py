# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:24:50 2019

@author: Samy Abud Yoshima
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from patsy import dmatrices
import statsmodels.discrete.discrete_model as sm
data = pd.read_csv("challenger-data.csv")

# subsetting data
failures = data.loc[(data.Y == 1)]
no_failures = data.loc[(data.Y == 0)]

# frequencies
failures_freq = failures.X.value_counts()#failures.groupby('X')
no_failures_freq = no_failures.X.value_counts()
# plotting


fig, ax = plt.subplots()
plt.scatter(failures_freq.index, failures_freq, c='red', s=20)
plt.scatter(no_failures_freq.index, np.zeros(len(no_failures_freq)), c='blue', s=40)
plt.xlabel('X: Temperature')
plt.ylabel('Number of Failures')
ax.grid()


#get the data in correct format
y, X = dmatrices('Y ~ X', data, return_type = 'dataframe')
#build the model
logit = sm.Logit(y, X)
result = logit.fit()

# summarize the model
print(result.summary(),'\n')

print('Parameters: ', result.params,'\n')

yhat = logit.predict(result.params, exog=None, linear=False)#Predict response variable of a model given exogenous variables.
yhatsum = yhat**5
fig, ax = plt.subplots()
plt.plot(yhat, c='red')
ax.set_ylabel('Probability of failures')
ax.set_xlabel('Temperature')
ax.set_title('\n(Logit Model)')
#ax.legend(loc='upper center', shadow=True, fontsize=8)
ax.grid()
fig.savefig("Logit_Challenger.png")   
plt.show()

prob_fail = [np.exp(result.params[0] + i*result.params[1])/(1+ np.exp(result.params[0] + i*result.params[1])) for i in range(0,120,1)]                      
prob_cast = (np.array(prob_fail)) ** 5 # failures are independent events
fig, ax1 = plt.subplots()
ax1.set_yticks(np.arange(0, 1, step=0.1))
ax1.set_ylim(0,1)
ax1.set_ylabel('Probability')
ax1.set_xlabel('Temperature (F)')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Probability from model prediction (yhat)',color=color)  # we already handled the x-label with ax1
ax1.plot(prob_fail,label='Ring Failure', c='red')
ax1.plot(prob_cast,label='5 Ring Failure', c='black')
ax1.plot(yhat, label='model response variable prediction',color=color)
#ax2.plot(yhatsum, 'o-', label='yhatcum',color=color)
ax1.tick_params(axis='y')
ax2.tick_params(axis='y', labelcolor=color)
ax1.grid()
ax2.grid()
ax1.legend(loc='upper right', shadow=True, fontsize=8)
ax1.set_title('Probability of one ring failure and disaster \n(Logit Model)')
#ax2.legend(loc='upper right', shadow=True, fontsize=8)
fig.savefig("Logit_Challenger_prob.png")


odds_fail = [np.exp(i*result.params[1]) for i in range(0,120,1)]
odds_cat = [np.exp((i*result.params[1])**5) for i in range(0,120,1)]
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_yticks(np.arange(0, 1, step=0.1))
ax1.set_ylim(0,1)
ax1.set_xticks(np.arange(0, 120, step=10))
ax1.set_xlim(0,120)
ax1.set_xlabel('temp(F)')
ax1.set_ylabel('Odds', color=color)
ax1.plot(odds_fail, c='red')
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_yticks(np.arange(0, 1, step=0.1))
ax2.set_ylim(0,1)
ax2.set_ylabel('Disaster odds', color=color)  # we already handled the x-label with ax1
ax2.plot(odds_cat, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax1.set_title('Odds of failure \n(Logit Model)')
#ax.legend(loc='upper center', shadow=True, fontsize=8)
ax1.grid()
fig.savefig("Logit_Challenger_odds.png")



plt.show()
T = 36
pf_T = np.exp(result.params[0] + T*result.params[1])/(1+ np.exp(result.params[0] + T*result.params[1]))
pc_T = ((pf_T)) ** 5
print('Probability of 1-ring failure at ',T,'F:',round(pf_T*100,0),"%",'\n','Probability of disaster at ',T,'F (=5-ring failure):',round(100*pc_T,1),"%")
exp_cte = np.exp(result.params[0])
odds_T = np.exp(1*result.params[1])-1
odds_T1 = np.exp(-1*result.params[1])-1
print('Change in odds of failure with ',1,'F increase in temp:',round(100*odds_T,1),'%')
print('Change in odds of failure with ',1,'F decrease in temp:',round(100*odds_T1,1),'%')
print(exp_cte) 

# Analyze results the model
margeff = result.get_margeff()
print(margeff.summary(),'\n')
mean = np.mean(X,axis=0)
p_hat = 1/(1+np.exp(mean[1]*result.params[1]))
me_own = p_hat*(1-p_hat)*result.params[1]
print('MEM',round(me_own*100,5))
#print('Marginal Effect at the mean (MEM) ewuals',result.margeff[1],'%','\n')
'''
Marginal effects can be an informative means for summarizing how change in a
response is related to change in a covariate. 
For continuous independent variables, the marginal effect measures the instantaneous rate of
change. However, it will depend, in part, on how Xk is scaled.
'''



#confusion matrix

# Precision

# Recall

# F-Score (harmonic Mean)


# Confidence interval

# Normal distribution

#binominal distribution