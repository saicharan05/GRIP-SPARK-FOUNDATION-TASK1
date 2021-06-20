mport pandas as pd
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn import svm 
from sklearn import tree
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from random import sample
import numpy as np
import math

data=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
data

figr,axr =  plt.subplots(1,2)
axr[0].boxplot(data['Hours'])
axr[1].boxplot(data['Scores'])
plt.show()

X=data['Hours'].values.reshape(-1,1)
X

regr1=LinearRegression()
regr2 = BayesianRidge()
regr3 = svm.SVR()
regr4 = KNeighborsRegressor()
regr5 = tree.DecisionTreeRegressor()
regr6= VotingRegressor(estimators=[['LR',regr1],['BR',regr2],['KN',regr4],['DT',regr5]])
regr6

X_train,X_test,y_train,y_test=train_test_split(X,data['Scores'],test_size=0.5,random_state=1)

regr1.fit(X_train,y_train)
regr2.fit(X_train,y_train)
regr3.fit(X_train,y_train)
regr4.fit(X_train,y_train)
regr5.fit(X_train,y_train)
regr6.fit(X_train,y_train)

Y_pred1 = regr1.predict(X)
Y_pred2 = regr2.predict(X)
Y_pred3 = regr3.predict(X)
Y_pred4 = regr4.predict(X)
Y_pred5 = regr5.predict(X)
Y_pred6 = regr6.predict(X)

figure, axis = plt.subplots(1,2)

axis[0].plot(data['Hours'],data['Scores'],'o')
axis[0].plot(X,Y_pred1)
axis[0].set_title('linear regression')

axis[1].plot(data['Hours'],data['Scores'],'o')
axis[1].plot(X,Y_pred2)
axis[1].set_title('BayesianRidge')
plt.show()

y_pred_test1=regr1.predict(X_test)
y_pred_test2=regr2.predict(X_test)
y_pred_test3=regr3.predict(X_test)
y_pred_test4=regr4.predict(X_test)
y_pred_test5=regr5.predict(X_test)
y_pred_test6=regr6.predict(X_test)

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

from sklearn import metrics
print('{:<25} : {}{}'.format('MAPE for linear',round(MAPE(y_test, y_pred_test1),2),'%'))  
print('{:<25} : {}{}'.format('MAPE for BayesianRidge',round(MAPE(y_test, y_pred_test2),2),'%'))  
print('{:<25} : {}{}'.format('MAPE for SVM',round(MAPE(y_test, y_pred_test3),2),'%'))  
print('{:<25} : {}{}'.format('MAPE for KNeign',round(MAPE(y_test, y_pred_test4),2),'%'))  
print('{:<25} : {}{}'.format('MAPE for Decisiontree',round(MAPE(y_test, y_pred_test5),2),'%'))  
print('{:<25} : {}{}'.format('MAPE for Votingensemble',round(MAPE(y_test, y_pred_test6),2),'%'))  

print('{:<45} : {}'.format('prediction for 9.25hours by linear',regr1.predict([[9.25]])[0]))
print('{:<45} : {}'.format('prediction for 9.25hours by BayesianRidge',regr2.predict([[9.25]])[0]))
print('{:<45} : {}'.format('prediction for 9.25hours by SVM',regr3.predict([[9.25]])[0]))
print('{:<45} : {}'.format('prediction for 9.25hours by KNeign',regr4.predict([[9.25]])[0]))
print('{:<45} : {}'.format('prediction for 9.25hours by Decisiontree',regr5.predict([[9.25]])[0]))
print('{:<45} : {}'.format('prediction for 9.25hours by Votingensemble',regr6.predict([[9.25]])[0]))

