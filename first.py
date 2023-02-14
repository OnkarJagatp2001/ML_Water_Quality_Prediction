#!/usr/bin/env python
# coding: utf-8

# In[158]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[159]:


data = pd.read_csv(r'C:\Users\Onkar\Downloads\archive\water_potability.csv')


# In[160]:


data.head()


# In[161]:


data.shape  #rows,columns


# # Data Cleaning

# In[162]:


data.info()


# In[163]:


data.isnull().sum()   #ph, sulphate,trihalomethane have non zero value 


# In[164]:


data = data.fillna(data.mean(),)


# In[165]:


data


# In[166]:


data.isnull().sum()


# #Exploratory data analysis

# In[167]:


data.describe()


# #checking if we need to do dimensonlity Reduction

# In[168]:


sns.heatmap(data.corr(),annot = True,cmap='terrain')
fig = plt.gcf()
fig.set_size_inches(8,6)
plt.show()


# In[ ]:





# # lets check the outlire using box plot

# In[169]:


data.boxplot(figsize=(15,6))
plt.show()


# In[170]:


data['Solids'].describe()


# In[171]:


data['Solids']


# #not removing the outliers coz they may be importantto decide the quality of water

# MORE EDA

# In[ ]:





# In[172]:


data['Potability'].value_counts()


# In[173]:


sns.countplot(data['Potability'])
plt.show()


# In[174]:


data.hist(figsize=(14,9))
plt.show()


# In[175]:


sns.barplot(x=data['ph'],y=data['Hardness'],hue=data['Potability'])
plt.show()


# In[176]:


sns.scatterplot(x=data['ph'],y=data['Potability'])
plt.show()


# #Partitioning

# In[177]:


X = data.drop('Potability',axis=1) #Inpute data


# In[178]:


Y = data['Potability']  #Target Variable


# In[179]:


from sklearn.model_selection import train_test_split


# In[180]:


X_train , X_test , Y_train, Y_test =  train_test_split(X,Y,test_size=0.2,shuffle= True,random_state= None)


# In[181]:


X_train


# In[182]:


Y_train


# In[ ]:





# In[ ]:





# #Normalization

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # step2 MOdel training and model optimization

# In[ ]:





# Model Training

# #Decision Tree

# In[183]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[184]:


dt.fit(X_train,Y_train)


# In[185]:


Y_test


# In[ ]:





# In[ ]:





# In[186]:


Y_prediction = dt.predict(X_test)


# In[187]:


from sklearn.metrics import accuracy_score , confusion_matrix


# In[188]:


accuracy_score(Y_prediction,Y_test)


# In[189]:


confusion_matrix(Y_prediction,Y_test)


# In[190]:


Y_test.shape


# In[ ]:





# # Try more ML Model / Hyper parameter tuning

# #model optimization 

# In[191]:


dt.get_params().keys()


# In[192]:


#example of grid searching key hyperparametres for logistic regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# define models and parameters
model = DecisionTreeClassifier()
criterion = ["gini", "entropy"]
splitter = ["best", "random"]
min_samples_split = [2,4,6,8,10]

# define grid search
grid = dict(splitter=splitter, criterion=criterion, min_samples_split=min_samples_split)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search_dt = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, 
                           scoring='accuracy',error_score=0)
grid_search_dt.fit(X_train, Y_train)

# summarize results
print(f"Best: {grid_search_dt.best_score_:.3f} using {grid_search_dt.best_params_}")
means = grid_search_dt.cv_results_['mean_test_score']
stds = grid_search_dt.cv_results_['std_test_score']
params = grid_search_dt.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")
    
print("Training Score:",grid_search_dt.score(X_train, Y_train)*100)
print("Testing Score:", grid_search_dt.score(X_test, Y_test)*100)


# In[193]:


from sklearn.metrics import  make_scorer
from sklearn.model_selection import cross_val_score

def classification_report_with_accuracy_score(Y_test, y_pred2):
    print (classification_report(Y_test, y_pred2)) # print classification report
    return accuracy_score(Y_test, y_pred2) # return accuracy score

    
nested_score = cross_val_score(grid_search_dt, X=X_train, y=Y_train, cv=cv, 
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)


# In[194]:


dt_y_predicted = grid_search_dt.predict(X_test)
dt_y_predicted


# In[195]:


grid_search_dt.best_params_


# In[196]:


dt_grid_score=accuracy_score(Y_test, dt_y_predicted)
dt_grid_score


# In[197]:


confusion_matrix(Y_test, dt_y_predicted)


# In[ ]:





# In[ ]:





# # KNN HPT

# In[198]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# define models and parameters
model = KNeighborsClassifier()
n_neighbors = range(1, 31)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
grid_search_knn = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, 
                           scoring='accuracy',error_score=0)
grid_search_knn.fit(X_train, Y_train)

# summarize results
print(f"Best: {grid_search_knn.best_score_:.3f} using {grid_search_knn.best_params_}")
means = grid_search_knn.cv_results_['mean_test_score']
stds = grid_search_knn.cv_results_['std_test_score']
params = grid_search_knn.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")


# In[199]:


from sklearn.metrics import  make_scorer
from sklearn.model_selection import cross_val_score

def classification_report_with_accuracy_score(Y_test, y_pred2):
    print (classification_report(Y_test, y_pred2)) # print classification report
    return accuracy_score(Y_test, y_pred2) # return accuracy score

    
nested_score = cross_val_score(grid_search_knn, X=X_train, y=Y_train, cv=cv, 
               scoring=make_scorer(classification_report_with_accuracy_score))
print (nested_score)


# In[200]:


knn_y_predicted = grid_search_knn.predict(X_test)


# In[201]:


knn_y_predicted


# In[202]:


knn_grid_score=accuracy_score(Y_test, knn_y_predicted)


# In[203]:


knn_grid_score


# In[204]:


grid_search_knn.best_params_


# In[205]:


confusion_matrix(Y_test, knn_y_predicted)


# In[ ]:





# # Prediction on only one set of data

# In[209]:


X_KNN=knn_grid_score.predict([[5.735724, 158.318741,25363.016594,7.728601,377.543291,568.304671,13.626624,75.952337,4.732954]])


# In[ ]:


X_KNN


# In[ ]:




