#!/usr/bin/env python
# coding: utf-8

# In[23]:


#importing libraries
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC


# In[13]:


#reading csv file
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pd.read_csv(url, names=names)


# In[14]:


#dimension of dataset
print(dataset.shape)


# In[15]:


#printing first 10 data from dataset
print(dataset.head(10))


# In[16]:


#statistical summary
print(dataset.describe())


# In[17]:


#class distribution
print(dataset.groupby('class').size())


# In[18]:


#whisker-plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# In[19]:


#histogram of variables
dataset.hist()
pyplot.show()


# In[20]:


#multivariate plot
scatter_matrix(dataset)
pyplot.show()


# In[29]:


#splitting dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)


# In[30]:


#building different models
#1.Logistic Regression
#2.Linear Discriment Analysis
#3.K-nearest Neigbors
#4.Classification and Regression Tree
#5.Gaussian Naive Bayes
#6.Support Vector Machine

models = []
models.append(('LR',LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))


# In[31]:


#evaluate the created models
results = []
names = []
for name,model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' %(name, cv_results.mean(), cv_results.std()))


# In[32]:


#compare our models
pyplot.boxplot(results, labels=names)
pyplot.title('Alogrithms comparison')
pyplot.show()


# In[35]:


#make prediction on svm
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)


# In[36]:


#evaluate prediction
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


# In[ ]:




