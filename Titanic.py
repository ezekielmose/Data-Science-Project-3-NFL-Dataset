#!/usr/bin/env python
# coding: utf-8

# # Training the model using linear regression in machine learning

# In[40]:


import pandas as pd


# In[4]:


headbrain_dataset=pd.read_csv("D:\Studies\Data Science\Projects\Edureka Projects\\headbrain.csv")
headbrain_dataset


# In[39]:


#collecting x and y

x=headbrain_dataset['Head Size(cm^3)'].values
x1=len(x)
print(x1)
y=headbrain_dataset['Brain Weight(grams)'].values
y1=len(y)
print(y1)


# In[29]:



import numpy as np
#Calcutating the means of y and x
mean_x=np.mean(x)
print("Mean of x: ",+ mean_x)
mean_y=y.mean()
print("Mean of y: ",+ mean_y)
#Total number of values
n= len(x)
print("The total number of values is: ",+ n)

#using the formula to calculate b1 and b0
numb=0
denom=0

for i in range (n):
    numb+=(x[i]-mean_x)*(y[i]-mean_y)
    denom+=(x[i]-mean_x)**2
b1=numb/denom
b0=mean_y-(b1*mean_x)

print(b1,b0)


# In[36]:


import matplotlib.pyplot as plt
#plotting values and regression line
max_x=np.max(x)+100
min_x=np.min(x)-100

#calculating the line values x and y
x=np.linspace(min_x, max_x, 1000)
y=b0+b1*x

#plotting the line
plt.plot(x,y, color='#58b970', label='Regression Line')
plt.scatter(x,y, c='#000000', label='scatter plot')

plt.xlabel('Head size in cm3')
plt.ylabel('Brain weight in grams')
plt.legend()
plt.show()


# In[38]:


ss_t=0
ss_r=0
for i in range (n):
    y_pred=b0+b1*x[i]
    ss_t +=(y[i]-mean_y)**2
    ss_r +=(y[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print(r2)


# # Logistic regression

# In[44]:


import seaborn as sns
import math
titanic=pd.read_csv("D:\Studies\Data Science\Projects\Edureka Projects\Project 2\\Titanic-Dataset.csv")
titanic


# In[46]:


print(str(len(titanic.index)))


# # Analyzing the data

# In[55]:


sns.countplot(x="Survived", hue="Sex", data=titanic )


# In[56]:


titanic["Age"].plot.hist()


# In[57]:


titanic['Fare'].plot.hist()


# In[58]:


titanic.info()


# In[59]:


titanic.head()


# # Data wrangling

# In[60]:


# To view the null values
titanic.isnull()


# In[61]:


# show the number of passengers with n ull values
titanic.isnull().sum()


# In[64]:


sns.heatmap(titanic.isnull(), cmap="viridis")


# In[65]:


sns.boxplot(x="Pclass", y="Age", data=titanic)


# In[67]:


titanic.head(5)


# In[69]:


# droping cabin column
titanic.drop("Cabin", axis=1, inplace=True)
titanic.head(5)


# In[74]:


titanic.dropna(inplace=True)


# In[78]:


sns.heatmap(titanic.isnull())


# In[80]:


# Recheck the null values again
titanic.isnull().sum()


# In[82]:


#convert the string sex string variables into binary figures
sex=pd.get_dummies(titanic["Sex"], drop_first=True)
sex.head(3)


# In[83]:


pcl=pd.get_dummies(titanic["Pclass"], drop_first=True)
pcl.head(3)


# In[84]:


# concartinating all new rows into the dataset
titanic=pd.concat([titanic, sex,pcl], axis=1)
titanic.head(4)


# In[87]:


titanic.drop(['PassengerId', 'Sex', 'Embarked','Name','Ticket'], axis=1, inplace=True)


# In[88]:


titanic.head()


# In[89]:


titanic.drop(['Pclass'], axis=1)


# # Training and testing the model

# In[90]:


x= titanic.drop("Survived", axis=1)
y= titanic ["Survived"]


# In[100]:


# Splitting the data to training and testing data set
import numpy as np
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3, random_state=1)


# In[97]:


from sklearn.linear_model import LogisticRegression


# In[98]:


logmodel=LogisticRegression()


# In[102]:


from sklearn import preprocessing
logmodel.fit(x_train, y_train)


# In[108]:


prediction=logmodel.predict(x_test)


# In[109]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)


# In[ ]:




