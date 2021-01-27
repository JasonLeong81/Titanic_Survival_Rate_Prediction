import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# predict whether passengers have survived or not

train = pd.read_csv('titanic_train.csv')
# print(train.head())
# print(train.columns)
# print(train.dtypes)
# print(train.shape)


# print(train.isnull().sum())
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")
# sns.heatmap(train.isnull(),cbar=False,cmap="viridis")
# sns.heatmap(train.isnull(),yticklabels=False,cmap="viridis")
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue='Sex',data=train)
# sns.displot(train['Age'].dropna(),kde=False,color='darkred',bins=40)
# sns.countplot(x='SibSp',data=train)


# sns.boxplot(x='Pclass',y='Age',data=train)
def replace_NA_for_age(cols):
    age, t_class = cols[0],cols[1]
    if pd.isnull(age):
        if t_class == 1:
            # age = 37
            return 37
        elif t_class == 2:
            # age = 29
            return 29
        else:
            # age = 24
            return 24
    else:
        return age
# sns.countplot('Embarked',data=train)
def replace_NA_for_embark(col):
    if pd.isnull(col):
        return 'S'
    else:
        return col
train['Age'] = train[['Age','Pclass']].apply(replace_NA_for_age,axis=1)
train['Embarked'] = train['Embarked'].apply(replace_NA_for_embark) # no need axis
# print(train.isnull().sum())
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")
train.drop('Cabin',axis=1,inplace=True)
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")


# print(pd.get_dummies(train['Embarked'],drop_first=True).head())
# print(pd.get_dummies(train['Sex'],drop_first=True).head())
sex = pd.get_dummies(train['Embarked'],drop_first=True)
embarked = pd.get_dummies(train['Sex'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embarked],axis=1)
# print(train.head())

########## Build a logistic regression model ##############################
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.3,
                                                 random_state=101)
# shuffling the data prevents biases, https://www.youtube.com/watch?v=5SNSmnPXODM
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train) # The point of fitting the model is to find this equation – to find the values of m and b such that y=mx+b describes a line that fits our observed data well.
predictions = logmodel.predict(x_test)
from sklearn.metrics import confusion_matrix
accuracy = confusion_matrix(y_test,predictions)
print('Confusion Matrix')
for i in accuracy:
    print(i)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print('Accuracy Score',accuracy)


# plt.show()