# Importing important Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing Data Preprocessing Libraries
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importing the Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Importing the Dataset
df = pd.read_csv('data.csv')

#----------------------------------------Data Preprocessing and Data Cleaning----------------------------------------

# Dropping Patient ID Columns i.e p_id because it is of no use to our model
df = df.drop('p_id', axis=1)

# By looking at the Dataset, we can observe Several Anomalies present in it like,
# Woman getting Pregnant for more than 10 Times (Rarely Possible, (17 times for a Data Point))
# Blood Pressure Equals to 0 (Not Possible) and etc
df.loc[df['no_times_pregnant'] >= 10, 'no_times_pregnant'] = np.nan
should_not_be_zero = ['glucose_concentration', 'blood_pressure', 'skin_fold_thickness', 'serum_insulin', 'bmi']
df[should_not_be_zero] = df[should_not_be_zero].replace(0, np.nan)

# Storing All Column Names in a List
columns = list(df.columns)

# Dropping Columns with high amount of presence of missing values
for col in columns:
    if (df[col].isnull().sum() / df[col].count()) > 0.1:
        df = df.drop(col, axis=1)

# Creating Instances of Imputer Class for Missing Value Management
imputer_mode = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imputer_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)

# Separating Features and Class
X = df.iloc[:, :-1].values
y = df.iloc[:, 6].values

# Managing Missing Data
X[:, 1:4] = imputer_mean.fit_transform(X[:, 1:4])
X[:, :1] = imputer_mode.fit_transform(X[:, :1])

# Splitting the Dataset into Training Set and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#----------------------------------------Data Preprocessing and Data Cleaning----------------------------------------

#----------------------------------------Model Building and Training----------------------------------------

# Creating Classifier and Accuracy Score Lists
classifiers = ['Decision Tree Classifier', 'K-Nearest Neighbor Classifier', 'Random Forest Classifier', 'Logistic Regression']
scores = list()

# Training using Decision Tree Classifier
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)
# print(score)

# Training using K-Nearest Neighbor Classifier
clf2 = KNeighborsClassifier(n_neighbors = 9)
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)
# print(score)

# Training using Random Forest Classifier
clf3 = RandomForestClassifier(n_estimators = 20)
clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)
# print(score)

# Training using Logistic Regression
clf4 = LogisticRegression()
clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)
# print(score)

#----------------------------------------Model Building and Training----------------------------------------

#----------------------------------------Model Evaluation----------------------------------------

# Evaluating Performance of all the Classifiers
sns.barplot(x=scores, y=classifiers)
plt.xlabel('Accuracy Score')
plt.ylabel('Classifier')
plt.title('Classifier Performance')
plt.show()

# As we can see that Random Forest Classifier has the best Accuracy Score, therefore we'll use it as the Final Model

#----------------------------------------Model Evaluation----------------------------------------

# Checking on Sample Data
ds = [
    ['no_times_pregnant', 'glucose_concentration', 'blood_pressure', 'bmi', 'diabetes pedigree', 'age'],
    [0, 100, 140, 31.1, 0.5, 57]
]

ds = pd.DataFrame(ds[1:], columns=ds[0])

[print('Yes') if clf3.predict(ds)[0] == 0 else print('No')]