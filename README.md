# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Dataset

2.Create a Copy of the Original Data

3.Drop Irrelevant Columns (sl_no, salary)

4.Check for Missing Values

5.Check for Duplicate Rows

6.Encode Categorical Features using Label Encoding

7.Split Data into Features (X) and Target (y)

8.Split Data into Training and Testing Sets

9.Initialize and Train Logistic Regression Model

10.Make Predictions on Test Set

11.Evaluate Model using Accuracy Score

12.Generate and Display Confusion Matrix

13.Generate and Display Classification Report

14.Make Prediction on a New Sample Input
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Vedagiri Indusree
RegisterNumber:212223230236  
*/
```
```
import pandas as pd
import numpy as np 
df = pd.read_csv('Placement_Data.csv')
df
```
```
df.head()
```
```
data1 = df.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
```
```
data1.isnull().sum()
```
```
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_p"] = le.fit_transform(data1["degree_p"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["etest_p"] = le.fit_transform(data1["etest_p"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
```
```
x = data1.iloc[:,:-1]
x
```
```
y = data1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)
```
```
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
print("Name:Vedagiri Indu Sree")
print("Reg No:212223230236")
```
## Output:
## data.head()

![image](https://github.com/user-attachments/assets/ed415b6c-4958-4c28-9ea1-86014dc55c74)

## data1.head()

![image](https://github.com/user-attachments/assets/09a96f33-266c-4bc1-a697-4c5fdfba4755)

## isnull()

![image](https://github.com/user-attachments/assets/2c810a64-10fe-4d30-9c90-406a9120b9c4)

## duplicated()

![image](https://github.com/user-attachments/assets/6e94c559-1457-443d-b4c6-edf67bb4223f)

## data1

![image](https://github.com/user-attachments/assets/6f510cec-1d2f-4676-a91f-032bfa01b7d9)

## X

![image](https://github.com/user-attachments/assets/ce7b7ab8-0420-4f51-af52-f5f0ea097b74)

## y

![image](https://github.com/user-attachments/assets/35c2b47c-6972-4dad-98f5-f4528d8f24e3)

## y_pred

![image](https://github.com/user-attachments/assets/117bc417-c9de-4102-914e-15754c92e48b)

## confusion matrix

![image](https://github.com/user-attachments/assets/917b52ca-76b5-4d68-a4ba-1d6d85d5bb49)

## classification report

![image](https://github.com/user-attachments/assets/bf45a2d5-c321-4b5c-a0a8-e76940b06ba7)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
