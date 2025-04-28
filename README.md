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
## Program // ## Output:
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
![image](https://github.com/user-attachments/assets/1cc9e922-a4d2-48a9-89b9-eda08a08f3de)

```
df.head()
```
![image](https://github.com/user-attachments/assets/fae079cf-9ba5-4721-b01f-15da83b0f5e2)

```
data1 = df.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
```
![image](https://github.com/user-attachments/assets/1bbb3028-fb3d-44e2-9f6d-af91d550d30d)

```
data1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/e0f9c5f7-d6e3-46cf-92bd-c8b6ba6477af)

```
data1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/76e92ace-8bdb-4c5f-b5bd-2f1ef5592ad2)

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
![image](https://github.com/user-attachments/assets/4912bd17-f5b6-47b2-b884-0de40b62e1c3)

```
x = data1.iloc[:,:-1]
x
```
![image](https://github.com/user-attachments/assets/0bc4c153-6cb9-4d93-8dc1-900c024c249d)

```
y = data1["status"]
y
```
![image](https://github.com/user-attachments/assets/891451e0-cd8d-471a-913f-a78e8d88c13e)
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
![image](https://github.com/user-attachments/assets/d9b5988e-ce66-4551-9a6f-ed5209d78bac)

```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/e0908e3d-0934-40fd-b5e3-a70366e0e244)

```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/484fe291-30fc-4583-84ff-a20c0ac3029e)

```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
![image](https://github.com/user-attachments/assets/4ae5d1cb-4171-4b87-8d5f-76682cd4c64f)

```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
![image](https://github.com/user-attachments/assets/905f8ee9-09ab-458a-a8b2-d5a85f8c2b3e)

```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
print("Name:Vedagiri Indu Sree")
print("Reg No:212223230236")
```
![image](https://github.com/user-attachments/assets/d7305d96-85fa-4a0e-b11a-a40237f68365)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
