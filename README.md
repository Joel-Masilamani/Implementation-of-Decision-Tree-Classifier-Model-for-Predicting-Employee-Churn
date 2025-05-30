# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## Name: Joel Masilamani 
## Reg no: 212224220043

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries and load the dataset.
2.Handle null values and encode categorical columns.
3.Split data into training and testing sets.
4.Train a DecisionTreeClassifier using entropy.
5.Predict and evaluate the model using accuracy and metrics. 

## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
*/

from google.colab import drive
drive.mount('/content/drive')
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io

data = pd.read_csv(io.BytesIO(uploaded['Employee.csv']))
data.head()


data.head()
data.info()
print("Null values:\n", data.isnull().sum())
print("Class distribution:\n", data["left"].value_counts())
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Display the updated data
data.head()
# Select input features
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
print(x.head())

# Define target variable
y = data["left"]

from sklearn.model_selection import train_test_split

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
from sklearn.tree import DecisionTreeClassifier

# Create and train the model
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
from sklearn.tree import DecisionTreeClassifier

# Create and train the model
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
from sklearn import metrics

# Predict on test set
y_pred = dt.predict(x_test)

# Evaluate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Predict on new employee data
sample_prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print("Sample Prediction:", sample_prediction)

*/
```

## Output:
![image](https://github.com/user-attachments/assets/313bcf91-362c-453a-90e2-f4d0750e9aa8)

![image](https://github.com/user-attachments/assets/95e51b29-6d17-4b66-bea6-443bf2ef9f48)

![image](https://github.com/user-attachments/assets/c21008c0-3783-45c5-813d-42d8fd1838cd)

![image](https://github.com/user-attachments/assets/0a584d78-f53a-42ae-9e64-7a88669ff368)

![image](https://github.com/user-attachments/assets/84c28262-e57c-42b3-95c7-b0ad24faddba)

![image](https://github.com/user-attachments/assets/1be51f3c-fbca-4d5d-9255-8bfc60876946)

![image](https://github.com/user-attachments/assets/b0362716-cf46-447d-969e-56d55bb0e5e3)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
