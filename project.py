# importing libraries
import os as os #not in use
import numpy as np #not in use 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer

# importing data
data = pd.read_csv('train_data.csv')
data.drop("Loan_ID",axis=1, inplace=True)

# finding null values 
print(data.isnull().sum())

#imputing missing values for categorical values 
for i in [data]:
    i["Gender"] = i["Gender"].fillna(data.Gender.dropna().mode()[0])
    i["Married"] = i["Married"].fillna(data.Married.dropna().mode()[0])
    i["Dependents"] = i["Dependents"].fillna(data.Dependents.dropna().mode()[0])
    i["Self_Employed"] = i["Self_Employed"].fillna(data.Self_Employed.dropna().mode()[0])
    i["Credit_History"] = i["Credit_History"].fillna(data.Credit_History.dropna().mode()[0])
#imputing missing values for numerical values 
data1 = data.loc[:,['LoanAmount','Loan_Amount_Term']]

imp = IterativeImputer(RandomForestRegressor(), max_iter=10,random_state=0)
data1 = pd.DataFrame(imp.fit_transform(data1), columns=data1.columns)

#encoding categorical data
for i in [data]:
    i["Gender"] =i["Gender"].map({"Male":0,"Female":1}).astype(int)
    i["Married"] = i["Married"].map({"No":0,"Yes":1}).astype(int)
    i["Education"] = i["Education"].map({"Not Graduate":0,"Graduate":1}).astype(int)
    i["Self_Employed"]=i["Self_Employed"].map({"No":0,"Yes":1}).astype(int)
    i["Credit_History"] = i["Credit_History"].astype(int)

#encoding numerical data 
for i in [data]:
    i["Property_Area"] = i["Property_Area"].map({"Urban":0,"Rural":1,"Semiurban":2}).astype(int)
    i["Dependents"] = i["Dependents"].map({"0":0,"1":1,"2":2,"3+":3})

data['LoanAmount']= data1['LoanAmount'].copy()
data['Loan_Amount_Term']=data1['Loan_Amount_Term'].copy()
    
data["Loan_Status"] = data["Loan_Status"].map({'N':0,"Y":1}).astype(int)

#correlation matrix
plt.figure(figsize = (10,10))
correlation_matrix =data.corr()
sns.heatmap(correlation_matrix,annot=True)
plt.show()

#separating features and target values
x= data.drop("Loan_Status",axis=1)
y= data["Loan_Status"]
#splitting training and test set 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#training model 
classifier = LogisticRegression()
from sklearn.model_selection import cross_val_score
cross_val_score(classifier, x_train,y_train,scoring= make_scorer(accuracy_score),cv=3)

#predicting values for test set 
y_predict= classifier.fit(x_train,y_train).predict(x_test)
accuracy_score(y_predict,y_test)

#performance matrix (confusion matrix)
cm= confusion_matrix(y_test, y_predict)
print(cm)

df_confusion = pd.DataFrame(cm, index=['Yes','No'],columns=['Yes','NO'])

df_confusion


sns.heatmap(df_confusion, annot=True)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()