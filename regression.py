import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('./dataset/diabetes.csv', sep = ',')
X = data.iloc[:, :8].values
Y = data['Outcome'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
accuracy_list = []

model =  LogisticRegression()
model.fit(X_train, Y_train)
prediction=model.predict(X_test)
accuracy=accuracy_score(Y_test, prediction)
accuracy_list.append(accuracy)
print("Accuracy: ", accuracy_list[-1])