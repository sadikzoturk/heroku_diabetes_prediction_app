
import pickle #python nesnelerini kaydetmek ve cagirmak icin kullanilir.
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

#model = pickle.load(open('regression_model.pkl','rb'))
#print(model.predict())
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("data/diabetes.csv", index_col=False)

dataset = df.drop([ 'SkinThickness', 'BloodPressure','Insulin'], axis=1)


X = dataset.drop(['Outcome'], axis=1)

y = dataset['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=123456)
print(X_test)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
print("Accuracy", accuracy_score(y_test, predictions))

pickle.dump(model, open('model.pkl','wb'))

print("Model Kaydedildi")
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6,1,1]]))