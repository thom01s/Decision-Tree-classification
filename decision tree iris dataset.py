from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from seaborn import heatmap
from sklearn import preprocessing


iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)

y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)
dt_classifier.fit(X_train, y_train)
y_predict = dt_classifier.predict(X_test)
dt_classifier.score(X_test, y_test)

le = preprocessing.LabelEncoder()
le.fit_transform(np.array(y).flatten())
labels = list(le.classes_)

matriz = confusion_matrix(y_test, y_predict)
df_cm = pd.DataFrame(matriz, index = [i for i in labels], columns = [i for i in labels])
plt.figure(figsize = (10,7))
plt.title("wine classification")
heatmap(df_cm, annot=True, fmt='g')
plt.xlabel("Previsão")
plt.ylabel("Verdadeiro")
plt.show()

print("*****RF*****")
print(metrics.classification_report(y_test, y_predict))