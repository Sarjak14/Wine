import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

mydata = pd.read_csv("winequality-red.csv")
outputdata = mydata["quality"]
mydata.drop(columns=["quality"], inplace=True)

scaler = StandardScaler()
mydata = pd.DataFrame(scaler.fit_transform(mydata), columns=mydata.columns)

X_train, X_test, Y_train, Y_test = train_test_split(mydata, outputdata, test_size=0.5, random_state=42)

param_dist = {
    "n_estimators": [10, 50],
    "max_depth": [None, 10],
    "min_samples_split": [2],
}

mymodel = RandomForestClassifier()
grid_search = GridSearchCV(estimator=mymodel, param_grid=param_dist, cv=3, scoring="accuracy")
grid_search.fit(X_train, Y_train)

mymodel = grid_search.best_estimator_
mymodel.fit(X_train, Y_train)

predicted = mymodel.predict(X_test)

print("Accuracy: ", accuracy_score(Y_test, predicted))
print("Cross-Validation Score: ", grid_search.best_score_)

confusion = confusion_matrix(Y_test, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=mymodel.classes_)
disp.plot()
plt.show()

importance = mymodel.feature_importances_
for i, j in enumerate(importance):
    print(f"{mydata.columns[i]}: {j}")

plt.figure(figsize=(10, 5))
plt.boxplot(predicted)
plt.title("Predicted Values")
plt.show()