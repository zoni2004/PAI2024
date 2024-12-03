#Implement K-Nearest Neighbor classifier on the heart disease dataset and analyze the performance using accuracy, while varying the number of neighbors (e.g. 1 – 250). Also
#print the neighbor(s) having the highest accuracy and those with the lowest accuracy.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

url ="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"    
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

heartData = pd.read_csv(url, names = columns, na_values = ['?'])

imputer = SimpleImputer(strategy='most_frequent')
heartData = pd.DataFrame(imputer.fit_transform(heartData), columns=heartData.columns)

data = heartData.drop('target', axis = 1)
target = heartData['target']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, random_state = 42) 

accuracies = []

for k in range(1, len(X_train) + 1):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    
accuracies_series = pd.Series(accuracies, index=range(1, len(accuracies) + 1))
plt.figure(figsize=(10, 6))
plt.plot(accuracies_series.index, accuracies_series.values, marker='o', color='b')
plt.title("Accuracy vs k in KNN for Heart Disease Dataset")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

best_k = accuracies_series.idxmax()
best_accuracies = accuracies_series.max()

worst_k = accuracies_series.idxmin()
worst_accuracies = accuracies_series.min()

print(f"Best k: {best_k} with Accuracy: {best_accuracies:.4f}")
print(f"Worst k: {worst_k} with Accuracy: {worst_accuracies:.4f}")


    
