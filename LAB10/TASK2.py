#Run the above provided KNN Algorithm for different random seed values from 1 to 10. Print all accuracies and then print the highest and the lowest.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"    
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

HeartDisease = pd.read_csv(url, names = columns, na_values=['?'])

imputer = SimpleImputer(strategy='most_frequent')
HeartDisease = pd.DataFrame(imputer.fit_transform(HeartDisease), columns=HeartDisease.columns)

data = HeartDisease.drop('target', axis = 1)
target = HeartDisease['target']

accuracies = []

for random in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, random_state = random)
    
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    accuracies.append(accuracy)
    
accuracies_series = pd.Series(accuracies, index=range(1, 11))
plt.plot(accuracies_series.index, accuracies_series.values)
plt.title("RANDOM SEED VALUES VS ACCURACY for Heart Disease Dataset")
plt.xlabel("Random seed")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

best_r = accuracies_series.idxmax()
best_accuracy = accuracies_series.max()

worst_r = accuracies_series.idxmin()
worst_accuracy = accuracies_series.min()

print(f"Best k: {best_r} with Accuracy: {best_accuracy:.4f}")
print(f"Worst k: {worst_r} with Accuracy: {worst_accuracy:.4f}")
