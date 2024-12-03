# Download any clustering dataset from internet and after finding how many optimum number of
# clusters should be formed using elbow curve, apply k-means on it.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv("banknote.csv")

imputer = SimpleImputer(strategy="mean")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

scaler = StandardScaler()
df = scaler.fit_transform(df)

inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, init = "k-means++")
    kmeans.fit(df)
    inertias.append(kmeans.inertia_)

frame = pd.DataFrame({'k':range(1,11), 'inertias':inertias})
plt.plot(frame['k'], frame['inertias'], marker = "o")
plt.title("ELBOW METHOD FOR OPTIMAL K")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42)
kmeans.fit(df)

print("Cluster centers:", kmeans.cluster_centers_)
print("Cluster sizes:", pd.Series(kmeans.labels_).value_counts())  

