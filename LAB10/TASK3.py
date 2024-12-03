#Download the Dermatology data set from the UCI depositary, apply KNN and analyze the performance. 
#Display the confusion matrix and discuss it. Use 10-Fold Cross Validation and random 
# train/test split (70%, 30%).

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data'
columns = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner', 'polygonal_papules', 
           'follicular_papules', 'oral_mucosal_involvement', 'knee_elbow_involvement', 'scalp_involvement', 
           'family_history', 'melanin_incontinence', 'eosinophils_infiltrate', 'PNL_infiltrate', 'fibrosis', 
           'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis', 'clubbing', 'elongation_rete', 
           'thinning_suprapapillary', 'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis', 
           'disappearance_granular_layer', 'vacuolisation_basal_layer', 'spongiosis', 'saw_tooth', 
           'follicular_horn_plug', 'perifollicular_parakeratosis', 'inflammatory_monoluclear_infiltrate', 
           'band_like_infiltrate', 'age', 'class']
           
           
Dermatology = pd.read_csv(url, names = columns, na_values = ["?"])

imputer = SimpleImputer(strategy = 'most_frequent')
Dermatology_imputed = pd.DataFrame(imputer.fit_transform(Dermatology), columns = columns)

label_encoder = LabelEncoder()
Dermatology_imputed['class'] = label_encoder.fit_transform(Dermatology_imputed['class'])

data = Dermatology_imputed.drop('class', axis = 1)
target = Dermatology_imputed['class']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, random_state = 42)

knn = KNeighborsClassifier(n_neighbors = 5)

cv_scores = cross_val_score(knn, data, target, cv = 10)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV score: {cv_scores.mean()}')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

cm_display.plot()
plt.title("Confusion Matrix for KNN")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

