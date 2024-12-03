#Figure out which handwritten digits are most differentiated with PCA:
#Imagine you are working on an image recognition service for a postal service. It would be very useful to be able to read in the digits automatically, even if they are 
#handwritten. (Quick note, this is very much how modern postal services work for a long time now and its actually more accurate than a human). The manager of the postal service
#wants to know which handwritten numbers are the hardest to tell apart, so he can focus on getting more labeled examples of that data. You will have a dataset of hand written 
#digits (a very famous data set) and you will perform PCA to get better insight into which numbers are easily separable from the rest.
# Complete the following tasks:
# a. Import the libraries and relevant data set. 
# b. Create a new DataFrame called pixels that consists only of the pixel feature values by dropping the number_label column.(DONE)
# c. Grab a single image row representation by getting the first row of the pixels DataFrame.
# d. Convert the above single row Series into a numpy array.
# e. Reshape this numpy array into an (8,8) array.
# f. Use Matplotlib or Seaborn to display the array as an image representation of the number drawn. Remember your palette or cmap choice would change the colors, but not the
#actual pixel values.
# g. Use Scikit-Learn to scale the pixel feature dataframe.
# h. Perform PCA on the scaled pixel data set with 2 components.
# i. Show how much variance is explained by 2 principal components.
# j. Create a scatterplot of the digits in the 2-dimensional PCA space, color/label based on the original number_label column in the original dataset.

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

loadDigits = load_digits()
df = pd.DataFrame(loadDigits.data, columns=loadDigits.feature_names)

df['number_label'] = loadDigits.target
pixels = df.drop('number_label', axis = 1)

rowSeries = pixels.iloc[0]
rowSeries = rowSeries.to_numpy()
rowSeries = rowSeries.reshape(8,8)

plt.imshow(rowSeries, cmap='gray') 
plt.title(f"Handwritten Digit: {loadDigits.target[0]}")
plt.colorbar()
plt.show()

scaler = StandardScaler()
scaled_pixels = scaler.fit_transform(pixels)

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(scaled_pixels)

print("Explained Variance", pca.explained_variance_ratio_)
print("Sum Explained Variance" , np.sum(pca.explained_variance_ratio_))

pca_df = pd.DataFrame(principalComponents, columns=['PCA1', 'PCA2'])
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=df["number_label"], cmap="viridis")
plt.title('PIXELS PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
