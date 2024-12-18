import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np


data = pd.read_csv('wine-clustering.csv')

#Scale the data so that K-Means and PCA will work effectively 
scaler = StandardScaler()
scaler.fit(data) #makes the data have a mean of 0 and an std of 1
scaled_data = pd.DataFrame(scaler.transform(data), columns = data.columns)
scaled_data.head()

#We need to find how many dimensions of PCA to use
pca_dim = PCA()
pca_dim.fit(scaled_data)
plt.plot(np.cumsum(pca_dim.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.show()



#Use dimensionality reduction with principle component analysis for preprocessing
pca= PCA(n_components=3)
pca.fit(scaled_data) # learns the PCA from the dataset
pca_data = pd.DataFrame(pca.transform(scaled_data), columns = ['Component 1', 'Component 2', 'Component 3'])
pca_data.head().T  # prints out the features with corresponding data

x = pca_data['Component 1']
y = pca_data['Component 2']
z = pca_data['Component 3']


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_data)
pca_data['Kmeans'] = clusters

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, s=40, c=pca_data['Kmeans'], cmap='viridis', marker = 'o')

#Add labeling to the X and Y axis
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA Scatter Plot with Clusters')

plt.show()

# Get PCA Weightings
weightings = pd.DataFrame(
    pca.components_.T,  #Transpose the rows
    columns=['Component 1', 'Component 2', 'Component 3'],  #Name the components
    index=scaled_data.columns  #Use original attribute names as the index
)

print("PCA Weightings:")
print(weightings)

