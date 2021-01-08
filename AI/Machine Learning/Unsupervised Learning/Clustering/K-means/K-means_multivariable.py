import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

Dataset = pd.read_csv(r'Datasets/WineClass.csv')

X=Dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10]].values

#FeatureScaling
sc_x = StandardScaler()
data_scaled = sc_x.fit_transform(X)

#ElbowMethod
SSE = []
for cluster in range(1,20):
    kmeans = KMeans( n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.title("The Elbow Method")
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#According to the plot, K=3

kmeans = KMeans(init='k-means++', n_clusters=4)
kmeans.fit(data_scaled)