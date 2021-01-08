  
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


Dataset=pd.read_csv(r'C:Datasets/Country-data.csv')
df=pd.get_dummies(Dataset)

X=Dataset.iloc[:,[3,5]].values

#plot
plt.scatter(X[:,0],X[:,1],s=15)
plt.title("Country wise data")
plt.xlabel("Total Expenditure on Health")
plt.ylabel("Total Income")
plt.show()

#scaling
sc_x = StandardScaler()
X_scaled = sc_x.fit_transform(X)

outlier_detection = DBSCAN(eps = .2,metric='euclidean',min_samples = 5, n_jobs = -1)
pred = outlier_detection.fit(X_scaled)

anomaly_index = np.where(pred == -1)
anomaly_values = X_scaled[anomaly_index]

# Finalplot

plt.scatter(X_scaled[:,0],X_scaled[:,1],s=20)
plt.scatter(anomaly_values[:,0], anomaly_values[:,1], color='y',s=20)
plt.title("Country wise data")
plt.xlabel("Total Expenditure on Health")
plt.ylabel("Total Income")
plt.show()


# UNIVARIATE

print("Skewness: %f" % df['health'].skew())
print("Kurtosis: %f" % df['health'].kurt())