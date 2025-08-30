# k-means on a mall customer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
df = pd.read_csv('/content/Mall_Customers 2.csv')
#select feature for clustering
x=df[['Annual Income (k$)','Spending Score (1-100)']].copy()
#scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled)
# fit k-means (k=5 is common for this dataset)
k=5
kmeans =KMeans(n_clusters=k,n_init=10,random_state=42)
labels = kmeans.fit_predict(x_scaled)
#evaluate clustering quality
sil = silhouette_score(x_scaled,labels)
print(f"Silhouette score: {sil:.3f}")
# Add cluster labels back to the dataFrame
df['Cluster'] = labels
#visualize cluster
plt.figure(figsize=(7,5))
plt.scatter(x['Annual Income (k$)'],x['Spending Score (1-100)'],c= labels,)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],marker = "x",s = 200)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer: K=Means Clustering ( k=5)")
plt.show()
