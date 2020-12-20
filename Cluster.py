from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_credit2(1).csv')



X = df.loc[:, ['credit_amount','savings','age']].values

rg = np.random.default_rng(12345)
r = int(rg.random()*10)
kmeans = KMeans(n_clusters=3, random_state=r).fit(X)
y_kmeans = kmeans.predict(X)
print(X[:,0])
print(y_kmeans)
fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
ax.scatter(X[:,1],X[:,2],X[:,0],c=y_kmeans,cmap='viridis',alpha=0.5)
centers = kmeans.cluster_centers_
ax.scatter(centers[:,1],centers[:,2],centers[:,0],c='black',alpha=1)


ax.set_xlabel('savings')
ax.set_ylabel('age')
ax.set_zlabel('credit_amount')

plt.show()


print(kmeans.labels_)

#New_data_fram = pd.DataFrame({"kmeans_labels":kmeans.labels_})
df.insert(25, "kmeans_labels", kmeans.labels_, True)
#print(New_data_fram)
V = df.loc[:, ['credit_amount','savings','age']].copy()
V.insert(3, "kmeans_labels", kmeans.labels_, True)





df.to_csv('cleaned.csv')




#print(kmeans.predict([[10, 10], [5, 0]]))
