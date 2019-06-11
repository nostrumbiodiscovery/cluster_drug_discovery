# Retrieve Dataset


# Analyze dataset
from sklearn import cluster, datasets, mixture
import cluster_drug_discovery.methods.kmeans as ks

X, y = datasets.make_blobs(n_samples=1500, random_state=170)
analysis_cluster = ks.KmeansAlg(X, nclust=3)
analysis_cluster.analyze()

# Cluster
from cluster_drug_discovery.methods import kmeans
import matplotlib.pyplot as plt
cluster = kmeans.KmeansAlg(X, nclust=3)
y_pred = cluster.run()
plt.plot(X[:, 0], X[:, 1], y_pred)


