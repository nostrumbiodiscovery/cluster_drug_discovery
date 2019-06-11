from cluster_drug_discovery.methods import kmeans, dbscan, agglomerative 
from sklearn import cluster, datasets, mixture


X, y = datasets.make_blobs(n_samples=1500, random_state=170)


def test_kmeans(X=X):
    cluster = kmeans.KmeansAlg(X, nclust=3)
    y_pred = cluster.run()

def test_agglomerative(X=X):
    cluster = agglomerative.AgglomerativeAlg(X, nclust=3)
    y_pred = cluster.run()

def test_dbscan(X=X):
    cluster = dbscan.DbscanAlg(X, epsilon=0.5)
    y_pred = cluster.run()
