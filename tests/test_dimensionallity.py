from sklearn.datasets import load_wine
from cluster_drug_discovery.methods import kmeans, dbscan, agglomerative 

data = load_wine()
X = data.data


def test_pca(X=X):
    cluster = kmeans.KmeansAlg(X, nclust=13)
    pca = cluster.compute_pca()
    if pca is not None:
        assert True

def test_umap(X=X):
    cluster = kmeans.KmeansAlg(X, nclust=13)
    umap = cluster.compute_umap()
    if umap is not None:
        assert True
