import matplotlib
matplotlib.use('pdf')
import os
import shutil
from sklearn.datasets import load_wine
from cluster_drug_discovery.methods import kmeans, dbscan, agglomerative 

data = load_wine()
X = data.data


def test_metrics(X=X):
    output = os.path.join(os.getcwd(), "analisis")
    if os.path.exists(output):
        shutil.rmtree(output)
    cluster = kmeans.KmeansAlg(X, nclust=3)
    cluster.analyze()
    if os.path.exists(output):
        assert True
    else:
        assert False
