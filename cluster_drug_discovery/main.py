import argparse
import numpy as np
from sklearn.datasets import make_blobs
from cluster_drug_discovery.methods import kmeans, dbscan 
import cluster_drug_discovery.visualization.plots as pl
from sklearn import cluster, datasets, mixture



def add_args(parser):
    parser.add_argument('algorithm', type=str, help="cluster algorithm to use")
    parser.add_argument('--nclust', type=int, help="n_cluster", default=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find analogs to a query molecule on your private database')
    add_args(parser)
    args = parser.parse_args()
    n_samples = 1500
    random_state = 170

    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None
    
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    
    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
        cluster_std=[1.0, 2.5, 0.5],random_state=random_state)

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}
    
    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                  'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {})]

    for i_dataset, (dataset, algo_params) in enumerate(datasets[4:5]):
        X, y = dataset
        cluster = dbscan.DbscanAlg(X, epsilon=3)
        cluster.analyze() 
        cluster.nclust = 3
        y_pred = cluster.run()
        pl.plot(X[:, 0], X[:, 1], y_pred, output=str(i_dataset))
