import argparse
import sys
import os
import glob
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from cluster_drug_discovery.methods import kmeans, dbscan, agglomerative 
from cluster_drug_discovery.input_preprocess import coords_extract as ce
import cluster_drug_discovery.visualization.plots as pl
from sklearn import cluster, datasets, mixture
from AdaptivePELE.utilities import utilities
from AdaptivePELE.analysis import splitTrajectory, simulationToCsv 



def add_args(parser):
    parser.add_argument('algorithm', type=str, help="cluster algorithm to use")
    parser.add_argument('--nclust', type=int, help="n_cluster", default=5)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clusterization algorithm with analysis techniques')
    parser = add_args(parser)
    args = parser.parse_args()

    #Initialize variables
    pele_sim = "/data/JJ/IL17A/63O_inducefit/output/"
    pele_path = os.path.join(pele_sim, "*/*.pdb")
    residue = "63O"
    limit_structs = 1
    trajectory_basename = "run_trajectory_"
    top =  "/data/JJ/IL17A/63O_inducefit/IL17A_4HR9_complex_processed.pdb"

    pdb_files = glob.glob(pele_path)

    #Extract ligand coordinates
    feat_ext = ce.CoordinatesExtractor(pdb_files, [residue, ], 20)
    if os.path.exists("extracted_feature.txt"):
        X = np.loadtxt("extracted_feature.txt") 
    else:
        X, samples = feat_ext.retrieve_coords()
        np.savetxt("extracted_feature.txt", X)
    
    #Clusterize
    cluster = agglomerative.AgglomerativeAlg(X, nclust=6) 
    y_pred = cluster.run()

    
    silhouette_values = silhouette_samples(X, y_pred)
    idx = np.argsort(silhouette_values)[::-1]

    #Check that there is no previously generated file
    try:
        samples[idx]
    except NameError:
        raise NameError("Remove previously generated extracted_feature.txt file and run again") 

    #Extract cluster
    for i in range(0, cluster.nclust):
        output_structs = 0
        for clust, sample, silh in zip(y_pred[idx], samples[idx], silhouette_values[idx]):
            print(sample.epoch, clust, sample.traj)
            if output_structs < limit_structs and clust == i:
                if pele_path[-3:] == "pdb":
                    topology = None
                    filename = "path{}.{}.{}.cluster{}.pdb".format(sample.epoch, trajectory_basename, sample.traj, i)
                    trajectory = os.path.join(pele_sim, "{}/{}{}.pdb".format(sample.epoch, trajectory_basename, sample.traj))
                    snapshots = utilities.getSnapshots(trajectory, topology=topology, use_pdb=False)
                    with open(filename, "w") as fw:
                        fw.write(snapshots[sample.model-1])
                elif pele_path[-3:] == "xtc":
                    topology = top
                    filename = "path{}.{}.{}.cluster{}.pdb".format(sample.epoch, trajectory_basename, sample.traj, i)
                    trajectory = os.path.join(pele_sim, "{}/{}{}.xtc".format(sample.epoch, trajectory_basename, sample.traj))
                    splitTrajectory.main("", [trajectory, ], topology, [sample.model,],template=filename, use_pdb=False)
                output_structs += 1

    """
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
        #cluster = dbscan.DbscanAlg(X, epsilon=3)
        #cluster = kmeans.KmeansAlg(X, nclust=3)
        cluster = agglomerative.AgglomerativeAlg(X, nclust=3) 
        cluster.analyze() 
        cluster.nclust = 3
        y_pred = cluster.run()
        pl.plot(X[:, 0], X[:, 1], y_pred, output=str(i_dataset))
        """
