import hdbscan
import cluster_drug_discovery.methods.clusterclass as cls


class HdbscanAlg(cls.Cluster):
    """
     The DBSCAN algorithm is deterministic, always generating the same clusters when given the same data in the same order. However, the results can differ when data is provided in a different order. First, even though the core samples will always be assigned to the same clusters, the labels of those clusters will depend on the order in which those samples are encountered in the data. Second and more importantly, the clusters to which non-core samples are assigned can differ depending on the data order. This would happen when a non-core sample has a distance lower than eps to two core samples in different clusters. By the triangular inequality, those two core samples must be more distant than eps from each other, or they would be in the same cluster. The non-core sample is assigned to whichever cluster is generated first in a pass through the data, and so the results will depend on the data ordering.
     Implementation from : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    """
    def __init__(self, data, epsilon=None, min_samples=None):
        cls.Cluster.__init__(self, data)
        self.epsilon = epsilon
        self.min_samples = min_samples

    def _run(self):
        print("Clustering with hdbscan Algorithm...")
        self._clusterer = clusterer = hdbscan.HDBSCAN()
        self._clusterer.fit(self.data)
        self.nclust = self._clusterer.labels_.max()
        return clusterer.labels_












