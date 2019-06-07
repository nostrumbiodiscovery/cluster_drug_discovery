from sklearn.cluster import KMeans
import cluster_drug_discovery.methods.clusterclass as cls


class KmeansAlg(cls.Cluster):
    """
     K-Means Clustering Advantages and Disadvantages
     K-Means Advantages :
    
     1) If variables are huge, then  K-Means most of the times computationally faster than hierarchical clustering, if we keep k smalls.
    
     2) K-Means produce tighter clusters than hierarchical clustering, especially if the clusters are globular.
    
     K-Means Disadvantages :
    
     1) Difficult to predict K-Value.
     2) With global cluster, it didn't work well.
     3) Different initial partitions can result in different final clusters.
     4) It does not work well with different size and density clusters
     
     Implementation from : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """
    def __init__(self, data, nclust):
        cls.Cluster.__init__(self, data)
        self.nclust = nclust

    def _run(self):
        print("Clustering with Kmeans Algorithm...")
        self._clusterer = KMeans(n_clusters=self.nclust)
        return self._clusterer.fit_predict(self.data)












