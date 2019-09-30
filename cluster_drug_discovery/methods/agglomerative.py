from sklearn.cluster import AgglomerativeClustering
import cluster_drug_discovery.methods.clusterclass as cls


class AgglomerativeAlg(cls.Cluster):
    """
     Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree (or dendrogram). The root of the tree is the unique cluster that gathers all the samples, the leaves being the clusters with only one sample. See the Wikipedia page for more details.

     Implementation from : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
    """
    def __init__(self, data, nclust):
        cls.Cluster.__init__(self, data)
        self.nclust = nclust

    def _run(self):
        print("Clustering with Agglomerative Clustering...")
        self._clusterer = AgglomerativeClustering(n_clusters=self.nclust)
        return self._clusterer.fit_predict(self.data)












