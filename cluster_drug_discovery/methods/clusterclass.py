import cluster_drug_discovery.metrics.silhouette as sl


class Cluster(object):

    def __init__(self, data):
        self.data = data

    def run(self):
        self._labels = self._run()
        return self._labels

    def silhouette(self):
        sl.silhouette_run(self.data, self)
        

