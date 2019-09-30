import os
import numpy as np
import cluster_drug_discovery.analysis.metrics as ms
import cluster_drug_discovery.helpers.helpers as hs
import cluster_drug_discovery.dimensionallity.reduction as rd


class Cluster(rd.ReduceDimension):

    def __init__(self, data):
        self.data = data
        if np.array(data).shape[1] > 2:
            rd.ReduceDimension.__init__(self, self.data, 2)

    def run(self):
        self._labels = self._run()
        return self._labels

    def analyze(self,folder="analisis", range_clust=[2, 3, 4, 5, 6]):
        if os.path.exists(folder):
            raise OSError("Folder {} already exists".format(folder))
        else:
            os.mkdir(folder)
            with hs.cd(folder):
                ms.analysis_run(self.data, self, range_n_clusters = range_clust)

        

