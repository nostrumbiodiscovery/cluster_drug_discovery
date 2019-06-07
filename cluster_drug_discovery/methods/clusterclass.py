import cluster_drug_discovery.analysis.metrics as ms


class Cluster(object):

    def __init__(self, data):
        self.data = data

    def run(self):
        self._labels = self._run()
        return self._labels

    def analyze(self):
        ms.analysis_run(self.data, self)

        

