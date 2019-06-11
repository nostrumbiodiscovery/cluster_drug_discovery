
class FeatureExtractor(object):

    def __init__(self, files):
        self.files = files

    def retrieve_coords(self):
        return self._retrieve_coords()    

    def retrieve_dihedrals(self):
        return self._retrieve_dihedrals()
