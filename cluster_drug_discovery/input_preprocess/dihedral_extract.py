from prody import *
import numpy as np
import cluster_drug_discovery.input_preprocess.feature_extraction as fte


class DihedralsExtractor(fte.FeatureExtractor):

    def __init__(self, files, dihedrals):
        self.dihedrals = dihedrals
        self.files = files
        fte.FeatureExtractor.__init__(self, files)

    def _retrieve_dihedrals(self):
        assert type(np.array(self.dihedrals)) == np.ndarray , "dihedrals need to be an array of shape(X, 3)"
        assert np.array(self.dihedrals).shape[1] == 3, "dihedrals need to be an array of shape (X, 3)"
        all_files_dihedrals = []
        for f in self.files:
            dihedrals = []
            atoms = parsePDB(f)
            for atom1, atom2, atom3 in self.dihedrals:
                dihedral = calcAngle(atoms[atom1], atoms[atom2], atoms[atom3])
                dihedrals.append(dihedral)
            all_files_dihedrals.append(dihedrals)
        return all_files_dihedrals



if __name__ == "__main__":
    extraction = DihedralsExtractor(["/home/ywest/Downloads/ref.pdb",] , [[11,12,13],])
    extraction.retrieve_dihedrals()   
                

