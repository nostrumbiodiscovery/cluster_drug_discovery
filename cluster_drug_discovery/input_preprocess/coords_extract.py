from prody import *
import cluster_drug_discovery.input_preprocess.feature_extraction as fte


class CoordinatesExtractor(fte.FeatureExtractor):

    def __init__(self, files, residues):
        self.residues = residues
        self.files = files
        fte.FeatureExtractor.__init__(self, files)

    def _retrieve_coords(self):
        all_files_coords = []
        for f in self.files:
            coordinates = []
            atoms = parsePDB(f)
            for res in self.residues:
                residue_specified = atoms.select('resname {}'.format(res))
                coordinates.extend(atom.getCoords().tolist() for atom in residue_specified)
            all_files_coords.append(coordinates)
        return all_files_coords





if __name__ == "__main__":
    extraction = CoordinatesExtractor(["/home/ywest/Downloads/ref.pdb",] , ["ATP",])
    extraction.retrieve_coords()   
                

