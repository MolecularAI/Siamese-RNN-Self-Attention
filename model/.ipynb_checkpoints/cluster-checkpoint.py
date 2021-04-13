"""
This class implements clustering of a given set of fingerprints based on Butina clustering from RDKit
https://www.rdkit.org/docs/source/rdkit.ML.Cluster.Butina.html
    
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

from operator import itemgetter
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

class Cluster(object):
    def __call__(self):
        return self.make_clusters()
    
    def __init__(self, smi, dataset_name = None):
        """
        Initialiser.
        
        : smi (pd.Series) :
        : dataset_name (str):
        
        """
        
        self.smi = smi
        self.dataset_name = dataset_name
        self.mol = self.generate_molecule()
        self.fps = self.calculate_fps()
    
    def generate_molecule(self):
        """
        Generates RDKit Mol object from SMILES strings
        
        """
        
        return [Chem.MolFromSmiles(smiles) for smiles in self.smi]
    
    def calculate_fps(self, radius = 3, bits = 2048):
        """
        Calculates fingerprints from RDKit mol object.
        
        : radius (int, default):
        : bits (int, default):
        
        """
        
        return [AllChem.GetMorganFingerprintAsBitVect(molecule, radius, bits) for molecule in self.mol]
    
    def cluster_fps(self, fps, cutoff = 0.5):
        """
        Generation of clusters given a threshold.
        
        : fps (pd.Series): Series containing the FPS of each molecule
        : cutoff (float): elements within this range of each other are considered to be neighbors. Ranging from 0 to 1
        
        """       
        
        dists = []
        n_fps = len(fps)
        
        # Distance matrix generation
        for i in range(1, n_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1-x for x in sims])
        
        # Now the data is clustered
        cs = Butina.ClusterData(dists, n_fps, cutoff, isDistData = True)
        
        print('There are {} clusters'.format(len(cs)))
        
        return cs
    
    def make_clusters(self):
        """
        The DataFrame is handled so that clusters can be prepared.
        
        """
    
        self.clusters = self.cluster_fps(self.fps)
        self.filtered_clusters = self._remove_clusters(self.clusters)
        return self.filtered_clusters
    
    def plot_cluster_distribution(self):
        """
        Plots distribution of clusters by cluster size and Tanimoto similarity.
        
        """
        
        # Calculating cluster sizes
        cluster_sizes = self.calculate_cluster_size(self.clusters)
        
        # Estimating Tanimoto similarities
        tanimoto = self.calculate_tanimoto_similarity(self.filtered_clusters)
        
        # Plotting it
        fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (20, 10))

        ax[0].plot(cluster_sizes)
        ax[0].set_title('Cluster sizes', size = 12)
        ax[0].set_xlabel('cluster index', size = 13.5)
        ax[0].set_ylabel('log$_{10}$(cluster size)', size = 13.5)
        ax[0].grid()

        ax[1].hist(tanimoto, color = 'skyblue', alpha = 0.4)
        ax[1].set_xlim(0,1)
        ax[1].set_xlabel('Tanimoto similarity', size = 13.5)
        ax[1].set_ylabel('N compounds', size = 13.5)
        ax[1].set_title('Distribution of Tanimoto similarity from centroid', size = 12)
        ax[1].grid()
        
        plt.margins(0.01)
        plt.show()

        print('The maximum size of the clusters is {}.'.format(max(list(map(len,self.clusters)))))
    
    @staticmethod
    def calculate_cluster_size(clusters):
        """
        Helper function to calculate cluster sizes and express them with logarithmic basis.
        
        : clusters (list): clusters whose size is to be calculated
        
        """
        
        cluster_sizes = [np.log10(len(cluster)) for cluster in clusters]
        cluster_sizes.sort()
        
        return cluster_sizes
    
    def calculate_tanimoto_similarity(self, clusters):
        """
        Calculate Tanimoto similarity between centroid and the rest of the compounds in a cluster.
        
        : clusters (list): clusters with compounds whose Tanimoto similarity will be calculated
        
        """
        
        # Calculating Tanimoto similarity
        tanimoto_similarity = []

        for cluster in clusters:
            tuple_cluster_compounds = itemgetter(*list(cluster))(self.fps) # select relevant FPS
            cluster_compounds = [fp for fp in tuple_cluster_compounds]
            centroid = cluster_compounds[0]
            similarity_cluster = []
            
            for compound in cluster_compounds[1:]:
                tanimoto_distance = self.calculate_bit_similarity(centroid, compound)
                similarity_cluster.append(tanimoto_distance)

            tanimoto_similarity.append(similarity_cluster)

        flattened_tanimoto = [similarity for sublist in tanimoto_similarity for similarity in sublist]
        
        # Saving the tanimoto similarities
        with open('/projects/cc/kdqm927/PythonNotebooks/cluster_similarity/' + str(self.dataset_name) + '.pkl','wb') as f:
            pickle.dump(flattened_tanimoto,f)
        
        return flattened_tanimoto
    
    @staticmethod
    def calculate_bit_similarity(fp1, fp2, metric = 'tanimoto'):
        """
        Function taken from Eva Nittinger from Computational Chemistry Department led by Christian Tyrchan.
        
        : fp1 ():
        : fp2 ():
        : metric (str): similarity metric to assess similarity between fp1 and fp2. Tanimoto distance selected as default
        
        """
        
        metrics_available = {
            "allbit": DataStructs.AllBitSimilarity,  # bit
            "asymmetric": DataStructs.AsymmetricSimilarity,  # bit
            "braunblanquet": DataStructs.BraunBlanquetSimilarity, # bit
            "cosine": DataStructs.CosineSimilarity,  # bit
            "dice": DataStructs.DiceSimilarity, # bit or int
            "kulczynski": DataStructs.KulczynskiSimilarity,  # bit
            "mcconnaughey": DataStructs.McConnaugheySimilarity, # bit
            "rogotgoldberg": DataStructs.RogotGoldbergSimilarity,  # bit
            "russel": DataStructs.RusselSimilarity,  # bit
            "sokal": DataStructs.SokalSimilarity,  # bit
            "tanimoto": DataStructs.TanimotoSimilarity, # bit or int
            "tversky": DataStructs.TverskySimilarity # bit or int
        }
        
        if metric.lower() not in metrics_available:
            print ("The given metric is unknown! Selecting 'tanimoto' as default")
            metric = 'tanimoto'
        
        # Calculate Fingerprint similarity Matrix
        return metrics_available[metric.lower()](fp1, fp2)
    
    @staticmethod
    def _remove_clusters(clusters, size = 2):
        """
        Removes clusters of below a given number of compounds.
        
        : size (int, default): maximum number of compounds in a cluster to be removed
        
        """ 
        
        filtered_clusters = [cluster for cluster in clusters if len(cluster)>size]
        
        print('There are {} singlets and doublets'.format(len(clusters) - len(filtered_clusters)))
        
        return filtered_clusters