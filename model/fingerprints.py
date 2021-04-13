"""
Implementation of circular fingerprint calculation. Class developed from a pre-existing class from Vigneshwari Subramanian, 
postdoc in Computational Chemistry (led by Christian Tyrchan) and under the supervision of Ola Engkvist and Antonio Llinás.

"""

import numpy as np # Linear algebra
import pandas as pd # Data wrangling

# Chemistry packages
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import DataStructs

class Fingerprint(object):
    def __init__(self, smiles):
        """
        Initialiser.
        
        : smi (pd.Series): contains canonical SMILES strings
        
        """
        
        self.smiles = smiles
        
        # Performs conversion into RDKit molecules automatically
        self.smiles_converted = self.smiles_convert()
        
    def smiles_convert(self):
        """
        Converts SMILES strings into RDKit molecules suitable for fingerprint calculation.
        
        """
        
        smiles_convert = [Chem.MolFromSmiles(smiles) for smiles in self.smiles]
                    
        return smiles_convert

    def morgan(self, radius, size = None):
        """
        Calculates circular fingerprints and renders them as a DataFrame, so that it is easier to handle.
        
        : radius (int): radius = 2 ~ ECFP4, radius = 3 ~ ECFP6
        : size (int, optional): number of bits to generate. If None, it will be assigned 
        the standard value of 2048
        
        """
        
        if size is None:
            size = 2048
            
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, size) for m in self.smiles_converted]
        np_fps = []
        
        for fp in fps:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            np_fps.append(arr)
            
        df = pd.DataFrame(np_fps)
        
        return df
    
def calculate_circular_fp(data, radius = 3, size = 2048):
    """
    Generates ~ECFP6 given SMILES strings.
    
    : data (pd.DataFrame):
    : radius (int):
    : size (int):
    
    """
    
    fp = Fingerprint(data)
    fps = fp.morgan(radius = radius, size = size)
    
    return fps