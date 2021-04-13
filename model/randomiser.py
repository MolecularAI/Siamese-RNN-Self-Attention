"""
Generates randomised SMILES strings from canonical SMILES and translates them.

"""

import pandas as pd
import sys

from rdkit import Chem

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from helper import *
from main_decoder import main_decoder
from main_translator import *

class Randomiser():
    def __call__(self, DataFrame):
        """
        Callable.
        
        : DataFrame (pd.DataFrame): DataFrame containing canonical SMILES strings
        
        """
               
        if self.is_mlp:
            return DataFrame

        if self.is_n_shot:
            
            DataFrame.reset_index(drop = True, inplace = True)
            
            canonical_smi = main_decoder(DataFrame.iloc[:,0])
                
            # randomising
            randomised_smi = self.randomise(canonical_smi)
            
            # translating to tokens
            randomised = translate_sets(randomised_smi, verbose = 0)
            
            return pd.concat([randomised, DataFrame.iloc[:,1], DataFrame.iloc[:,2]], axis = 1)
            
        if self.randomising_times is not None:
            # Randomisation
            randomised_col1, randomised_col2 = list(map(self.randomise,[DataFrame.iloc[:,0], DataFrame.iloc[:,1]]))
            randomised_col1, randomised_col2 = pd.DataFrame(randomised_col1, columns = ['SMILES']), pd.DataFrame(randomised_col2, columns = ['SMILES'])

            # Translation
            translated_col1, translated_col2 = translate_sets(randomised_col1, randomised_col2, verbose = 0)

            # Augmenting the bioactivity label for each canonical SMILES
            # according to the randomised SMILES generated
            activity = pd.concat([DataFrame.iloc[:,-1]]*self.randomising_times).reset_index(drop = True)

            return pd.concat([translated_col1, translated_col2, activity], axis = 1)
        else:

            # Direct translation of canonical SMILES strings
            translated_col1, translated_col2 = translate_sets(DataFrame.iloc[:,0], DataFrame.iloc[:,1], verbose = 0)

            # Merging re-generate the pairs
            DataFrame = pd.concat([translated_col1, translated_col2, DataFrame.iloc[:,-1]], axis = 1)

            return DataFrame
        
    def __init__(self, randomising_times = None, is_mlp = False, is_n_shot = False, is_test = False):
        """
        Initialiser.
        
        : randomising_times (int, optional): number of randomised SMILES strings to generate from canonical SMILES
        : is_mlp (bool):
        : is_n_shot (bool): 
        
        """
        
        self.randomising_times = randomising_times
        self.is_mlp = is_mlp
        self.is_n_shot = is_n_shot
        self.is_test = is_test
    
    def randomise(self, smi):
        """
        Generates random SMILES given a canonical SMILES string.
        
        : smi (str): canonical SMILES string
        
        """
        
        mols = [Chem.MolFromSmiles(i) for i in smi]
        
        random_smiles = [[Chem.MolToSmiles(x, doRandom = True, isomericSmiles = False) for x in mols] for i in range(self.randomising_times)]
        
        random_smiles = [randomised_smiles for sublist in random_smiles for randomised_smiles in sublist]
        
        return pd.Series(random_smiles)