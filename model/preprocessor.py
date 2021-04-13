""" 
Implementation of preprocessing steps.

"""

import gzip
import io
import molvs as mv
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import sys

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from molvs import standardize_smiles

class PreProcess(object):
       
    def __init__(self, DataFrame, threshold = None, set_threshold = False, standardise = True, process = True):
        """
        Initialiser.
        
        : name (str/pd.DataFrame):
        : threshold (int):
        : set_threshold (bool):
        : standardise (bool):
        
        """
        
        self.threshold = threshold
        self.set_threshold = set_threshold
        self.process = process
        self.standardise = standardise
        self.DataFrame = DataFrame
        
        # path with stored datasets
        self.path = '/projects/../../datasets/'
        self.pool = mp.Pool(processes = mp.cpu_count())   
        
        if self.standardise:
            if self.process:
                self.name = DataFrame
                # Preparing data for preprocessing  
                self.open_file()
                self.filter_data()
            
            self.standardiser = mv.Standardizer()
            self.salt_remover = SaltRemover()
            self.accepted_atoms = ['H','C','N','O','F','S','Cl','Br']
        
        
    def open_file(self):
        """
        Opens the file depending on the type of dataset provided (which, in its turn depends on the data source)
                
        """
        
        # ExCAPE-DB or other files files
        if os.path.isfile(self.path + self.name + '.csv'):
            self.DataFrame = pd.read_csv(self.path + self.name + '.csv')
            
        #ChEMBL files
        elif os.path.isfile(self.path + self.name + '.gz'):
            with gzip.open(self.path + self.name + '.gz', 'rb') as f:
                file_content = f.read()
                self.DataFrame = pd.read_csv(io.StringIO(file_content.decode('utf-8')), delimiter = ';')

        # Pipeline pilot files
        elif os.path.isfile(self.path + self.name + '.txt'):
            self.DataFrame = pd.read_csv(self.path + self.name + '.txt', delimiter = '\t')
                
        else:
            raise FileNotFoundError
                    
    def filter_data(self):
        """
        Pre-treatment of files for ChEMBL, ExCAPE and Pipeline pilot datasets.
        WARNING: if they come from another database they should be handled manually!
        
        """   
    
        # ChEMBL files
        if os.path.isfile(self.path + self.name + '.gz'):
            DataFrame = self.DataFrame[['Smiles','Standard Type','Standard Relation', 'Standard Value', 'Standard Units']]
            DataFrame = DataFrame.dropna()
            DataFrame = DataFrame.loc[DataFrame['Standard Type'].str.contains('C50$'),:]
            DataFrame = DataFrame.reset_index(drop = True)
            DataFrame['Standard Value'] = DataFrame.loc[DataFrame['Standard Units'] == 'nM',:]['Standard Value']/(10**9)
            DataFrame['Standard Relation'] = DataFrame.loc[DataFrame['Standard Relation'] == "'='",:]
            DataFrame = DataFrame.dropna()
            DataFrame = DataFrame.reset_index(drop = True)
            DataFrame['pXC50'] = -np.log10(DataFrame['Standard Value'])
            DataFrame.drop(columns = ['Standard Units', 'Standard Relation','Standard Value','Standard Type'], inplace = True)
            DataFrame.columns.values[0] = 'SMILES'
            self.DataFrame = DataFrame[['pXC50','SMILES']]          
            
        # Pipeline pilot files
        elif os.path.isfile(self.path + self.name + '.txt'):
            DataFrame = self.DataFrame.loc[self.DataFrame['CC_DbName'] != 'IBIS',:]
            DataFrame = DataFrame.loc[DataFrame['CC_OriginalUnit'] != '%',:]
            DataFrame = DataFrame[['CC_Structure','CC_ActivityValue', 'CC_OriginalResult','CC_OriginalResultType']]
            DataFrame.columns = ['SMILES','pXC50','Original_result','Standard Type']
            self.DataFrame = DataFrame.loc[DataFrame['Standard Type'].str.contains('C50$'),:]
        
        self.DataFrame = self.DataFrame.dropna()
        self.DataFrame = self.DataFrame.reset_index(drop = True)
        
        if self.set_threshold:
            self.DataFrame = self.DataFrame[['SMILES','pXC50']]
        else:
            self.DataFrame = self.DataFrame[['SMILES','Activity']] # No pXC50 column is needed
        
    def preprocess(self, remove_charge = False):
        """
        Prepare the dataset by standardising all the SMILES strings provided. 
        Activity thresholds are set if desired.
        
        : remove_charge (bool): boolean
        
        """
        
        self.duplicates_removed = False

        if self.standardise:
            self.DataFrame = self.to_molecule(self.DataFrame)
            self.DataFrame.reset_index(drop = True, inplace = True)
            
            standardised_smiles = [self.pool.apply(self.standardise_compound, args = (x,self.standardiser, self.salt_remover, self.accepted_atoms,)) for x in self.DataFrame['SMILES']]
            
            if self.set_threshold:
                # There is no activity column but pXC50. Hence, activities have to be assigned
                self.DataFrame = pd.concat([self.DataFrame['pXC50'], pd.Series(standardised_smiles)], axis = 1)
            else:
                # It is assumed that the activities are already filtered and therefore there is an activity column
                self.DataFrame = pd.concat([self.DataFrame['Activity'], pd.Series(standardised_smiles)], axis = 1)

            self.DataFrame.dropna(inplace = True)
            self.DataFrame.reset_index(drop = True, inplace = True)
            self.DataFrame.columns = [*self.DataFrame.columns[:-1], 'SMILES']
        
        if not self.duplicates_removed:
            self.remove_duplicates()

        if self.set_threshold:
            self.DataFrame = self.assign_activities(self.standardise_activities(self.DataFrame), self.threshold)
        
        return self.DataFrame
        
    def to_molecule(self, DataFrame):
        """
        Converts SMILES strings to Molecules
        
        : DataFrame (pd.DataFrame): DataFrame with SMILES strings to be converted into Molecules
        
        """
        
        DataFrame['SMILES'] = DataFrame['SMILES'].map(lambda x: Chem.MolFromSmiles(x))
        
        return DataFrame
    
    @classmethod
    def standardise_compound(cls, mol, standardiser, salt_remover, organic_atoms, remove_charge = False):
        """
        Compound standardisation according to the following steps
           1. Standardises the molecule
           2. Removes salts using SaltRemover
           3. Retrieves fragments from molecules
           4. Selects the organic fragment from the molecule
           5. Removes the charge if required
           6. Filters the molecule by number of heavy atoms
           7. Filters the molecule by string length
           
        : mol (rdkit.Chem.rdchem.Mol): 
        : standardiser ():
        : salt_remover ():
        : organic_atoms (list): 
        : remove_charge (bool):
        
        """
    
        min_heavy_atoms = 0
        max_heavy_atoms = 50
        max_len = 150
        
        try:
            mol = cls.standardise_mol(mol, standardiser)
            mol = salt_remover.StripMol(mol)

            if mol.GetNumAtoms()==0:
                return None

            fragments = Chem.GetMolFrags(mol, asMols = True)

            selected_fragment = None

            for fragment in fragments:
                if cls.is_organic(fragment, organic_atoms):
                    if selected_fragment is None:
                        selected_fragment = fragment
                    else:
                        # The organic fragment has already been found (e.g. there are multiple organic fragments)
                        selected_fragment = None
                        break

            if selected_fragment is None:
                return None

            if remove_charge:
                mol = cls.remove_charge_mol(selected_fragment)

            if min_heavy_atoms <= mol.GetNumHeavyAtoms() <= max_heavy_atoms:
                smiles = Chem.MolToSmiles(selected_fragment, isomericSmiles = False, canonical = True)   

                if len(smiles) <= max_len:
                    return smiles
                
        except Exception as e:
            print('Exception')
            print(e)
    
    @classmethod
    def standardise_mol(cls, mol, standardiser):
        """
        Standardisation of a given molecule following the super_parent approach from MolVS but without charge_parent
        nor isotope_parent methods

        : mol (rdkit.Chem.rdchem.Mol):
        : standardiser ():
        
        """
        
        mol = standardiser.standardize(mol)
        mol = standardiser.stereo_parent(mol, skip_standardize = True)
        mol = standardiser.tautomer_parent(mol, skip_standardize = True)
        mol = standardiser.standardize(mol)
        
        return mol

    def remove_charge_mol(self,mol):
        """
        
        
        : mol (rdkit.Chem.rdchem.Mol):
        
        """
        
        return self.standardiser.charge_parent(mol, skip_standarize = False)
    
    @classmethod
    def is_organic(cls, fragment, organic_atoms):
        """
        Identifies the organic fragment of a molecule if the molecule consists of more than one fragment
        
        : fragment: fragment of the molecule to assess
        : organic atoms: list of atoms considered to configure an organic molecule. Disregards atoms which could
        yield organometallic compounds (i.e. Mg, Cu, Fe, etc.)
        """
        
        contains_carbon = False
        in_organic_set = True

        for atom in fragment.GetAtoms():
            atom_symbol = atom.GetSymbol()
            
            if atom_symbol not in organic_atoms:
                in_organic_set = False
                break

            if atom_symbol == 'C':
                contains_carbon = True

        return contains_carbon and in_organic_set
    
    def standardise_activities(self, DataFrame):
        """
        Takes all the pXC50s collected by different assays and calculates the median if the maximum pXC50 is not 2.5
        higher than the minimum reported pXC50.
                
        : DataFrame (pd.DataFrame) : DataFrame from which to estimate median pXC50
        
        """
        
        DataFrame = DataFrame.groupby('SMILES').agg(
                                            min_val = ('pXC50', min),
                                            max_val = ('pXC50', max),
                                            median = ('pXC50', 'median'))

        DataFrame = DataFrame[(DataFrame['max_val']/DataFrame['min_val'])<2.5]
        DataFrame.reset_index(level = 0, inplace = True)
        DataFrame.drop(columns = ['min_val','max_val'], inplace = True)
        DataFrame.columns = ['SMILES','pXC50']

        return DataFrame
    
    @staticmethod
    def assign_activities(DataFrame, threshold):
        """
        Classifies activities according to a given threshold.
        
        : DataFrame (pd.DataFrame): 
        : threshold (int): integer. Specifies the threshold
        
        """
        
        DataFrame.loc[DataFrame.loc[:,'pXC50'] < threshold, 'Activity'] = 0
        DataFrame.loc[DataFrame.loc[:,'pXC50'] >= threshold, 'Activity'] = 1
    
        return DataFrame

    def remove_duplicates(self):
        """
        
        """
            
        self.DataFrame.drop_duplicates(subset = 'SMILES', inplace = True)
        self.DataFrame.reset_index(drop = True, inplace = True)

        self.duplicates_removed = True

    def save_data(self, name):
        """
        
        : name (str):
        
        """
        
        standardised_path = '/projects/../../datasets/standardised_datasets/'

        self.DataFrame.to_csv(standardised_path + name + '_threshold_' + str(self.threshold) + '_standardised.csv')