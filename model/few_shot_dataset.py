import numpy as np # Linear algebra
import pandas as pd # Data wrangling
import sys

# Deep learning
import torch
import torch.utils.data as tud

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from main_translator import*
from main_decoder import main_decoder
from helper import *
from randomiser import Randomiser


class FewShotDataset(object):
    """
    Custom dataset for one-shot learning task.
    
    """
    
    def __init__(self, index, support_set, query_set, N,  threshold = None, replacement = False,\
                 balanced = False, randomise = False):
        """
        Initialiser.
        
        : index (int): index from the pd.Series containing query molecules
        : support_set (pd.DataFrame): molecules whose bioactivity is known and will be used to compare against the query molecule
        : query_set (pd.DataFrame): molecule whose bioactivity is unknown
        : threshold (int): bioactivity threshold
        : replacement (bool): determines whether sampling for the support set in classification task is with or without replacement
        : balanced (bool): determines whether the support set in classification task is necessarily balanced or not
        : randomise (bool): whether the SMILES strings that are being dealt with are random or not
        
        """
        
        self.index = index
        self.N = N
        self.threshold = threshold
        self.replacement = replacement
        self.balanced = balanced
        self.randomise = randomise
        
        # Both query and support sets get translated
        support_set_translated, query_set_translated = translate_sets(support_set, query_set, verbose = 0)
        
        self.support_set = pd.concat([support_set_translated, support_set['pXC50'], support_set['Activity']], axis = 1)
        self.query_set = pd.concat([query_set_translated, query_set['pXC50'], query_set['Activity']], axis = 1)
        
    def __call__(self, is_regression = False):
        """
        Callable.
        
        : is_regression (bool): 
        
        """
        
        self.is_regression = is_regression
        
        if self.is_regression:
            return self.create_regression_dataset()
        else:
            return self.create_dataset() 
    
    def create_regression_dataset(self):
        """
        Generates the support set and the corresponding dataloader for a regression task.
        
        """
        
        # (dfernandez) PERHAPS SHOULD BE ADDING FURTHER ASSERTIONS...
        try:
            assert self.threshold is None, 'No threshold should be specified for regression modelling.'
        except AssertionError as msg:
                sys.exit(msg)
        
        idxs_support_set = random.sample(list(self.support_set.index), self.N)
        k_support_set = self.support_set.loc[idxs_support_set,:]
        
        # Select compound from query set(test set)
        query_molecule = self.query_set.loc[self.index,'SMILES']
        query_molecule_label = self.query_set.loc[self.index,'pXC50']
        
        query_set_molecule = np.array([np.array(query_molecule) for i in range(len(k_support_set))])
        
        # Generating array with labels from support set
        label = np.array([k_support_set.iloc[i,1] for i in range(len(k_support_set))])
        
        # Generating array with repeated SMILES strings
        k_support_set = np.array([k_support_set.iloc[i,0] for i in range(len(k_support_set))])
              
        dataloader = self._initialise_dataloader(query_set_molecule, k_support_set, label)
        
        return dataloader
    
    def create_dataset(self):
        """
        Generates the support set and the corresponding dataloader for a classification task.
        
        """
        
        if self.threshold is not None:
            self.generate_support_set(self.threshold)
        
        # Warning issuing (perhaps not necessary)
        self.determine_sample()
        
        # Separate actives and inactives
        support_set_actives, support_set_inactives = self.sample_support_set()
        
        if self.randomise:
            randomiser = Randomiser(randomising_times = 1, is_n_shot = True)
            support_set_actives, support_set_inactives = [randomiser(i) for i in [support_set_actives, support_set_inactives]]
            query_randomiser = Randomiser(randomising_times = 1, is_n_shot = True)
            self.query_set = query_randomiser(self.query_set)
            
        # Generate kth support set
        k_support_set = pd.concat([support_set_actives, support_set_inactives], axis = 0).reset_index(drop = True)
        
        # Assign class labels based on active and inactive support sets
        k_support_set_label = np.concatenate([np.ones(len(support_set_actives)), np.zeros(len(support_set_inactives))])
        
        # Select compound from query set(test set)
        query_molecule = self.query_set.loc[self.index,'SMILES']
        self.query_molecule_label = self.query_set.loc[self.index,'Activity']
        
        query_set_molecule = np.array([np.array(query_molecule) for i in range(len(k_support_set))]) # arrays speed up operations
        
        # Similarity class label
        class_label = k_support_set.loc[:,'Activity'] == self.query_molecule_label
        class_label = np.array(class_label.astype(int))
        
        # Generating array with repeated SMILES strings
        k_support_set = np.array([k_support_set.iloc[i,0] for i in range(len(k_support_set))])
        
        dataloader = self._initialise_dataloader(k_support_set, query_set_molecule, class_label)
        
        return dataloader, k_support_set_label
    
    def generate_support_set(self, threshold):
        """
        Establishes a threshold for a given support set.
        
        : threshold (int): bioactivity threshold 
        
        """
        
        try:
            isinstance(threshold, (float, int))
        except TypeError:
            print('Please pass a float or an integer as a threshold')
        
        self.support_set.loc[self.support_set.loc[:, 'pXC50'] >= threshold, ['Activity']] = 1
        self.support_set.loc[self.support_set.loc[:, 'pXC50'] < threshold, ['Activity']] = 0
    
    def determine_sample(self):
        """
        If the size of the required support set exceeds the available number of instances, a warning is raised, albeit
        the process does not stop.
        
        """
        
        if self.balanced:
            # Total count of actives and inactives in the support set
            n_actives = (len(self.support_set[self.support_set['Activity']==1])//2)*2
            n_inactives = (len(self.support_set[self.support_set['Activity']==0])//2)*2
            
            self.N = min(n_actives, n_inactives, (self.N//2)*2) # Establish the bottleneck and force self.N to have the same value
            
        else:
            if self.N > len(self.support_set):
                raise ValueError('Cannot sample more than {} compounds.'.format(len(self.support_set)))
    
    def sample_support_set(self):
        """
        Generates support sets for active and inactive compounds with sampling with replacement
        (following the rationale of bootstrap aggregation)
        
        """
        
        if self.replacement:
            if self.balanced:
                actives, inactives = self.classify_compounds(self.support_set)
                support_set_actives_idxs, support_set_inactives_idxs = [np.random.choice(list(x.index), self.N//2) for x in [actives, inactives]]
                support_set_actives, support_set_inactives = [self.support_set.iloc[x,:] for x in [support_set_actives_idxs, support_set_inactives_idxs]]

            else:
                support_set_idxs = np.random.choice(list(self.support_set.index), self.N)
                support_set_compounds = self.support_set.iloc[support_set_idxs,:]
                support_set_actives, support_set_inactives = self.classify_compounds(support_set_compounds)
        
        else:
            if self.balanced:
                actives, inactives = self.classify_compounds(self.support_set)
                support_set_actives_idxs, support_set_inactives_idxs = [random.sample(list(x.index), self.N//2) for x in [actives, inactives]]
                support_set_actives, support_set_inactives = [self.support_set.iloc[x,:] for x in [support_set_actives_idxs, support_set_inactives_idxs]]

            else:
                support_set_idxs = random.sample(list(self.support_set.index), self.N)
                support_set_compounds = self.support_set.iloc[support_set_idxs,:]
                support_set_actives, support_set_inactives = self.classify_compounds(support_set_compounds)
           
        return support_set_actives, support_set_inactives
    
    def classify_compounds(self, support_set):
        """
        Support set compounds are separated according to the class they belong to.
        
        : support_set (pd.DataFrame): contains the class label for the relevant compounds 
        
        """
                
        support_set_actives = support_set[support_set['Activity']==1]
        support_set_inactives = support_set[support_set['Activity']==0]
        
        return support_set_actives, support_set_inactives
        
    def _initialise_dataloader(self, support_set_molecule, query_set_molecule, label):
        """
        
        : support_set_molecule (np.array): contains translated molecules which conform the support set
        : query_set_molecule (np.array): contains the translated molecule whose activity is to be established
        : label (np.array): similarity or pXC50 labels (for classification or regression, respectively)
        
        """
        # Conversion into tensors
        support_set_molecule = torch.from_numpy(support_set_molecule)
        query_set_molecule = torch.from_numpy(query_set_molecule)
        label = torch.from_numpy(label)
        
        tensor_dataset = tud.TensorDataset(support_set_molecule, query_set_molecule, label)
        
        
        return tud.DataLoader(dataset = tensor_dataset, shuffle = False, batch_size = self.N, 
                              collate_fn = FewShotDataset.collate_fn, drop_last = True, pin_memory = True)
    @staticmethod
    def collate_fn(batch_data):
        
        (inputs1, inputs2, labels) = zip(*batch_data)
        inputs1_lens = [np.count_nonzero(x.detach().numpy()) for x in inputs1]
        inputs2_lens = [np.count_nonzero(x.detach().numpy()) for x in inputs2]
        
        return torch.stack(inputs1), torch.stack(inputs2), inputs1_lens, inputs2_lens, torch.stack(labels) 