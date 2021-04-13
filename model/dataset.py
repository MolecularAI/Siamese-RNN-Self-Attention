"""
Implementation of custom dataset to feed into the Recurrent Siamese Neural Network.

"""

import numpy as np # Linear Algebra
import pandas as pd # Data wrangling
import sys

# Deep Learning
import torch 
import torch.utils.data as tud

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from helper import *

class Dataset(tud.Dataset):      
        
    def __init__(self, DataFrame, transform = None):
        """
        Initialiser.
        
        : DataFrame (pd.DataFrame): DataFrame from which pairs will be generated
        : transform (callable, optional): generation of pairs.
        
        """
        
        # Ensuring reproducibility
        set_seed()
        
        # Data loading
        self.DataFrame = DataFrame
        self.transform = transform
        
        # Applying transformations if required (pairing)
        if self.transform:
            self.DataFrame = self.transform(self.DataFrame)
        
    def __getitem__(self, idx):
        """
        Allows indexing the DataFrame (dataset).
        
        : idx (int):
        
        """
        
        return self.DataFrame[idx]
    
    def __len__(self):
        """
        Overloading of the length function. 
        
        """
        return len(self.DataFrame)
    
    @staticmethod
    def collate_fn(batch_data):
        """
        Allows calculation of length of the input sequences, so that the padded sequences can be packed afterwards.
        
        : batch_data (torch.Tensor): contains mini-batch data
        
        """
        
        (inputs1, inputs2, labels) = zip(*batch_data)
        inputs1_lens = [np.count_nonzero(x.detach().numpy()) for x in inputs1]
        inputs2_lens = [np.count_nonzero(x.detach().numpy()) for x in inputs2]
        
        return torch.stack(inputs1), torch.stack(inputs2), inputs1_lens, inputs2_lens, torch.stack(labels)
    
    
"""
Converts paired DataFrame with numpy arrays into Pytorch tensors.

"""
    
class ToTensor():
    
    def __init__(self, is_mlp = False):
        """
        Constructor.
        
        : is_mlp (bool): indicates whether MLP is going to be trained (ECFP-based)
        
        """
        
        self.is_mlp = is_mlp
           
    def __call__(self, DataFrame):
        """
        Callable.
        
        : DataFrame (pd.DataFrame): DataFrame containing
        
        """
        
        # Contains fingerprints
        if self.is_mlp:
            pairs1, pairs2, label = torch.tensor(DataFrame.iloc[:,:2048].values), torch.tensor(DataFrame.iloc[:,2048:-1].values), torch.tensor(DataFrame.iloc[:,-1].values)
        # Contains tokenised SMILES
        else:
            pairs1, pairs2, label = torch.tensor(DataFrame.iloc[:,0], dtype = torch.long), torch.tensor(DataFrame.iloc[:,1], dtype = torch.long), torch.tensor(DataFrame.iloc[:,2])
        
        # Wrap the tensors
        dataset = tud.TensorDataset(pairs1, pairs2, label)
        
        return dataset