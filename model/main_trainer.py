"""
Makes it easier to train a RSNN. This class has been tested once, consequently its robustness is not 100% guaranteed yet.

"""

import ast 
import numpy as np # Linear algebra
import pandas as pd # Data wrangling
import re # Regular expressions
import sys
import matplotlib
import yaml

# Avoid plotting graphs
matplotlib.use('Agg')

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')

from snn import *
from splitter import *
from trainer import *
from unbuffered import *

from typing import Union

# Ensures output gets printed
# from SLURM
sys.stdout = Unbuffered(sys.stdout)

def main_trainer(path, how = 'k-fold', save = True):
    """
    : path (str): path to the .yaml file containing model and training info
    : how (str): which kind of training to apply
    : save (bool): whether the model is saved or not

    """
    
    dataset_name, model_dict, train_dict = extract_info(path)
    
    # String arguments standardisation
    how = how.lower()   
    dataset_name = dataset_name.lower()
    
    dataset_path = '/projects/../../PythonNotebooks/datasets/' + str(dataset_name.lower()) + '.csv'
    
    # Loading the dataset
    df = pd.read_csv(dataset_path, index_col = 0)
    
    # Splitting the dataset
    split = Splitter(df, 50, 40)
    if 'is_regression' in list(model_dict.keys()) and model_dict['is_regression'] == True:
        train, test, val = split.random_split(is_regression = True)
    else:
        train, test, val = split.random_split()
    
    dataset_name = check_dictionaries(dataset_name, model_dict, train_dict)
    
    if model_dict is not None:
        rsnn = SNN(**model_dict)
    else:
        rsnn = SNN()

    if train_dict is not None:
        trainer = Train(rsnn, dataset_name, **train_dict)
    else:
        trainer = Train(rsnn, dataset_name)
    
    if how == 'k-fold':
        trainer.cross_validate(10, train, val)
    elif how == 'fit':
        trainer.fit(train, val)
    else:
        raise ValueError('RSNN must be trained either with "fit" or "k-fold" modes')
    
    if save:
        trainer.save_model(dataset_name)

def extract_info(path: str) -> Union[str, dict]:
    # Separating relevant hyperparameters
    # to get the dictionaries
    
    # Loading .yaml files with training hyperparameters
    with open(path,'r') as configFile:
        file = yaml.load(configFile, Loader = yaml.FullLoader)
    
    dataset_name = file['NAME']
    model_dict = file['MODEL']

    training_dict = file['TRAINING']
    
    return dataset_name, model_dict, training_dict
        
def dictionary_update(dict1, dict2):
    """
    Updates dictionary values if required
    
    : dict1 (dict): reference dictionary
    : dict2 (dict): dictionary with potential updates
    
    """
    
    # Generate a set of common keys
    common_keys = dict1.keys() & dict2.keys()
    
    # Updating hyperparameters
    if len(common_keys) != 0:
        for k in common_keys:
            dict1[k] = dict2[k]

    return dict1

def check_dictionaries(dataset_name, model_dict = None, train_dict = None):
    """
    Checks that dictionaries are filled in correctly, meeting models' specifications.
    
    : dataset_name (str): name assigned to the model for saving purposes
    : model_dict (dict, optional): contains hyperparameters for SNN model
    : train_dict (dict, optional): contains hyperparameters for training the SNN model
    
    """
    
    if model_dict is not None:
        print(model_dict)
        if 'is_mlp' in list(model_dict.keys()):
            assert train_dict is not None, 'Please, introduce training dictionary.'
            assert 'is_mlp' in list(train_dict.keys()), 'Please, introduce is_mlp key in the training dictionary.'
            if model_dict['is_mlp'] is True:  
                assert train_dict['is_mlp'] is True, 'Please, introduce is_mlp = True in the training dictionary, too.'
                dataset_name += '_mlp_'
            else:
                # Adding hidden states processor in the dataset name
                assert model_dict['hidden_states_processor'] is not None, 'Please introduce hidden states processor'
                if model_dict['hidden_states_processor']=='internal_processing':
                    dataset_name += '_internal_processing_'
                elif model_dict['hidden_states_processor']=='attention':
                    dataset_name += '_attention_'
                else:
                    raise NotImplementedError('hidden states processor not recognised.')
                
        if 'is_regression' in list(model_dict.keys()) and model_dict['is_regression'] is True:
            assert train_dict is not None, 'Please, introduce training dictionary.'
            assert 'is_regression' in list(train_dict.keys()), 'Please, introduce is_regression key in the training dictionary.'
            assert train_dict['is_regression'] is True, 'Please, introduce is_regression = True in the training dictionary, too.'
            dataset_name += '_regression_'
    else:
        # default hidden states processor not covered in previous cases
        dataset_name += 'attention'

    if train_dict is not None:
        # Update dataset name according to conditions
        if 'augmentation_factor_train' in list(train_dict.keys()) and train_dict['augmentation_factor_train'] > 1:
            dataset_name += '_train_augmentation_' + str(train_dict['augmentation_factor_train'])

        if 'randomising_times' in list(train_dict.keys()) and train_dict['randomising_times'] is not None:
            dataset_name += '_random_' + str(train_dict['randomising_times'])
        else:
            dataset_name += '_canonical'
   
    # Remove potential double underscores
    dataset_name = re.sub('__','_', dataset_name)
    
    return dataset_name

if __name__ == '__main__':
    file, how = sys.argv[1], sys.argv[2]
    main_trainer(file, how)