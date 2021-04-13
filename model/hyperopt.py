"""
Generates hyperparameter space to be assessed by the RSNN.

"""
# Of note, this class has to be sent to the cluster. Otherwise it fails due to lack of memory.
#### Attention this class would need to be updated
#### Attention this class could potentially be tweaked to generate array job

import itertools
import json
import numpy as np
import os
import pandas as pd
import random
import sys
import yaml

from csv import writer

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from helper import *
from snn import *
from trainer import *

class HyperParameterGenerator(object):
    def __init__(self, n_combinations, dataset, hidden_states_processing = None):
        """
        Initialiser.
        
        : n_combinations (int):
        : dataset (str):
        : hidden_states_processing (str, optional)

        """
        # Ensure reproducibility
        set_seed()
        
        self.n_combinations = n_combinations
        self.dataset = dataset
        self.hidden_states_processing = hidden_states_processing
        
        # Loads the dictionary from the corresponding dataset
        self.hyperparam_dictionary = self.get_hyperparameters()
        
        # Converts the dictionary into a list of keys and values
        self.hyperparams_list = self.convert_to_list()
    
    def set_hyperparameters(self):
        """
        Generates a .yaml file with hyperparameters to test.
        
        """
        
        hyperparameters = {
        'dataset': self.dataset,
        'SNN': {'hidden_size': [64, 512],
                'embedding_dimensions': [64, 512],
                'num_layers': [2,3,5],
                'cell_type': ['LSTM','GRU'],
                'embedding_dropout': [0.05, 0.5],
                'dropout': [0.05, 0.5],
                'learning_rate': [10**-5,0.1],
                'weight_decay': [0.00, 0.1],
                'dist_fn': ['cos','l1'],
                'weight_initialisation': ['xavier_uniform','xavier_normal','kaiming_uniform','kaiming_normal'],
                'similarity_function': ['exp','sigmoid'],
                'loss_fn': ['mse','bce','contrastive','logcosh','mae','l1','l2','huber'],
                'normalisation': ['batch','layer','']
                },
        'Training': {'batch_size': [64, 512],
                     'augmentation_factor_train': [1,2],
                     'augmentation_factor_val': [1,2]
                     }
        }

        if self.hidden_states_processing == 'self-attention':
            hyperparameters.update({'Attention': {'expansion_size': [64, 512],
                                                 'attention_layers': [64, 512],
                                                 'activation_fn': ['sigmoid','tanh','leaky ReLU']
                                                }})

        elif self.hidden_states_processing == 'internal processing':
            hyperparameters.update({'Processing': {'expansion_size': [64, 512],
                                                  'activation_fn': ['sigmoid','tanh','leaky ReLU']
                                                  }})    

        with open('/projects/../../PythonNotebooks/hyperparameters/'+ str(self.dataset) + '.yaml','w') as configFile:
            yaml.dump(hyperparameters, configFile)
    
    def get_hyperparameters(self):
        """
        Retrieves hyperparameters from already generated .yaml file.
        
        """
        
        with open('/projects/../../PythonNotebooks/hyperparameters/' + str(self.dataset) + '.yaml','r') as configFile:
            dictionary = yaml.load(configFile)
        
        return dictionary
        
    def convert_to_list(self):
        """
        Converts dictionary containing hyperparameter values into a list.
        
        """
        
        hyperparams_list = []
        
        # Converting dictionary into list
        for section in self.hyperparam_dictionary.keys():
            if section == 'dataset':
                pass
            else:
                hyperparams_list.append(list(self.hyperparam_dictionary[section].values()))
        
        # Generating one list out of the list of lists
        hyperparams_list = list(itertools.chain(*hyperparams_list))
        
        return hyperparams_list
    
    def get_keys(self):
        """
        Retrieves the keys from the dictionary containing the hyperparameter values.
        
        """
        
        # Indentifying primary keys
        dictionary_keys = self.hyperparam_dictionary.keys()
        
        key_list = []
        
        for key in dictionary_keys:
            if key == 'dataset':
                pass
            else:
                # Taking the secondary keys
                key_list.append(list(self.hyperparam_dictionary[key].keys()))
        
        keys = list(itertools.chain(*key_list))
        return keys
        
    def random_grid_search(self, hyperparameters_list):
        """
        Generates all possible hyperparameter combinations and selects a given number (n_combinations) at random.
        
        : hyperparameters_list (list)
        
        """
    
        all_combinations = list(itertools.product(*hyperparameters_list))
        
        random_combinations = random.sample(all_combinations, self.n_combinations)
        
        for i in range(self.n_combinations):
            yield random_combinations[i]


"""
Implements hyperparameter optimisation algorithm with random grid search for RSNN.

"""

class HyperOpt(object):
    def __init__(self, data, hyperparameter_generator, dataset_name, hidden_states_processing = None):
        """
        Intialiser.
        
        : n_iterations (int):
        : hyperparam_generator (class):
        : dataset_name (str):
        : hidden_states_processing (str):
        : attention (class, optional):

        """
        
        if hidden_states_processing is not None:
            try:
                assert hidden_states_processing in ['self-attention','internal processing']
            except:
                raise ValueError('Please, change hidden_states_processing to "self-attention"')
       
        self.data = data
        self.hyperparameter_generator = hyperparameter_generator
        self.dataset_name = dataset_name
        self.hidden_states_processing = hidden_states_processing
        
        # Create directories
        if not os.path.exists('/projects/../../PythonNotebooks/hyperparameters/statistics'):
            os.makedirs('/projects/../../PythonNotebooks/hyperparameters/statistics')
    
        self.hyperparameter_list = self.hyperparameter_generator.convert_to_list()
    
    def run_optimisation(self):
        """
        Carries out random search hyperparameter optimisation.
        
        """
        
        # Create dataset-specific directories
        if not os.path.exists('/projects/../../PythonNotebooks/hyperparameters/statistics/' + str(self.dataset_name)):
            os.makedirs('/projects/../../PythonNotebooks/hyperparameters/statistics/' + str(self.dataset_name))
        
        for hyperparam in self.hyperparameter_generator.random_grid_search(self.hyperparameter_list):
            self.keys = self.hyperparameter_generator.get_keys()
            hyperparam_dict = dict(zip(self.keys, hyperparam))
            
            # Generate values for model
            self.assign_hyperparameters(hyperparam_dict)
            
            # Run hyperparameter combination through the model
            stats = self.fit_model()
            
            # Reporting conditions for each experiment
            experiment_conditions = '; '.join(['%s = %s' % (key, value) for (key, value) in hyperparam_dict.items()])
            stats.insert(0, experiment_conditions)
            
            # Writing results in a csv file
            self.append_rows(self.dataset_name, stats)
            
    def assign_hyperparameters(self, hyperparameter_dictionary):
        """
        Assigns values for class instantiation.
        
        : hyperparameter_dictionary (dict):
        
        """
        
        if self.hidden_states_processing is not None:
            # Extraction of hidden state processing hyperparameters
            self.activation_fn = hyperparameter_dictionary['activation_fn']
            self.expansion_size = hyperparameter_dictionary['expansion_size']

            if self.hidden_states_processing == 'self-attention':
                self.attention_layers = hyperparameter_dictionary['attention_layers']
        
        # Extraction of model hyperparameters
        self.cell_type = hyperparameter_dictionary['cell_type']
        self.dist_fn = hyperparameter_dictionary['dist_fn']
        self.dropout = hyperparameter_dictionary['dropout']
        self.embedding_dimensions = hyperparameter_dictionary['embedding_dimensions']
        self.embedding_dropout = hyperparameter_dictionary['embedding_dropout']
        self.hidden_size = hyperparameter_dictionary['hidden_size']
        self.learning_rate = hyperparameter_dictionary['learning_rate']
        self.loss_fn = hyperparameter_dictionary['loss_fn']
        self.normalisation = hyperparameter_dictionary['normalisation']
        self.num_layers = hyperparameter_dictionary['num_layers']
        self.similarity_fn = hyperparameter_dictionary['similarity_function']
        self.weight_decay = hyperparameter_dictionary['weight_decay']
        self.weight_initialisation = hyperparameter_dictionary['weight_initialisation']
        
        # Training hyperparameters
        self.augmentation_factor_train = hyperparameter_dictionary['augmentation_factor_train']
        self.augmentation_factor_val = hyperparameter_dictionary['augmentation_factor_val']
        self.batch_size = hyperparameter_dictionary['batch_size']
        
    def fit_model(self):
        """
        Trains the model with the given hyperparameters with 10-fold cross validation and returns performance
        metrics (loss, accuracy, F1-score, precision, recall and MCC).
        
        """
        
        self.instantiate_model()
        
        # Training
        epochs = 150
        trainer = Train(self.model, epochs, self.batch_size, self.augmentation_factor_train, self.augmentation_factor_val, verbose = 0)
        data = translate_sets(self.data, verbose = 0)
        stats = trainer.cross_validate(10, data, hyperopt = True)
        
        return stats
        
    def instantiate_model(self):
        """
        Instantiates the model, so that it can be fitted thereafter.
        
        """
        
        if self.hidden_states_processing is None:
            self_hidden_state_processor = None
        elif self.hidden_states_processing == 'self-attention':
            self.hidden_states_processor = SelfAttention(self.expansion_size, self.attention_layers, activation_fn = self.activation_function)
        elif self.hidden_states_processing == 'internal processing':
            self.hidden_states_processor = InternalProcessing(self.hidden_size, self.expansion_size, activation_fn = self.activation_fn)
        else:
            raise ValueError('"hidden_states_processing" input is not correct.')
            
        self.model = SNN(self.hidden_size, self.num_layers, self.dist_fn, self.similarity_fn, self.cell_type, self.loss_fn,
                        self.learning_rate, self.weight_decay, self.embedding_dropout, self.dropout, 
                         hidden_states_processor = self.hidden_states_processor, embedding_dimensions = self.embedding_dimensions,
                         initialisation_process = self.weight_initialisation, normalisation = self.normalisation)

    @staticmethod
    def append_rows(file_name, metrics_list):
        """
        Appends metrics as a new row to csv file.
        From https://thispointer.com/python-how-to-append-a-new-row-to-an-existing-csv-file/
        
        : file_name (str): name to assign to csv file
        : metrics_list (list): list from cross-validation with performance metrics
        
        """
        
        with open(str(file_name),'a+',newline = '') as writeObj:
            # Create w writer object from csv module
            csv_writer = writer(writeObj)
        
            # Add contents of a list as last row in the csv file
            csv_writer.writerow(metrics_list)

            
# Quick test it runs
if __name__ =='__main__':

    hp_generator = HyperParameterGenerator(150, 'DRD2', 'self-attention')
    hp_generator.set_hyperparameters()

    df = pd.read_csv('/projects/../../datasets/standardised_datasets/NR1H2_standardised.csv')

    hyperopt = HyperOpt(df, hp_generator, dataset_name = 'test_dataset', hidden_states_processing = 'internal processing')
    hyperopt.run_optimisation()