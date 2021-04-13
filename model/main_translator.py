import numpy as np
import pandas as pd
import sys

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from translator import *

def translate_sets(*args, one_hot_encode = False, verbose = 1):
    """
    Translates at once a given number sets based on dictionaries generated from the vocabulary of those same sets.
    
    : args (pd.DataFrame): DataFrames containing SMILES strings and their associated activity
    : one_hot_encode (bool): whether the sequence has to be one-hot encoded or not
    : verbose (int): regulates verbosity of translation
    
    """
    
    # Training, validation and test sets are concatenated in order to ensure that all the vocabulary is captured
    data = pd.concat([*args], axis = 0).reset_index(drop = True)
    
    # Dictionaries for translation are generated
    # SMILES column is sliced
    dict_generator = Translator(data)
    
    if verbose > 0:
        print('Generating dictionaries ...') 

    special_characters_dictionary, dictionary = dict_generator.init_dict()
    
    master_dictionary = MasterDictionary(special_characters_dictionary, dictionary, verbose = verbose)
    
    # Updating master dictionaries
    master_dictionary()
    
    # Loading master dictionaries
    master_dictionary, master_special_characters_dictionary = master_dictionary(load = True)
    
    # Translation is performed for the three datasets
    translated_args = [translate(dataset, master_dictionary, master_special_characters_dictionary, one_hot_encode) for dataset in [*args]]
    
    # Unlist the generated list if *args contains only one DataFrame
    if len(translated_args) == 1:
        translated_args = translated_args[0]
        
    if verbose > 0:
        print('Translation finished.')
    
    return translated_args

def translate(data, standard_dictionary, special_characters_dictionary, one_hot_encode):
    """
    Maps the SMILES strings contained in the rows of a given dataset into their corresponding array of integer values. 
    
    : data (pd.DataFrame): contains the rows with SMILES strings to be translated
    : standard_dictionary (dictionary): translation dictionary where each token is assigned a unique number
    : special_characters_dictionary (dictionary): translation dictionary of special characters where each special character is 
    assigned a unique character
    : one_hot_encode (bool): whether the sequence has to be one-hot encoded or not
    
    """
    
    t = Translator(data, one_hot_encode = one_hot_encode)
    
    translated = t.init_translation(standard_dictionary, special_characters_dictionary)
    
    return translated