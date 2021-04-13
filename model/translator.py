import numpy as np # Linear algebra
import re # Regular expressions
import pandas as pd # Data wrangling
import string
import sys

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from helper import *

class Translator(object):
              
    def __init__(self, DataFrame, maxlen = 150, one_hot_encode = False):
        """
        Constructor.
        
        : DataFrame (pd.DataFrame): structures to be encoded
        : maxlen (int, default = 150): activities corresponding to compounds fed into the system
        : one_hot_encode (bool): toggle to one-hot encode
        
        """
        
        self.DataFrame = DataFrame
        
        if isinstance(DataFrame, pd.Series):
            self.structures = DataFrame
        else:
            self.structures = self.DataFrame['SMILES']
        
        self.maxlen = maxlen
        self.additional_chars = set()
        self.special_char_regex = r'(\[[^\]]*\])'
        self.sos, self.eos = '^', '$'
        self.ohe = one_hot_encode
            
    def encode(self):
        """
        Encodes SMILES strings
        
        """
        
        translated = [[self.dictionary[atom] for atom in molecule] for molecule in self.smiles]
        
        if self.ohe:
            translated = [self.one_hot_encode(molecule) for molecule in translated]
            
        #ranslated = [np.asarray(i) for i in translated]
        self.smiles = pd.Series(translated).map(lambda x: np.asarray(x))
        
        #elf.smiles.append([translated])
        
    def pad(self):
        """
        Pads SMILES strings in the given series to the maximum length specified in the constructor.
        
        """
                
        if self.ohe:
            seq_lengths = list(map(len, self.smiles))
            self.padded = pd.DataFrame(index = np.arange(len(self.smiles)), columns = ['SMILES'])
            
            # Padding the sequence with 0s at the bottom
            for idx, (seq, seqlen) in enumerate(list(zip(self.smiles, seq_lengths))):
                seq_smi = np.zeros((self.maxlen, len(self.dictionary)))
                seq_smi[:seqlen] = seq
                self.padded.iloc[idx,0] = seq_smi
            
        else:
            self.padded = self.smiles.map(lambda  x: np.pad(x, (0,self.maxlen - len(x)), 'constant')) 
        
    def one_hot_encode(self, encoded):
        """
        One-hot encodes tokenised SMILES strings
        
        : encoded (list): contains encoded sequences of each SMILES strings
        
        """
        
        # Correcting translation ex professo for OHE
        array = np.array(encoded) - 1
        
        # Creation of matrix of 0s
        one_hot_array = np.zeros((array.size, len(self.dictionary)), dtype = np.int16)
        
        # Filling the matrix accordingly
        one_hot_array[np.arange(one_hot_array.shape[0]), array.flatten()] = 1
        
        one_hot_array= one_hot_array.reshape((*array.shape, len(self.dictionary)))
        
        return one_hot_array
        
    def _decode(self, dictionary, special_chars_dictionary):
        """
        Converts array of integers back into SMILES strings based on the dictionaries provided.
        
        : dictionary (dict): standard dictionary containing the mapping between tokens and integers
        : special_chars_dictionary (dict): dictionary containing the mapping between special characters tokens and 'standard tokens'
        from the standard dictionary
        WARNING, DECODING IS NOT PROVIDED FOR OHE, YET
        """
        unpadded = self.structures.map(lambda x: np.trim_zeros(x, 'b'))
        
        # Decoding characters
        decoding_dictionary = {v: k for k, v in dictionary.items()}
           
        # (dfernandez): this is the original list comprehension that had been 
        # omitted for exploration/debugging purposes...
        # will recover once the bug is identified
        decoded_chars = [[decoding_dictionary[integer] for integer in mol] for mol in unpadded]
        
        # Decoding special characters
        decoding_special_characters_dictionary = {v: k for k, v in special_chars_dictionary.items()}
        decoded = [[decoding_special_characters_dictionary[integer] if integer in decoding_special_characters_dictionary.keys() else integer for integer in mol] for mol in decoded_chars]
        
        decoded = [''.join(i) for i in decoded]
        decoded = pd.Series(decoded)
        decoded = decoded.str.replace(self.sos,'').str.replace(self.eos,'').str.replace('A','Cl').str.replace('D','Br')
        
        return decoded
           
    def recode(self):
        """
        Recodes the SMILES strings, so that each special character corresponds to a single symbol
        
        """
        
        self._replace_halogen()
        self._sos_and_eos()
            
    def _replace_halogen(self):
        """
        Replaces halogen atoms with single-characters.
        
        """
        
        self.smiles = self.structures.str.replace('Cl','A').str.replace('Br','D')
        
    def recode_special_chars(self):
        """
        Identifies special characters and replaces them with single-characters.
        
        """

        # Recoding        
        self.smiles = self.smiles.map(lambda x: list(filter(None,re.split(self.special_char_regex, x)))) # filtering to avoid '' after split

        recoded = [[self.special_chars_dictionary[char] if char in self.special_chars_dictionary.keys() else char for char in chars] for chars in self.smiles]
       
        recoded = [''.join(i) for i in recoded]
        
        self.smiles = pd.Series(recoded)
        
    def _sos_and_eos(self):
        """
        Adds Start Of Sequence and End Of Sequence characters to a given SMILES string.
        
        """
        
        # Generate SOS and EOS characters from special_chars list generated in recode_special_chars function
        self.smiles = self.sos + self.smiles + self.eos
               
    def add_characters(self, chars):
        """
        Add characters to current vocabulary
        
        : chars (set) : characters to add
        
        """
        
        # Collect characters
        for char in chars:
            self.additional_chars.add(char)
            
        char_list = list(self.additional_chars)
        
        # Sort list of characters
        char_list.sort()
        
        self.chars = char_list
        
        # Generate dictionaries
        self.int2token = dict(enumerate(self.chars, 1))
        self.dictionary = {token:integer for integer,token in self.int2token.items()}
        
    def generate_special_chars_dictionary(self):
        """
        Generates a dictionary mapping special characters to single characters (standard tokens).
        
        """
        
        current_chars = set(self.chars)
        
        # Generation of new characters for special characters
        new_chars = set(string.ascii_letters)
        self.new_chars = sorted(new_chars.difference(current_chars))

        keys = self.new_chars[:len(self.special_chars)]

        # Generation of the dictionary with single-characters and special characters
        self.special_chars_dictionary = dict(zip(self.special_chars, keys))
    
    def identify_special_chars(self):
        
        self.chars = self.smiles.str.cat(sep = ',').replace(',','')
        
        # Identification of special characters
        self.special_chars = sorted(set(re.compile(self.special_char_regex).findall(self.chars)))
    
    def init_dict(self, special_characters = True):
        """
        Takes the series with SMILES to initialise the vocabulary and generate the corresponding dictionaries
        
        """        
        
        # Generate standard dictionary
        self.recode()
        
        if special_characters:
            # Generate special characters dictionary
            self.identify_special_chars()
            self.generate_special_chars_dictionary()
            self.recode_special_chars()
        
        chars = sorted(set(self.smiles.str.cat(sep = ',').replace(',','')))
        self.add_characters(chars)
        
        return self.special_chars_dictionary, self.dictionary
    
    def init_translation(self, standard_dictionary, special_characters_dictionary = None):
        """
        Translates SMILES strings into tokens.
        
        : dictionary (dict): standard dictionary containing the mapping between tokens and integers
        : special_chars_dictionary (dict, optional): dictionary containing the mapping between special characters tokens and 'standard tokens'
        from the standard dictionary
        
        """
        
        self.dictionary = standard_dictionary
        
        if special_characters_dictionary is not None:
            self.special_chars_dictionary = special_characters_dictionary
        
        self.recode()
        
        if special_characters_dictionary is not None:
            self.recode_special_chars()
        
        self.encode()
        self.pad()

        self.translated_DataFrame = pd.DataFrame(self.padded, columns = ['SMILES'])
            
        return self.translated_DataFrame

    
"""
Implements creation and storing of special characters and standard character dictionaries

"""

import glob
import json

class MasterDictionary(object):
    
    def __call__(self, load = False):
        """
        Callable. 
        
        """
        
        # Loading or dumping dictionaries as relevant
        self.load_master_dictionary()
        
        if load:
            return self.master_standard_dictionary, self.master_special_characters_dictionary
        else:
            # If master dictionaries are present, update them
            if self.exist:
                self.update_master_dictionary()
        
    def __init__ (self, new_special_characters_dictionary = None, new_standard_dictionary = None, verbose = 1):
        """
        Initialiser.
        
        : new_special_characters_dictionary (dict, optional): special characters dictionary from new dataset
        : new_standard_dictionary (dict, optional): standard dictionary from new dataset
        : verbose (int): regulates verbosity
        
        """
        
        self.new_special_characters_dictionary = new_special_characters_dictionary
        self.new_standard_dictionary = new_standard_dictionary
        self.verbose = verbose
        
        # Toggle to update dictionaries if master dictionaries have already been created
        self.exist = False
        
    def update_master_dictionary(self): 
        """
        Updates both master dictionaries
        
        """
        
        # Update of special characters must come first
        # so that standard dictionary is updated accordingly
        self.update_special_characters_dictionary(self.new_special_characters_dictionary, self.master_special_characters_dictionary)
        self.update_standard_dictionary(self.new_standard_dictionary, self.master_standard_dictionary)
    
    def load_master_dictionary(self, which = 'both'):
        """
        Loads the master dictionaries if existing or dumps them if they have not been created beforehand.
        
        """
        
        try:
            dictionary = self.load_dictionary(which = which)
            
            if isinstance(dictionary, list):
                self.master_special_characters_dictionary, self.master_standard_dictionary, = dictionary
            else:
                if which == 'standard':
                    self.master_standard_dictionary = dictionary
                elif which == 'special_characters':
                    self.master_special_characters_dictionary = dictionary
            
            self.exist = True
            
        except:
            print('No existing master dictionaries!')
            print('Saving master dictionaries...')
            
            # Dumping dictionaries for the first time
            self.dump_dictionary('master_special_characters_dictionary',self.new_special_characters_dictionary)
            self.dump_dictionary('master_standard_dictionary',self.new_standard_dictionary)           
            
            print('Done!')
            print('Special characters dictionary:\n {}'.format(self.new_special_characters_dictionary))
            print('Standard dictionary:\n {}'.format(self.new_standard_dictionary))
    
    def load_dictionary(self, which = str):
        
        if which == 'both':
            dictionaries = []
            path = '/projects/../../PythonNotebooks/dictionary/'
            
            for filename in glob.glob(os.path.join(path,'*.json')):
                with open(filename,'r') as jsonFile:
                    dictionary = json.load(jsonFile)
                    dictionaries.append(dictionary)
                    
            return dictionaries
                
        else:
            if which == 'standard':
                dictionary = 'master_standard_dictionary.json'
            elif which == 'special_characters':
                dictionary = 'master_special_characters_dictionary.json'
            
            with open('/projects/../../PythonNotebooks/dictionary/'+ str(dictionary),'r') as jsonFile:
                dictionary = json.load(jsonFile) 
            
            return dictionary
        
    @staticmethod
    def dump_dictionary(name, dictionary):
        """
        Stores dictionaries in a directory.
        
        : name (str): name to assign to the dictionary to be stored
        : dictionary (dict): dictionary to store
        
        """
        
        with open('/projects/../../PythonNotebooks/dictionary/' + str(name) +'.json','w') as jsonFile1:
            json.dump(dictionary, jsonFile1, ensure_ascii = False)
    
    def update_special_characters_dictionary(self, new_special_characters_dictionary, special_characters_dictionary):
        """
        Updates the special characters dictionary with potential new special characters coming from new datasets
        
        : new_special_characters_dictionary (dict): dictionary generated from the new dataset
        : special_characters_dictionary (dict): master special characters dictionary, stored special characters dictionary
        : standard_dictionary (dict): standard dictionary, used to avoid repetition of vocabulary
        
        """
        
        # Getting special characters not present in master dictionary
        new_special_characters = set(new_special_characters_dictionary.keys()) - set(special_characters_dictionary.keys())
        
        if len(new_special_characters) != 0:
            
            # In the special characters dictionary, values act as keys
            keys = set(special_characters_dictionary.values())
            
            # Keys to use in new dictionary are based on ASCII letters
            keys_to_use = set(string.punctuation + string.ascii_letters).difference(keys)
            keys_to_use = sorted(keys_to_use.difference(self.master_standard_dictionary.keys()))
                        
            # Generating new keys and mixing them with current keys
            keys_to_use = keys_to_use[:len(new_special_characters)]
            
            new_keys = keys_to_use + list(keys)
            new_keys.sort()
            
            # Collecting new and current special characters
            special_characters = list(special_characters_dictionary.keys()) + list(new_special_characters)
            special_characters.sort()
            
            # Generating new dictionary
            self.master_special_characters_dictionary = dict(zip(special_characters, new_keys))
            
            # Dump the new master special characters dictionary
            self.dump_dictionary('master_special_characters_dictionary',self.master_special_characters_dictionary)
            print('The special characters dictionary has been updated.')
            print('There are {} new special characters'.format(len(new_special_characters)))
            print('These new special characters have been incorporated: \n {}'.format(sorted(new_special_characters)))
            
        else:
            if self.verbose > 0:
                print('No new special characters to add to the master special characters dictionary.')
    
    def update_standard_dictionary(self, new_standard_dictionary, standard_dictionary):
        """
        Updates the standard dictionary with standard dictionaries from new datasets
        
        : new_standard_dictionary (dict): dictionary generated from the new dataset
        : standard_dictionary (dict): master standard dictionary, stored standard dictionary
        
        """
    
        new_characters = set(new_standard_dictionary.keys()) - set(standard_dictionary.keys())
        
        # Including new special characters that might have been added
        # to the master special characters dictionary
        new_special_characters = set(self.master_special_characters_dictionary.values()) - set(standard_dictionary.keys())
        
        new_characters.update(list(new_special_characters))
        
        if len(new_characters) != 0:

            current_standard_dictionary_keys = set(standard_dictionary.keys())
            
            # Adding the new characters
            current_standard_dictionary_keys.update(new_characters)
            
            # Sorting the new characters
            standard_keys = sorted(current_standard_dictionary_keys)
            
            # Generating new dictionary
            token2int = dict(enumerate(standard_keys, 1))
            self.master_standard_dictionary = {token:integer for integer, token in token2int.items()}
            
            # Dump the new master special characters dictionary
            self.dump_dictionary('master_standard_dictionary',self.master_standard_dictionary)
            print('The standard dictionary has been updated.')
            print('There are {} new characters'.format(len(new_characters)))
            print('These new characters have been incorporated: \n {}'.format(sorted(new_characters)))
        
        else:
            if self.verbose > 0:
                print('No new characters to add to the master standard dictionary.')