import pandas as pd
import sys
sys.path.append('/projects/../../PythonNotebooks/model/')
from translator import *

def main_decoder(data):
    """
    Decodes tokenised SMILES strings into the original SMILES string (removing padding and post-processing steps).
    
    : data (pd.DataFrame): contains SMILES strings
    
    """
    
    master_dictionary = MasterDictionary()
    standard_dictionary, special_characters_dictionary = master_dictionary(load = True)

    t = Translator(data)
        
    decoded = t._decode(standard_dictionary, special_characters_dictionary)
    
    return decoded