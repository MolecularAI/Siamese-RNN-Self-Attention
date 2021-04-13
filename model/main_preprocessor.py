import numpy as np
import pandas as pd
import sys
sys.path.append('/projects/../../PythonNotebooks/model/')
from preprocessor import *

def preprocess_data(dataset, name = None, threshold = None, set_threshold = True, standardise = True, process = True):
    
    p = PreProcess(dataset, threshold, set_threshold, standardise, process = process)
    
    if process:
        print('Preprocessing of ' + name + ' dataset starting...')
    else:
        print('Preprocessing of the dataset starting...')
    print('\n')
    
    if set_threshold:
        try:
            assert threshold is not None and isinstance(threshold, float) or isinstance(threshold, int), 'Please enter a threshold value'
        except AssertionError as msg:
            sys.exit(msg)
    
    df = p.preprocess()
    
    print('\n')
    print('Preprocessing finished!')
    print('\n')
    print('Saving data...')
    
    p.save_data(name)
    
    print('Done!')
    
    return df