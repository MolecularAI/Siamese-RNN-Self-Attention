"""
Implements pairing either for classification or regression task. 
For classification, the pairs are distributed as follows:
 25% Active - Active
 25% Inactive - Inactive
 25% Active - Inactive
 25% Inactive - Active

For regression, the pairs do not suffer from class imbalance.
Consequently, the dataset is split into two and the columns merged.

"""
import numpy as np
import os
import pandas as pd
import torch
import random

def set_seed(seed = 42):
    """
    Enables reproducibility.
    
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED']=str(seed)
    
class PairGenerator(object):
    
    def __init__(self, augmentation_factor, fp = False, is_regression = False):
        """
        Constructor.
        
        : augmentation_factor (int): how many times the pairing process will take place
        : fp (bool): whether dealing with fingerprints or SMILES strings. When dealing with fingerprints, 
        the original splitting of the dataset is different.
        : is_regression (bool): signals whether it is a regression or a classification task
        
        """
        
        # Ensures reproducibility
        set_seed()
        
        self.augmentation_factor = augmentation_factor
        self.fp = fp
        self.is_regression = is_regression
        
    def __call__(self, DataFrame):
        """
        Callable.
        
        : DataFrame (pd.DataFrame): DataFrame from which pairs are to be generated
        
        """
        
        if not self.is_regression:
            # Resorting to alternative constructor 
            # to separate DataFrame into two columns
            self.X1, self.X2 = self.from_DataFrame(DataFrame)
            
            # Determines bottleneck
            self.pairs, self.idx = self.determine_pairs(self.X1, self.X2)
            self.random_shuffle()

        else:
            self.DataFrame = DataFrame
        
        return self.prepare_dataset(self.augmentation_factor)
    
    def from_DataFrame(self, DataFrame):
        """
        Alternative constructor. 
        The DataFrame just has to be fed in and class label splitting is performed here.
        
        : DataFrame (pd.DataFrame): DataFrame with 'Activity' column from which 'Molecule' are to be separated
        
        """
        
        # Of note, X1 and X2 are arbitrary names
        # which collect 'active' and 'inactive' compounds, respectively
        if self.fp:
            X1, X2 = DataFrame.iloc[DataFrame[DataFrame['Activity']==1].index,:2048], DataFrame.iloc[DataFrame[DataFrame['Activity']==0].index,:2048]
        else:
            X1, X2 = DataFrame[DataFrame.loc[:,'Activity']==1][['SMILES']], DataFrame[DataFrame.loc[:,'Activity']==0][['SMILES']]
        
        return X1, X2
    
    def make_regression_pairs(self):
        """
        Truncates the DataFrame into two after shuffling (for each augmentation, the shuffling is supposed to generate
        a different ordering in the DataFrame --> this has been checked, but should be revisited!) and concatenates
        the resulting equally-sized columns.
       
        """
        
        # Ensures that there is no bioactivity column
        if 'Activity' in list(self.DataFrame.columns):
            self.DataFrame.drop(columns = 'Activity', inplace = True)
         
        # Shuffle DataFrame
        DataFrame = self.DataFrame.sample(frac = 1)
        
        # Ensyuring that the number of indexes 
        # generates equally-sized columns
        index = np.array(DataFrame.index)
        
        # If the number of indexes is not even
        # then reject the last index #### (brute force approach, but works)
        if not len(index) % 2 == 0:
            index = index[:-1]
                
        index = index.reshape(2,-1)
        
        # Generating columns by splitting the DataFrame into two
        # Selecting the indexes, so that duplicates can be removed
        col1 = pd.Series(index[0])
        col2 = pd.Series(index[1])
        
        df = pd.concat([col1, col2], axis = 1)
        
        return df
    
    def load_regression_compound(self, index_set):
        """
        SMILES strings for regression task are loaded.
        
        : index_set (list): 
        
        """
        
        col1 = self.DataFrame.iloc[index_set[0],:].reset_index(drop = True)
        col2 = self.DataFrame.iloc[index_set[1],:].reset_index(drop = True)
        
        loaded_set = pd.concat([col1, col2], axis = 1)
    
        return loaded_set
    
    def calculate_z_score(self, loaded_set):
        """
        Calculates z-score (~ similarity score) between pairs of regression SMILES strings.
        
        : loaded_set (pd.DataFrame): DataFrame containing pairs of SMILES strings.
        
        """
       
        # Bioactivity difference
        delta_pXC50 = abs(loaded_set.iloc[:,1] - loaded_set.iloc[:,-1])
        
        # z-score generation
        z_score = delta_pXC50/0.43
        
        # Similarity label
        loaded_set['label'] = np.exp(-z_score)
        
        loaded_set.drop(columns = 'pXC50', inplace = True)
        
        return loaded_set
    
    @staticmethod
    def determine_pairs(type1, type2):
        """
        Determines the number of pairs to be generated and which class contains more instances.
        
        : type1 (pd.Series): active compounds
        : type2 (pd.Series): inactive compounds
        
        """
                
        pairs = max(len(type1),len(type2))//2 # Floor division to ensure even number of instances
        
        idx = np.argmax([len(type1), len(type2)]) # Determines which class contains more instances
        
        return pairs, idx
    
    def random_shuffle(self):
        """
        Shuffles indexes of active and inactive compounds to ensure randomness of the process.
        The minority class is sampled with replacement, so that all the instances from the majority class are matched
        
        """

        # np.random.choice is more computationally expensive than random.sample, and thus it is only used
        # where sampling with replacement is needed
        
        if self.idx:
            # Inactive compounds are the majority
            self.similar_active = np.random.choice(len(self.X1), self.pairs*2)
            self.similar_inactive = random.sample(range(len(self.X2)), self.pairs*2)
        else:
            # Active compounds are the majority
            self.similar_active = random.sample(range(len(self.X1)), self.pairs*2)
            self.similar_inactive = np.random.choice(len(self.X2), self.pairs*2)
               
    def load_index(self, index_order):
        """
        Creates two columns containing the indexes of the compounds to be paired up. 
        
        : index_order (pd.Series): indexes corresponding to compounds
        
        """
        
        # Even split from index_order into 2, so that two equally-sized columns are generated
        col1 = pd.Series(index_order[:self.pairs])
        col2 = pd.Series(index_order[self.pairs:2*self.pairs])
        
        return col1, col2
              
    def make_similar_pairs(self, active = False):
        """
        Generates similar pairs (e.g. active-active or inactive-inactive).
        
        : active (bool): indicates whether the pairs to be prepared consist of active or inactive compounds
        
        """
        
        # Shuffling of indexes without replacement
        self.random_shuffle()
        
        if active:
            shuffled_indexes = self.similar_active
        else:
            shuffled_indexes = self.similar_inactive
        
        # Loading of indexes, so that they are separated into two columns
        col1, col2 = self.load_index(shuffled_indexes)
        
        # Remove duplicate columns
        loaded_columns = [col1, col2]
        lst = pd.concat(loaded_columns, axis = 1).to_numpy().tolist()
        
        deduplicated = set(map(tuple, lst))
        
        # Back to columns
        loaded_columns = [[array[i] for array in list(deduplicated)] for i in range(len(loaded_columns))]
        col1, col2 = [pd.Series(col) for col in loaded_columns]
        
        # Class label column (similarity = 1)
        col3 = pd.Series(np.ones((len(col1))))
        
        # Column concatenation
        pairs = [col1, col2, col3]
        df = pd.concat(pairs, axis = 1)
        
        return df
    
    def make_dissimilar_pairs(self, active_inactive = False):
        """
        generates dissimilar pairs (e.g. active-inactive or inactive-active).
        
        : active_inactive (bool): indicates whether the pairs to be prepared are active-inactive or inactive-active pairs
        
        """
        
        reshuffled_actives = random.sample(list(self.similar_active), len(self.similar_active))
        reshuffled_inactives = random.sample(list(self.similar_inactive), len(self.similar_inactive))
        
        ### new addition
        merged = pd.concat([pd.Series(reshuffled_actives), pd.Series(reshuffled_inactives)], axis = 1)
        merged.drop_duplicates(inplace = True)
        
        reshuffled_actives, reshuffled_inactives = merged.iloc[:,0], merged.iloc[:,1]
        
        # Loading of active - inactive pairs or vice versa
        col1_a_ai, col2_a_ia = self.load_index(reshuffled_actives)
        col2_i_ai, col1_i_ia = self.load_index(reshuffled_inactives)
        
        # Class label column (similarity = 0)
        col3 = pd.Series(np.zeros((self.pairs)))
        
        # Matching columns depending on boolean, so that pairs are active-inactive or inactive-active
        if active_inactive:
            pairs = [col1_a_ai, col2_i_ai, col3]
        else:
            pairs = [col1_i_ia, col2_a_ia, col3]
            
        df = pd.concat(pairs, axis = 1)
        
        return df
    
    def load_compound(self, DataFrame):
        """
        Loads the compounds given the index ordering and generates a full set of pairs
        with the same pairs (AA, II, IA, AI).
        
        : DataFrame (pd.DataFrame): contains indexes from which compounds are to be loaded
        
        """
        
        if self.similar:
            # Establishing which compounds are to be loaded at once (actives or inactives)
            if self.activity:
                X = self.X1
            else:
                X = self.X2
            
            # Compounds are loaded
            col1 = X.iloc[DataFrame.iloc[:,0],:].reset_index(drop = True)
            col2 = X.iloc[DataFrame.iloc[:,1],:].reset_index(drop = True)
            
            # The length of col2 is taken as a reference
            col3 = DataFrame.iloc[:len(col2),2].reset_index(drop = True)
            
        else:
            #Same process as above, but for dissimilar compounds
            # Note that X1 and X2 are used in both cases, unlike for similar compounds
            if self.activity:
                col1 = self.X1.iloc[DataFrame.iloc[:,0],:].reset_index(drop = True)
                col2 = self.X2.iloc[DataFrame.iloc[:,1],:].reset_index(drop = True)
            else:
                col1 = self.X2.iloc[DataFrame.iloc[:,0],:].reset_index(drop = True)
                col2 = self.X1.iloc[DataFrame.iloc[:,1],:].reset_index(drop = True)
                
            col3 = DataFrame.iloc[:len(col2),2].reset_index(drop = True)
        
        loaded_set = pd.concat([col1, col2, col3], ignore_index = True, axis = 1)
        
        return loaded_set
    
    def augment_pairs(self, activity, similar):
        """
        Repeats the pairing process to generate more pairs and ensures that there are no repeated pairs.
        
        : boolean (bool): indicates the nature of the pairs to be generated in terms of bioactivity 
        : similar (bool): indicates whether the pairs of compounds to be generated are similar or dissimilar
        
        """
        
        sets = []
        
        self.similar = similar
        self.activity = activity
        
        for i in range(1, self.augmentation_factor + 1):
            if self.similar:
                # When active = True: generate active-active pairs
                # When active = False: generate inactive-inactive pairs
                new_set = self.make_similar_pairs(active = self.activity) 
            else:
                # When active_inactive = True: generate active-inactive pairs,
                # When active_inactive = False: generate inactive-active pairs 
                new_set = self.make_dissimilar_pairs(active_inactive = self.activity) 
            sets.append(new_set)

        df = pd.concat(sets)
        
        # Ensuring that no duplicates are present
        df.drop_duplicates(inplace = True)

        return df
    
    def prepare_dataset(self, augmentation_factor):
        """
        Prepares the whole dataset to be output.
        
        : augmentation_factor (int): indicates how many times the process of pairing up has to be performed
        
        """
        
        self.augmentation_factor = augmentation_factor
        
        if self.is_regression:
            df = self.prepare_regression_dataset() 
        else:
            df = self.prepare_binary_dataset()
            
        return df
    
    def prepare_regression_dataset(self):
        """
        Prepares a paired dataset for the regression task. Unlike in the classification task, in this case pairing
        occurs by splitting the dataset in two equally-sized columns which are merged thereafter. Indeed, no class imbalance
        problems occur due to the nature of regression problems.
        
        """
        
        pairs = []
        
        # Loop over to go through the pair generation process
        # as many times as required by the augmentation_factor
        for i in range(1, self.augmentation_factor + 1):
            regression_set = self.make_regression_pairs()
            pairs.append(regression_set)
        
        # Bring datasets together and drop duplicates
        augmented_regression_set = pd.concat(pairs).reset_index(drop = True)
        augmented_regression_set.drop_duplicates(inplace = True)
        
        # Assign SMILES according to index
        loaded_set = self.load_regression_compound(augmented_regression_set)
        
        # Estimates z-score
        z_set = self.calculate_z_score(loaded_set)
        
        return z_set
    
    def prepare_binary_dataset(self):
        """
        Prepares paired dataset for classification task. Consequently, it pairs compounds according to the binary similarity
        label.
        
        """
        
        similar = []
        dissimilar = []
        
        boolean_list = [[False, True],[False, True]]
        
        for boolean1 in boolean_list:
            for similarity in boolean1:
                
                # Augmented set with indexes of all the compounds to be loaded
                augmented_set = self.augment_pairs(boolean1, similar = similarity) 
                
                # Set with the pairs of compounds corresponding to the provided indexes in augmented_set
                loaded_set = self.load_compound(augmented_set)
                
                # Appending the sets as relevant
                if similarity:
                    similar.append(loaded_set)
                else: 
                    dissimilar.append(loaded_set)
    
        # Concatenation of similar and dissimilar sets
        full_similar = pd.concat(similar).reset_index(drop = True)
        full_dissimilar = pd.concat(dissimilar).reset_index(drop = True)
        
        # Generates fully balanced dataset
        df = self.adjust_instances(full_similar, full_dissimilar)
        
        # Pair shuffling
        df = df.sample(frac = 1).reset_index(drop = True)
        
        df.set_axis([*df.columns[:-1], 'label'], axis = 1, inplace = True)
        
        return df
    
    @staticmethod
    def adjust_instances(similar, dissimilar):
        """
        Adjusts the number of instances for similar and dissimilar sets for a classification task, so that 
        both sets have the same number of pairs, thereby yielding a balanced dataset.
        
        : similar (list): contains  similar pairs
        : dissimilar (list): contains dissimilar pairs
        
        """
        
        full_set = [similar, dissimilar]
        
        # Adjustment to get a balanced dataset 
        if len(similar) != len(dissimilar):
            remove_n = max(map(len,full_set)) - min(map(len,full_set)) # Indexes to drop
            subset = np.argmax(list(map(len,full_set))) # identification of largest subset
            drop_indices = np.random.choice(full_set[subset].index, remove_n, replace=False) # random indexes to drop
            full_set[subset] = full_set[subset].drop(drop_indices) # adjusting largest
        
        df = pd.concat(full_set, axis = 0).reset_index(drop = True)
        
        return df
    
def pair(DataFrame, augmentation_factor, fp = False, is_regression = False):
    """
    Helper function which generates pairs. 
    (It is used in baseline models and SiameseMLP classes).
        
    : DataFrame (pd.DataFrame): DataFrame to generate pairs from.
    : augmentation_factor (int): 
    : fp (bool):   
    : is_regression (bool):
    
    """
               
    pair = PairGenerator(augmentation_factor, fp = fp, is_regression = is_regression)
    DataFrame = pair(DataFrame)       
        
    return DataFrame