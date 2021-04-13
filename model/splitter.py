"""
Implementation of different splitting techniques for biased datasets.

"""

import math
import numpy as np
import pandas as pd
import sys
sys.path.append('/projects/../../PythonNotebooks/model/')
import itertools

from itertools import accumulate

from helper import *

class Splitter():
    
    def __init__(self, DataFrame, train_percentage, test_percentage, clusters = None):
        """
        Initialiser.
        
        : train_percentage (int): percentage of compounds to be assigned to the training set
        : val_percentage (int) : percentage of compounds to be assigne to the validation set
        : clusters (list): if splitting is based on clusters, it contains the indexes of the clusters (each cluster corresponds to
        one tuple of indexes) generated with Butina clustering algorithm from RDKit
        
        """      

        # To make it reproducible
        set_seed()
        
        self.DataFrame = DataFrame
        
        self.train_percentage = train_percentage
        self.test_percentage = test_percentage
        self.val_percentage = (100 - self.train_percentage - self.test_percentage)
    
        self.clusters = clusters

    @staticmethod
    def _remove_clusters(clusters, size = 2):
        """
        Removes clusters of below a given number of compounds.
        
        : clusters (list): contains the indexes of the clusters
        : size (int): maximum number of compounds in a cluster to be removed
        
        """
        
        # Filtering clusters
        clusters = [cluster for cluster in clusters if len(cluster)>size]  
        
        return clusters
    
    def split_cluster(self, leave_one_out = False, left_out_percentage = 0.1, sort = False):
        """
        Split compounds of clusters to train, validation and test sets iterativerly according to the specified proportions

        : leave_one_out (bool): toggle to set the method to the conventional splitting or leave-one out strategy 
        : left_out_percentage (float): percentage of clusters to be left out
        : sort (bool): toggle to set whether the clusters should be sorted in ascending order by number of compounds or not 
        
        """
        
        train_points, test_points, val_points = [], [], []
        
        # Removal of doublets
        self.clusters = self._remove_clusters(self.clusters)      
        
        # Ensuring clusters have been input
        self.assertion_error(self.clusters)
        
        if sort:
            self.clusters.sort(key = len)
        
        list_sizes = list(map(len, self.clusters))
        accumulated_sizes = list(accumulate(list_sizes))
           
        if leave_one_out:
            n_compounds = sum(list_sizes)
            n_compounds_left_out = math.floor(n_compounds*left_out_percentage)
            
            # Calculates closest cluster to the leave_one_out percentage set
            close = min(accumulated_sizes, key = lambda x: abs(x - n_compounds_left_out))
            
            index = accumulated_sizes.index(close)
            sizes = list_sizes[:index+1]
            left_out_clusters = self.clusters[:index+1]

            left_out_points = [point for cluster in left_out_clusters for point in cluster]
            self.clusters = self.clusters[index+1:]               
        
        # Proportional distribution of points (partition)
        for cluster in self.clusters:
            cluster_len = len(cluster)

            val_len = max(1,math.floor((self.val_percentage/100)*cluster_len))
            test_len = max(1, math.floor((self.test_percentage/100)*cluster_len))
            train_len = cluster_len - val_len - test_len         
            
            val_points += cluster[:val_len]
            test_points += cluster[val_len:val_len+test_len]
            train_points += cluster[val_len+test_len:val_len+test_len+train_len]

        if leave_one_out:
            val_points += left_out_points
        
        train_points, test_points, val_points = list(map(self.transfer_points,[train_points, test_points, val_points]))
        
        return train_points, test_points, val_points
    
    def distribute_cluster(self):
        """
        Assign clusters to train, validation and test sets iterativerly according to a specified split.
        Adapted from Panagiotis

        """
        
        train_points, test_points, val_points = [], [], []
        
        self.assertion_error(self.clusters)
        
        mod = (self.train_percentage/10)-1
        mod2 = mod + (self.test_percentage/10)
        mod3 =  mod2 + (self.val_percentage/10)

        self.clusters.sort(key = len, reverse = True)

        for idx, cluster in enumerate(self.clusters):
            if np.mod(idx, 10) <= mod:
                train_points.append(cluster)

            elif np.mod(idx,10) <= mod2:
                test_points.append(cluster)    

            elif np.mod(idx,10) <= mod3:
                val_points.append(cluster)

        train_points, test_points, val_points = list(map(self.tuples_to_list,[train_points, test_points, val_points]))
        
        train_points, test_points, val_points = list(map(self.transfer_points,[train_points, test_points, val_points]))

        return train_points, test_points, val_points
    
    @staticmethod
    def assertion_error(input_clusters):
        try:
            assert input_clusters is not None, 'Missing input clusters'
        except AssertionError as msg:
            sys.exit(msg)
    
    @staticmethod
    def tuples_to_list(list_to_convert):
        """
        Post-processing function
        
        : list_to_convert (list):
        
        """

        return list(itertools.chain(*list_to_convert))
    
    def transfer_points(self, points):
        """
        Post-processing function.
        
        : points ():
        
        """

        points = self.DataFrame.iloc[points].reset_index(drop = True)
        
        return points

    def random_split(self, is_regression = False):
        """
        Splits the DataFrame into train, test and validation set at random.
        
        : is_regression (bool): regulates whether it is a regression or a classification task. 

        """
    
        if is_regression:
            train_set, test_set, val_set = self.split()
        else:
            train_set, test_set, val_set = self.stratified_split()
        
        return train_set, test_set, val_set
    
    def split(self):
        """
        Random split. Performed in regression tasks.
        
        """
        
        df = self.DataFrame.sample(frac = 1, random_state = 42).reset_index(drop = True)
        
        idx_train, idx_test = self.index_selector(df)
        train_set, test_set, val_set = self.distribute_indexes(df, idx_train, idx_test)
        
        return train_set, test_set, val_set
    
    def stratified_split(self):
        """
        Stratified random shuffled split. Performed in classification task to ensure equal distribution of
        classes across sets.
        
        """
        
        # Separation of active and inactive compounds
        df_active = self.DataFrame[self.DataFrame.loc[:,'Activity']==1].reset_index(drop = True)
        df_inactive = self.DataFrame[self.DataFrame.loc[:,'Activity']==0].reset_index(drop = True)

        df_active = df_active.sample(frac = 1, random_state = 42).reset_index(drop = True)
        df_inactive = df_inactive.sample(frac = 1, random_state = 42).reset_index(drop = True)

        # Indexes for active compounds
        idx_train_active, idx_test_active = self.index_selector(df_active)
        
        # Active compounds are distributed accordingly
        X_train_positive, X_test_positive, X_val_positive = self.distribute_indexes(df_active, idx_train_active, idx_test_active)
        
        # Indexes for inactive compounds
        idx_train_inactive, idx_test_inactive = self.index_selector(df_inactive)

        # Inactive compounds are distributed as well
        X_train_negative, X_test_negative, X_val_negative = self.distribute_indexes(df_inactive, idx_train_inactive, idx_test_inactive)
        
        train_set, test_set, val_set = list(map(pd.concat,[[X_train_positive, X_train_negative],[X_test_positive, X_test_negative],[X_val_positive, X_val_negative]]))
        
        train_set.reset_index(drop = True, inplace = True)
        test_set.reset_index(drop = True, inplace = True)
        val_set.reset_index(drop = True, inplace = True)
        
        return train_set, test_set, val_set
    
    def index_selector(self, DataFrame):
        """
        Helper function. Selects indices for train and test sets according to given percentages.
        
        : DataFrame (pd.DataFrame):
        
        """
        
        idx_train = math.ceil(len(DataFrame) * (self.train_percentage/100))
        idx_test = math.ceil(len(DataFrame) * (self.test_percentage/100))
        
        return idx_train, idx_test
    
    @staticmethod
    def distribute_indexes(DataFrame, idx_train, idx_test):
        """
        Helper function. Slices original DataFrame into train, test and validation indices.
        
        : DataFrame (pd.DataFrame): contains all data instances.
        : idx_train ():
        : idx_test ():
        
        """
        
        X_train = DataFrame.loc[:idx_train,:].reset_index(drop = True)
        X_test = DataFrame.loc[idx_train:idx_train + idx_test,:].reset_index(drop = True)
        X_val = DataFrame.loc[idx_train + idx_test:,:].reset_index(drop = True)
     
        return X_train, X_test, X_val

    def mc_cross_validate(self, k_folds, splitting_strategy = 'random', clusters = None):
        """
        Performs Monte-Carlo cross validation (MCCV) task given a DataFrame according to the set strategy.
        
        : k_folds (int): number of folds in which the DataFrame is to be split
        : splitting_strategy (str): algorithm to use when splitting
        : clusters (list): clustered compounds contained in a list of tuples
        
        """
        
        splitting_strategy = splitting_strategy.lower()
               
        if 'cluster' in splitting_strategy:
            self.assertion_error(self.clusters)
        
        sets_kfolds = [] 
        
        self.generate_seeds(k_folds)
        
        for fold in range(k_folds):
            self.random_state = self.seeds[fold]
            
            if splitting_strategy == 'random':
                train, test, val = self.random_split()
            elif splitting_strategy == 'split_clusters':
                train, test, val = self.split_cluster()
            elif splitting_strategy == 'distribute_clusters':
                train, test, val = self.distribute_cluster()
            elif splitting_strategy == 'loo_clusters':
                train, test, val = self.split_cluster(leave_one_out = True)
            
            sets = [train, test]
            
            sets_kfolds.append(sets)
            
        return sets_kfolds

    def generate_seeds(self, k_folds):
       
        self.seeds = random.sample(list(np.arange(0,100)), k_folds)  
    
    
        
"""
Helper function to sample from a dataset

"""

def sample(DataFrame, percentage):
    """
    Samples without replacement a percentage of actives and inactives from a given DataFrame
    
    : DataFrame (pd.DataFrame): DataFrame containing active and inactive compounds
    : percentage (int): percentage of compounds from a given class to be sampled
    
    """
    
    import random
    
    active, inactive = DataFrame[DataFrame['Activity']==1], DataFrame[DataFrame['Activity']==0]
    
    active_sample, inactive_sample = [random.sample(list(i.index), int(len(i)*(percentage/100))) for i in [active, inactive]]
    sample = active_sample + inactive_sample
    
    sampled_df = DataFrame.iloc[sample,:].reset_index(drop = True)
    
    return sampled_df