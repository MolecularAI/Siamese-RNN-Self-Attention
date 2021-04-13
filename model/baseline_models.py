"""
Implements random forest and support vector machine as baseline models for benchmarking against the Siamese Neural Network

"""

import matplotlib.pyplot as plt # Data Visualisation
import numpy as np # Linear Algebra
import pandas as pd # Data handling
import pickle # Saving files
import seaborn as sns # Data visualisation
import sys

import rdkit

from scipy import stats
from sklearn.metrics import mean_squared_error
from statistics import mean # stats

# Chemistry packages

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

# Machine learning models, metrics and strategies
from sklearn import svm
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from sklearn.utils import shuffle

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from fingerprints import *
from main_translator import *
from pair_generator import *
from splitter import *

class BaselineModel(object):
    def __init__(self, DataFrame, augmentation_factor, model, dataset_name = None, fp = False, is_regression = False):
        """
        Initialiser.
        
        : DataFrame (pd.DataFrame): DataFrame with the encoded compounds
        : augmentation factor (int): augmentation factor when performing pairing
        : model (str): specifies whether to train a random forest (RF) or support vector machine (SVM)    
        : dataset_name (str): dataset whose compounds are being classified
        : fp (bool): signals whether data is ECFP or not
        : is_regression (bool): signals whether the task is classification or regression
        
        """
        
        # Ensuring reproducibility
        set_seed()
        
        self.smi = DataFrame['SMILES']
        self.activity = DataFrame['Activity']
        self.pXC50 = DataFrame['pXC50']
        self.augmentation_factor = augmentation_factor
        self.model = model.lower()
        self.dataset_name = dataset_name
        self.fp = fp
        self.is_regression = is_regression
        
        # SMILES strings translation
        if not self.fp:    
            self.translate()

        # Model selection
        if type(model) is not str:
            raise TypeError('The model should be in string format')
        
        if self.is_regression:
            if self.model == 'rf':
                self.clf = RandomForestRegressor(n_estimators = 100)
            elif self.model == 'svm':
                self.clf = svm.SVR()
        else:
            if self.model == 'rf':
                self.clf = RandomForestClassifier(n_estimators = 100)
            elif self.model == 'svm':
                self.clf = svm.SVC()
        
    def translate(self):
        """
        Translates the given structures from the DataFrame into tokens.
        
        """
        
        self.smi = translate_sets(self.smi, verbose = 0)
    
    def kfold_cross_validate(self, k_folds):
        """
        Performs k-fold cross-validation for a given unpaired dataset of compounds
        
        : k_folds (int): number of folds to generate
        
        """
        
        metrics = []
        
        x = self.smi
        
        if self.is_regression:
            y = self.pXC50
            self._kf = KFold(n_splits = k_folds, random_state = 42)
        else:
            y = self.activity
            self._kf = StratifiedKFold(n_splits = k_folds, random_state = 42)
        
        for k, (k_train, k_val) in enumerate(self._kf.split(x,y)):
            
            DataFrame = pd.concat([x,y], axis = 1)
            
            # Train and validation k-sets
            k_train = DataFrame.iloc[k_train].reset_index(drop = True)
            k_val = DataFrame.iloc[k_val,:].reset_index(drop = True)
            
            # Selection of train set
            if self.fp:
                # Calculate circular FPs
                X_train, y_train = calculate_circular_fp(k_train['SMILES']), k_train.iloc[:,-1]
                X_val, y_val = calculate_circular_fp(k_val['SMILES']), k_val.iloc[:,-1]
                
                # Regeneration of k-set
                k_val = pd.concat([X_val, y_val], axis = 1)
                
                # Pairing k-set
                val = pair(k_val, self.augmentation_factor, fp = self.fp, is_regression = self.is_regression)
                
                X_val, y_val = val.iloc[:,:-1], val.iloc[:,-1]
                
            else:
                X_train = self.disaggregate_tokens(k_train, 'SMILES')
                y_train = k_train.iloc[:,-1]

                # Validation set generation and pairing
                val = pair(k_val, self.augmentation_factor, fp = self.fp, is_regression = self.is_regression)

                # Selection of validation set
                X_val = self.disaggregate_pairs(val)
                y_val = val.iloc[:,-1]
                
            # Prediction with the classifier
            y_pred = self.classify(X_train, y_train, X_val, y_val)
            
            # Reporting of metrics
            k_metrics = self.calculate_metrics(y_val, y_pred)
            
            metrics.append(k_metrics)
            
            if self.is_regression:
                print("Fold {:2d}, Pearson's r: {:.5f}, RMSE: {:.5f}" .format(k+1, k_metrics[0], k_metrics[1]))
            else:
                print('Fold %2d, accuracy: %.1f%%, F1-score: %.1f%%, MCC: %.2f, recall: %.1f%%,\n  precision: %.1f%%, false negative rate: %.1f%%, false positive rate: %.1f%%' % (k+1, k_metrics[0], k_metrics[1], k_metrics[2], k_metrics[3], k_metrics[4], k_metrics[5], k_metrics[6]))
        
        if self.is_regression:
            task = 'regression'
        else:
            task = 'classification'
            
        if self.fp:
            model_type = 'fp'
        else:
            model_type = ''
        
        path = '/projects/../../PythonNotebooks/cv_statistics/' + str(k_folds) + '_fold_' + str(self.dataset_name) + '_' + str(self.model) + '_' + task + '_' + str(model_type) + '_' + '_metrics.pkl'
        
        with open(path,'wb') as f:
            pickle.dump(metrics, f)
        
        if self.is_regression:
            metric_list = list(zip(*metrics))
            
            print(100 * '-')
            print("Pearson's r:" + str(mean_confidence_interval(metric_list[0])))
            print('RMSE:'+ str(mean_confidence_interval(metric_list[1])))
        else:
            metric_list = list(zip(*metrics))[:7]

            print(100 * '-')
            print('Accuracy:'+ str(mean_confidence_interval(metric_list[0])))
            print('F1-score:'+ str(mean_confidence_interval(metric_list[1])))
            print("Matthew's Correlation Coefficient:"+ str(mean_confidence_interval(metric_list[2])))
            print('Recall:'+ str(mean_confidence_interval(metric_list[3])))
            print('Precision:'+ str(mean_confidence_interval(metric_list[4])))
            print('False negative rate:'+ str(mean_confidence_interval(metric_list[5])))
            print('False positive rate:'+ str(mean_confidence_interval(metric_list[6])))

            self.plot_confusion_matrix(list(zip(*metrics))[-1])
       
    @staticmethod
    def disaggregate_tokens(to_disaggregate, column = None):
        """
        Converts each token in a DataFrame row into a column.
        
        : to_disaggregate (pd.DataFrame): single-column DataFrame consisting of rows which contain arrays with encoded
        compounds
        : column (string): column   
        
        """
        
        if column is not None:
            tokens = pd.DataFrame(to_disaggregate.loc[:,column].tolist())
        else:
            tokens = pd.DataFrame(to_disaggregate.tolist())
        
        return tokens
    
    def disaggregate_pairs(self, paired_DataFrame):
        """
        Disaggregates tokens of each set of pairs.
        
        : paired_DataFrame (pd.DataFrame): DataFrame with pairs and similarity labels
        
        """
        
        # Generate one-column DataFrame with pairs to disaggregate each compound
        to_disaggregate = pd.concat([paired_DataFrame.iloc[:,0], paired_DataFrame.iloc[:,1]], axis = 0).reset_index(drop = True)
        
        # Disaggregation of tokens
        disaggregated = self.disaggregate_tokens(to_disaggregate)
        
        # Recover the paired DataFrame
        pair1 = disaggregated.iloc[len(to_disaggregate)//2:,:].reset_index(drop = True)
        pair2 = disaggregated.iloc[:len(to_disaggregate)//2,:].reset_index(drop = True)
        
        pairs = pd.concat([pair1, pair2], axis = 1)
        
        return pairs
    
    def classify(self, X_train, y_train, X_val, y_val):
        """
        Performs prediction based on Random Forest classification
        
        : X_train (pd.DataFrame):
        : y_train (pd.DataFrame):
        : X_val (pd.DataFrame):
        : y_val (pd.DataFrame):
        
        """
        
        # Determine column at which DataFrame is sliced
        if self.fp:
            col = 2048
        else:
            col = 150
        
        # Separate pairs
        X_val_pair1 = X_val.iloc[:,:col]
        X_val_pair2 = X_val.iloc[:,col:]
        
        # Fit the Random Forest to training data
        self.clf.fit(X_train, y_train)

        # Validation bioactivity predictions
        y_pred_pair1 = self.clf.predict(X_val_pair1)
        y_pred_pair2 = self.clf.predict(X_val_pair2)
        
        if self.is_regression:
            # Bioactivity difference
            delta_pXC50 = abs(y_pred_pair1 - y_pred_pair2)

            # z-score generation
            z_score = delta_pXC50/0.43
        
            y_pred = np.exp(-z_score)
            
        else:
            # Combine predictions to yield similarity prediction
            y_pred = y_pred_pair1 == y_pred_pair2
            y_pred = y_pred.astype(int)
        
        return y_pred
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculates metrics for a given model.
        
        : y_true (pd.Series): ground truth class labels.
        : y_pred (pd.Series): predicted class labels.
        
        """
        
        if self.is_regression:
            r, p_value = stats.pearsonr(y_true,y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            return [r, rmse]
        else:
            TP, FN, FP, TN, cm = self.generate_confusion_matrix(y_true, y_pred)

            accuracy = accuracy_score(y_true, y_pred)*100
            recall = recall_score(y_true, y_pred)*100
            precision = precision_score(y_true, y_pred)*100
            f1 = 2*(((precision*recall)/(precision + recall)))
            MCC = matthews_corrcoef(y_true, y_pred)

            FNR = FN/(FN+TP)*100 # False Negative Ratio or miss-rate
            FPR = FP/(FP+TN)*100 # False Positive Rate or fall-out

            return [accuracy, f1, MCC, recall, precision, FNR, FPR, cm]
    
    @staticmethod
    def generate_confusion_matrix(y_true, y_pred):
        """
        Generates the confusion matrix for the given labels and retrieves FP, 
        
        : y_true (pd.Series): ground truth class labels.
        : y_pred (pd.Series): predicted class labels.
        
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Retrieve test results
        TP, FN, FP, TN = [], [], [], []
        
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        TP = cm[1,1]
        
        return TP, FN, FP, TN, cm
    
    def plot_confusion_matrix(self, cm):
        """
        Plots the sum of the confusion matrixes generated at each fold. 
        
        : cm (np.array): array of confusion matrixes (n confusion matrixes for n folds)
        
        """
        
        plt.figure(figsize = (8.5,6.5))
        ax = plt.subplot()
        
        sns.set(font_scale = 2)
        sns.heatmap(sum(cm), annot = True, ax = ax, cmap = 'Greens', fmt = 'g') # annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels', size = 15)
        ax.set_ylabel('True labels', size = 15)
        ax.set_title('Confusion Matrix', size = 20)
        ax.xaxis.set_ticklabels(['dissimilar', 'similar'])
        ax.yaxis.set_ticklabels(['dissimilar', 'similar'])
        plt.savefig('/projects/../../PythonNotebooks/confusion_matrixes/' + str(self.dataset_name) + '_' + str(self.model.upper()) + '.svg', format = 'svg')
        
def pair(DataFrame, augmentation_factor, fp = False, is_regression = False):
    """
    Helper function which generates pairs. 
        
    : DataFrame (pd.DataFrame): DataFrame to generate pairs from.
    : augmentation_factor (int): number of times pairing process is repeated
    : fp (bool): signals whether data is ECFP or not
    : is_regression (bool): signals whether the task is classification or regression
    
    """
               
    pair = PairGenerator(augmentation_factor, fp = fp, is_regression = is_regression)
    DataFrame = pair(DataFrame)       
        
    return DataFrame