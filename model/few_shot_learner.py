import json # File format
import matplotlib.pyplot as plt # Visualisation
import numpy as np # Linear Algebra
import pandas as pd # Data Wrangling
import random # Random sampling
import seaborn as sns # Visualisations

import sys

# Deep learning
import torch
import torch.nn.functional as F

# Machine learning metrics
from sklearn.metrics import average_precision_score, classification_report, cohen_kappa_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix, matthews_corrcoef

# Descriptive statistics
from scipy import stats
from statistics import mean, mode

# Progress bar
from tqdm import tqdm_notebook as tqdm

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from few_shot_dataset import *
from helper import *

class FewShotLearner(object):
    def __init__ (self, support_set, query_set, model, model2 = None, threshold = None, threshold2 = None, 
                  strategy = 'few-shot', neighbours = None, replacement = True, balanced = True, randomise = False):
        """
        Initialiser.
        
        : support_set (pd.Series): set of compounds whose activity is known
        : query_set (pd.Series): set of comounds whose activity is to be inferred
        : model (class): SNN trained with threshold to output similarity scores
        : model2 (class): SNN trained with threshold2 to output similarity scores
        : threshold (int, optional): first threshold
        : threshold2 (int, optional): second threshold 
        : strategy (str): one-, few-shot or k-nearest neighbours strategy
        : replacement (bool): support set created by sampling with or without replacement
        : balanced (bool): support set will be balanced or not in terms of bioactivity
        : randomise (bool): randomiser flag
        
        """
              
        self.support_set = support_set
        self.query_set = query_set
        self.replacement = replacement
        self.balanced = balanced
        self.randomise = randomise
        
        self.model = model
        self.model2 = model2
        
        self.threshold = threshold
        self.threshold2 = threshold2
        
        self.strategy = strategy
        self.neighbours = neighbours

        if strategy not in ['few-shot','one-shot','knn','regression']:
            raise ValueError('Strategy can only be "few-shot", "one-shot", "knn" or "regression".')

        if strategy in ['few-shot','one-shot']:
            if self.replacement == True:
                raise ValueError('Sampling with replacement should be False if one-shot learning or knn is used.')
            # Should be balanced anywhere, since otherwise 
            # it would not be an equally evidence-based decision
            # (one could find more instances from one category than the other)
            if self.balanced == False:
                raise ValueError('Support set must be balanced is one-shot learning or knn is used.')
        elif strategy == 'knn':
            if self.neighbours is None:
                raise ValueError('Please, introduce an odd number of neighbours')
            if self.neighbours % 2 == 0:
                raise ValueError('The number of neighbours should be odd')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, N, k, dataset_name, multiclass = False):
        """
        Performs an N-shot k-way task.
        
        : N (int): number of compounds sampled to generate support set
        : k (int): number of times to repeat the learning process given N
        : dataset_name (str): name to assign to the file to save metrics
        : multiclass (bool): if running a classification problem, signals whether
        it is a binary problem or a categorical problem
        
        """
            
        self.dataset_name = dataset_name
        
        if multiclass:
            if None in [self.model, self.model2, self.threshold, self.threshold2]:
                raise ValueError('Please, instantiate the class with trained models for each threshold and specify the second threshold')
            if self.threshold > self.threshold2:
                raise ValueError('Please, introduce thresholds in ascending order')
            if self.strategy == 'regression':
                raise TypeError('Categorical classification is not compatible with regression.')
        
        self.multiclass = multiclass
        self.N = N
        self.k = k
        idxs = self.query_set.index
        
        predicted_class = []
        probability_class = []
        
        if self.k % 2 == 0:
            raise ValueError('k must be an odd number')
                
        for index in idxs:
            print('Compound {}/{}'.format(index + 1, len(self.query_set)))
            if self.strategy != 'regression':
                predicted, probability = self.run(index)
                predicted_class.append(predicted)
                probability_class.append(probability)

            else:
                predicted = self.run_regression(index)
                predicted_class.append(predicted)
                
        return predicted_class, probability_class
            
    @torch.no_grad()
    def run_regression(self, index):
        """
        Implements pXC50 assignment through the whole test set.
        
        : index (int): index reference to the SMILES strings to be predicted (from the test set)
        
        """
        
        self.model = self.model.to(self.device)
        
        pXC50s = []
        
        for i in range(self.k):
            pXC50 = self.predict_regression(index, self.model)
            pXC50s.append(pXC50)
        
        pXC50s = np.mean(pXC50s)
        
        return pXC50s
           
    @torch.no_grad()
    def run(self, index):
        """
        Implements the selected class-assignment strategy through the whole test set.
        
        : index (int): index reference to the SMILES strings to be predicted (from the test set)
        
        """
        
        # Counters
        prediction, probabilities = [], []
        
        # Passing models to GPU if available
        self.model = self.model.to(self.device)
    
        if self.model2 is not None:
            self.model2 = self.model2.to(self.device)
        
        for i in range(self.k):
            # If categorical it is a problem,
            # two the models come into play
            if self.multiclass:
                label1, probs1 = self.predict(index, self.model, self.threshold)    
                label2, probs2 = self.predict(index, self.model2, self.threshold2)
                
                # Here the decision is reached
                # once both models have output
                # their verdict
                if label2:
                    label = 2
                    probs = probs2
                elif label1:
                    label = 1
                    probs = probs1
                else:
                    label = 0
                    probs = probs1
                                     
                prediction.append(label)
                probabilities.append(probs)
            
            else:   
                # Binary label assignment, only
                # one model is required
                label, probs = self.predict(index, self.model)
                
                prediction.append(label)   
                probabilities.append(probs)
        
        # Voting occurs, that is, 
        # the label with maximal frequency
        # is assigned to the query molecule
        predicted_label = max(set(prediction), key = prediction.count)
        
        # Mean probability is computed
        probability = np.mean(probabilities)
        
        return predicted_label, probability
    
    def predict_regression(self, index, model):
        """
        Predicts z-score (~ similarity score) for the given SMILES strings (here passed as an index
        which is sliced from the generated dataset).
        
        : index (int): index reference to the SMILES strings to be predicted (from the test set)
        : model (class): instantiated RSNN model for inference
        
        """
        ### Implement assertions potentially 
        
        dataset = FewShotDataset(index, self.support_set, self.query_set, self.N,\
                                 randomise = self.randomise)
        
        dataloader = dataset(is_regression = True)
        
        for inputs1, inputs2, inputs1_lens, inputs2_lens, labels in dataloader:
            inputs1, inputs2, labels = inputs1.to(self.device), inputs2.to(self.device), labels.to(self.device)
            
            similarity = model(inputs1, inputs2, inputs1_lens, inputs2_lens, labels, predict = False).cpu().numpy()
            
            # 2 solutions since the z-score
            # imposes absolute value
            bioactivity_1 = labels.cpu().numpy() + 0.43*np.log(similarity) 
            bioactivity_2 = labels.cpu().numpy() - 0.43*np.log(similarity) 
            
            bioactivity_1 = np.mean(bioactivity_1[np.isfinite(bioactivity_1)])
            bioactivity_2 = np.mean(bioactivity_2[np.isfinite(bioactivity_2)])
            
            # Final label is taken as the mean
            bioactivity = mean([bioactivity_1, bioactivity_2])
            
        return bioactivity 
        
    def predict(self, index, model, threshold = None):
        """
        Inference for classification task.
        Predicts class label for query molecules (test set) given a threshold and a strategy.
        
        : index (int): index reference to the SMILES strings to be predicted (from the test set)
        : model (class): instantiated RSNN model for inference
        : threshold (int, optional): indicates which is the thresholding used. 
        
        """
        
        # Creating dataset
        dataset = FewShotDataset(index,  self.support_set, self.query_set, self.N, \
                                 threshold = threshold, replacement = self.replacement, balanced = self.balanced, \
                                 randomise = self.randomise)
                                          
        # Generating relevant variables from the Dataset
        dataloader, self.k_support_set_label = dataset()
        
        # Runs through the dataset
        for inputs1, inputs2, inputs1_lens, inputs2_lens, labels in dataloader:
            inputs1, inputs2, labels = inputs1.to(self.device), inputs2.to(self.device), labels.to(self.device)
              
            # Depending on n-shot learning strategy, requirements might change
            if self.strategy != 'knn':
                probs = model(inputs1, inputs2, inputs1_lens, inputs2_lens, labels, predict = False)
            else:
                label, probability = self.nearest_neighbour(inputs1, inputs2, labels) 
            
            if self.strategy == 'few-shot':
                label, probability = self.run_few_shot(probs)
            elif self.strategy == 'one-shot':
                label, probability = self.run_one_shot(probs)
        
        return label, probability
    
    def nearest_neighbour(self, inputs1, inputs2, labels):
        """
        Basline model based on k-nearest neighbours. Assignment of class to the query molecule is given by
        the closest compound (using the cosine similarity as the distance function) from the support set.
        
        : inputs1 (torch.Tensor): contains first column of tokenised SMILES strings
        : inputs2 (torch.Tensor): contains second column of tokenised SMILES strings
        : labels (torch.Tensor): contains class labels for pairs of tokenised SMILES strings
        
        """
        
        self.eps = 10e-8
        
        distances = F.cosine_similarity(inputs1.float(), inputs2.float(), dim = -1, eps = self.eps)
        
        # Squish distances to generate probabiities
        distances = torch.sigmoid(distances)
        
        if len(distances) < self.neighbours:
            raise ValueError('Select a lower number of neighbours.')
        
        dist, idxs = torch.topk(distances, self.neighbours)
        
        probs = np.mean(dist.cpu().numpy())
        
        label = self.k_support_set_label[idxs.cpu()]
        
        if not isinstance(label, float):
            label = mode(label)
            
        return label, probs
    
    def run_few_shot(self, probabilities, reference = 1):
        """
        Implements few-shot learning strategy, that is, assigns to the query molecule the class
        of the compound with which it has a higher similarity score.
        
        : probabilities (torch.Tensor): score ranging from [0,1] indicating the probability
        of similarity (being 1 complete similarity)
        : references (int, default = 1): how many compounds are taken as reference
        
        """
        
        if len(probabilities) < reference:
            raise ValueError('The support set is not big enough for the given number of instances')
       
        probs, idxs = torch.topk(probabilities, reference)
        
        probs = torch.mean(probs).cpu().numpy()
        
        label = self.k_support_set_label[idxs.cpu()]
        
        # Majority voting since there is
        # more than one inference
        if not isinstance(label, float):
            label = mode(label)
            
        if not label:
            probs = 1 - probs
            
        return label, probs
    
    def run_one_shot(self, probabilities):
        """
        Implements one-shot learning strategy, that is, averages the similarity score for each class
        and assigns the class with maximum similarity to the query molecule.
        
        : probabilities (torch.Tensor): similarity scores (similarity probability defined by SNN)
        
        """
        
        len_actives = np.count_nonzero(self.k_support_set_label)
        len_inactives = np.count_nonzero(self.k_support_set_label==0)
        
        # Because actives are placed first
        # and inactives placed last
        similarity_actives = torch.mean(probabilities[:len_actives])
        similarity_inactives = torch.mean(probabilities[-len_inactives:])
        
        probs, idx = torch.topk(torch.stack((similarity_actives, similarity_inactives)), 1)
	
        if idx:
            label = 0
            probs = 1 - probs.cpu().numpy()
        else:
            label = 1
            probs = probs.cpu().numpy()
                 
        return label, probs
    
    def calculate_metrics(self, y_true, y_pred, y_probs):
        """
        Calculates relevant metrics.
        
        : y_true (np.array): test set labels
        : y_pred (np.array): predicted labels
        : y_probs (np.array): probabilities (allows ROC calculation)
        
        """
        
        if not self.multiclass:
            TP, FN, FP, TN = self.generate_confusion_matrix(y_true, y_pred)

            accuracy = accuracy_score(y_true, y_pred)*100
            recall = recall_score(y_true, y_pred)*100
            precision = precision_score(y_true, y_pred)*100
            f1 = 2*((precision*recall/(precision + recall)))
            mcc = matthews_corrcoef(y_true, y_pred)

            FNR = FN/(FN+TP)*100 # False Negative Ratio or miss-rate
            FPR = FP/(FP+TN)*100 # False Positive Rate or fall-out

            roc_auc = roc_auc_score(y_true, y_probs)*100

            print('Accuracy: %.1f %%' % (accuracy))
            print('F1-score: %.1f %%' % (f1)) 
            print('MCC: %.2f' % (mcc))
            print('Recall: %.1f %%' % (recall))
            print('Precision: %.1f %%' % (precision))   
            print('False negative rate: %.1f %%' % (FNR))
            print('False positive rate: %.1f %%' % (FPR))
            print('ROC-AUC score: %.1f %%' % (roc_auc))
            
            return [accuracy, f1, mcc, recall, precision, FNR, FPR, roc_auc]
        else:
            weighted_kappa = cohen_kappa_score(y_true, y_pred, weights = 'linear')
            tau, p_value = stats.kendalltau(y_true, y_pred)
            
            return weighted_kappa, tau
        
    def generate_confusion_matrix(self, y_true, y_pred):
        """
        Generates the confusion matrix for the given labels and retrieves values from the contingency table.
        
        : y_true (pd.Series): ground truth class labels
        : y_pred (pd.Series): predicted class labels
        
        """
        
        if self.multiclass:
            self.cm = pd.DataFrame(confusion_matrix(y_true, y_pred),
                                   index = ['inactives', 'moderately actives', 'actives'],
                                  columns = ['inactives', 'moderately actives', 'actives'])
        else:
            self.cm = confusion_matrix(y_true, y_pred)
        
        if not self.multiclass:
            # Retrieve test results
            TP, FN, FP, TN = [], [], [], []

            TN = self.cm[0,0]
            FP = self.cm[0,1]
            FN = self.cm[1,0]
            TP = self.cm[1,1]

            return TP, FN, FP, TN
    
    def plot_confusion_matrix(self, cm):
        """
        Plots confusion matrix.
        
        : cm (np.array): array with the confusion matrix
        
        """
        
        plt.figure(figsize = (8.5,6.5))
        ax = plt.subplot()
        sns.set(font_scale = 2)
        sns.heatmap(cm, annot = True, ax = ax, cmap = 'Greens', fmt = 'g') #annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels', size = 15)
        ax.set_ylabel('True labels', size = 15)
        ax.set_title('Confusion Matrix', size = 20)
        plt.savefig(r'/projects/../../PythonNotebooks/OSL_stats/' + str(self.dataset_name) + '.svg', format = 'svg')
                
        if not self.multiclass:    
            ax.xaxis.set_ticklabels(['inactive', 'active'])
            ax.yaxis.set_ticklabels(['inactive', 'active'])
        
        plt.show()
    
    def plot_metrics(self, query_set, labels, y_probs):
        """
        Displays relevant metrics and confusion matrix.
        
        : query_set (pd.DataFrame): set containing the query molecule (molecule to be inferred)
        : labels (np.array): contains ground truth class labels
        : y_probs (np.array): probability of similarity
        
        """
        
        if self.multiclass:
            y_true = query_set.loc[:,'pXC50'].apply(self.assign_multiclass_activity)
        else:
            y_true = query_set.iloc[:,-1]
        
        y_pred = pd.Series(labels)
        
        self.generate_confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(self.cm)
        
        if self.multiclass:
            weighted_kappa, tau = self.calculate_metrics(y_true, y_pred, y_probs)
            
            return weighted_kappa, tau
        else:
            list_metrics = self.calculate_metrics(y_true, y_pred, y_probs)
            return list_metrics
                  
    @staticmethod
    def assign_multiclass_activity(row):
        """
        Given a pXC50, assigns an ordinal categorical value.
        
        : row (float):
        
        """
        
        if row < 5:
            activity = 0
        elif row >= 5 and row < 7:
            activity = 1
        elif row >= 7:
            activity = 2

        return activity