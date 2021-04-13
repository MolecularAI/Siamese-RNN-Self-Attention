"""
Helper class to calculate metrics to track learning of MLP or RSNN.

"""

from scipy import stats
from sklearn.metrics import mean_squared_error

import numpy as np
import torch

class CalculateMetrics():
    def __init__(self, similarity, labels, is_regression = False):
        """
        Constructor.
        
        : similarity (torch.Tensor): contains similarity scores
        : labels (torch.Tensor): contains the ground truth labels
        : is_regression (bool): signals whether regression or classification metrics should be calculated
        
        """
        
        self.similarity = similarity
        self.labels = labels
        self.is_regression = is_regression
        
        # Since the similarity ranges from 0 to 1, 
        # it can be rounded up to generate a binary score
        self.similarity_rounded = torch.round(similarity)
    
    def __call__(self):
        """
        Callable.
        
        """
        
        if not self.is_regression:
            # Classification metrics
            self.establish_mispredictions()

            # Get metrics
            accuracy = self.calculate_accuracy()
            f1, recall, precision = self.calculate_f1()
            mcc = self.calculate_mcc()
            fpr, fnr = self.calculate_rates()

            return [accuracy, f1, mcc, recall, precision, fpr, fnr]
        else:
            # Regression metrics
            r, rmse = self.calculate_regression_metrics()
                        
            return [r, rmse]

    def calculate_accuracy(self):
        """
        Computes accuracy given the rounded similarity score and the binary labels.
        
        """
        
        # Computation of correct predictions
        n_correct = torch.sum(torch.eq(self.similarity_rounded, self.labels))
        
        # Accuracy
        accuracy = n_correct.item()/self.similarity_rounded.size()[0]*100
        
        return accuracy
    
    def establish_mispredictions(self):
        """
        Given the binary labels and rounded similarity scores, establishes the number of true positives, 
        true negatives, false positives and false negatives.
        
        """
        
        
        # Establishing reference (true) and batch (predicted) positives and negatives
        pos_ref, neg_ref = (self.labels==1).nonzero().squeeze(), (self.labels==0).nonzero().squeeze()
        pos_batch, neg_batch = (self.similarity_rounded==1).nonzero().squeeze(), (self.similarity_rounded==0).nonzero().squeeze()
           
        # Establishing the number of true positives (tp), false positives (fp) and false negatives (fn)
        self.n_tp = len(np.intersect1d(pos_batch.cpu().numpy(), pos_ref.cpu().numpy()))
        self.n_tn = len(np.intersect1d(neg_batch.cpu().numpy(), neg_ref.cpu().numpy()))
        self.n_fp = len(np.setdiff1d(pos_batch.cpu().numpy(), pos_ref.cpu().numpy()))
        self.n_fn = len(np.setdiff1d(neg_batch.cpu().numpy(), neg_ref.cpu().numpy()))
    
    def calculate_mcc(self):
        """
        Calculates Matthew's Correlation Coefficient.
        
        """

        # Matthews Correlation coefficient calculation
        denominator = (self.n_tp+self.n_fp)*(self.n_tp+self.n_fn)*(self.n_tn+self.n_fp)*(self.n_tn+self.n_fn)
            
        # Correction if denominator is 0
        if not denominator:
            denominator = 1
        
        mcc = ((self.n_tp*self.n_tn)-(self.n_fp*self.n_fn))/np.sqrt(denominator)
        
        return mcc
    
    def calculate_recall(self):
        """
        Calculates recall.
        
        """
        
        # Avoids zero division error
        try:
            recall = self.n_tp/(self.n_tp + self.n_fn) * 100
        except:
            recall = 0
        return recall
    
    def calculate_precision(self):
        """
        Calculates precision.
        
        """
        
        
        try:
            precision = self.n_tp/(self.n_tp + self.n_fp) * 100
        except:
            precision = 0
        return precision 
    
    def calculate_f1(self):
        """
        Calculates F1-score given the precision and recall.
        
        """
        
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        
        try:
            f1 = 2*(precision*recall)/(precision+recall)
        except:
            f1 = 0
        return f1, recall, precision
    
    def calculate_rates(self):
        """
        Calculates false positive and false negative rates from mispredicted datapoints.
        
        """
        
        # False positive and false negative rates
        fpr = self.n_fp / (self.n_fp + self.n_tn) * 100
        fnr = self.n_fn / (self.n_fn + self.n_tp) * 100
        
        return fpr, fnr 
    
    def calculate_regression_metrics(self):
        """
        Calculates Pearson r and root mean squared error (RMSE) given continuous similarity labels (exponentiated z-scores)
        and predicted continuous similarity labels from RSNN.
        
        """
        
        r, p_value = stats.pearsonr(self.labels.cpu().detach().numpy(), self.similarity.cpu().detach().numpy())
        rmse = np.sqrt(mean_squared_error(self.labels.cpu().detach().numpy(), self.similarity.cpu().detach().numpy()))
        
        return r, rmse