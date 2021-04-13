import itertools
import numpy as np # linear algebra
import math # basic math operations
import matplotlib
import matplotlib.pyplot as plt # Visualisation
import pandas as pd # Data wrangling
import pickle # saving files
import sys

# Avoid plotting graphs
matplotlib.use('Agg')

# Deep learning
import torch
import torch.nn.utils as tnnu
import torch.utils.data as tud

from itertools import product

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from dataset import *
from helper import *
from pair_generator import *
from randomiser import Randomiser
from main_decoder import *

from sklearn.model_selection import StratifiedKFold # Cross-validation
from statistics import mean # statistics
from tqdm import tqdm_notebook as tqdm # progress bar

# Pytorch functions
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

"""
Implementation of model training method.

"""

class Train(object):
    
    def __init__(self, model, dataset_name, **kwargs):
        """
        Initialiser.
        
        : model (nn.Module): model to be trained 
        : epochs (int): number of epochs for which to train the model
        : batch_size (int): mini-batch size to feed at each iteration
        : augmentation_factor_train (int): number of times to generate pairs for the training set
        : augmentation_factor_val (int, optional): number of times to generate pairs for the validation set
        : randomising_times (int, optional): number of randomising SMILES to generate from a canonincal SMILES
        : dataset (str, optional): name of the dataset used
        : clip (int, default = 1): gradient clipping to avoid gradient explosion
        : is_mlp (bool): the SNN used is based on linear layers and morgan fingerprints
        : early_stopping (bool): enforce early stopping based on loss function
        : verbose (int, default = 2): handle verbosity of the training process
        
        """
        
        self._model = model
        self._dataset_name = dataset_name
        self._pretraining_model = kwargs.get('pretraining_model', None)
        self._epochs = kwargs.get('epochs',150)
        self._batch_size = kwargs.get('batch_size',64)
        self._augmentation_factor_train = kwargs.get('augmentation_factor_train',1)
        self._augmentation_factor_val = kwargs.get('augmentation_factor_val',1)
        self._randomising_times = kwargs.get('randomising_times', None)
        self._clip = kwargs.get('clip',1)
        self._is_regression = kwargs.get('is_regression',False)
        self._is_mlp = kwargs.get('is_mlp',False)
        self._early_stopping = kwargs.get('early_stopping', False)
        self._verbose = kwargs.get('verbose',2)
        
        # Device agnostic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Names of collected metrics for representation in figures
        if self._is_regression:
            self.metric_names = ['learning rate','loss',"Pearson's r",'RMSE']
        else:
            self.metric_names = ['learning rate','loss','accuracy','F1-score','Matthew Correlation Coefficient','precision','recall','false negative rate','false positive rate']
        
    def transfer_learning(self, layer = 'all'):
        """
        Enables loading a trained model/transfer learning.
        
        : layer (str, default): name of the layer to be loaded. If 'all', load all model layers
        
        """
        
        if layer == 'all':
            pretraining_dict = {k:v for k, v in self._pretraining_model.items()}
        else:
            # Selecting specific layer
            pretraining_dict = {k:v for k, v in self._pretraining_model.items() if str(layer) in k}
                
        # Transferring weights to model
        model_dict = self._model.state_dict()
        model_dict.update(pretraining_dict)
        
        self._model.load_state_dict(model_dict)
    
    def cross_validate(self, k_folds, training_set, validation_set = None, hyperopt = False, verbose = 1):
        """
        Performs k-fold cross-validation 
        
        : training_set (pd.DataFrame):
        : validation_set (pd.DataFrame, optional):
        : k_folds (int):
        : split (class):
        : randomising_times (int, optional):
        
        """
        
        self._verbose = verbose
        self.kfolds = k_folds
        
        # Creation of lists for fold statistics collection
        self.cross_val_train, self.cross_val_validation = [], []
        
        # Regeneration of dataset from train and validation sets
        merged_set = pd.concat([training_set, validation_set], axis = 0).reset_index(drop = True)
        
        self._skf = StratifiedKFold(n_splits = k_folds, random_state = 42)
        
        if self._verbose > 0:
            print('Cross-validation starting...')
            
        for k, (k_train, k_val) in enumerate(self._skf.split(merged_set['SMILES'], merged_set['Activity'])):
            
            # Generate k-fold
            k_train = merged_set.iloc[k_train, :]
            k_val = merged_set.iloc[k_val,:]
            
            k_train.reset_index(drop = True, inplace = True)
            k_val.reset_index(drop = True, inplace = True)
                        
            if self._verbose > 0:
                print('Fold {}/{} ...'.format(k+1, self.kfolds))
            
            self.fit(k_train, k_val)
            
            # Take the metrics in the last 5 epochs (assuming stabilisation)
            self.stats_fold_train = list(map(mean,zip(*self.stats_epoch_train[-5:])))
            self.stats_fold_val = list(map(mean,zip(*self.stats_epoch_val[-5:])))
            
            # Append the fold metrics
            self.cross_val_train.append(self.stats_fold_train)
            self.cross_val_validation.append(self.stats_fold_val)
        
        # Calculate the means of the folds
        self.cv_train = list(map(mean,zip(*self.cross_val_train)))
        self.cv_validation = list(map(mean,zip(*self.cross_val_validation)))
      
        if not hyperopt:    
            # Saving the metrics for each fold
            dump_pickle(self.cross_val_validation, '/projects/../../PythonNotebooks/cv_statistics', str(k_folds) + '_fold_' + str(self._dataset_name) + '_SNN_metrics')
            
            if self._verbose > 0:
                # Plotting boxplots
                self.plot_kfolds()
                
                # Printing averages of relevant metrics
                for metric, train, val in zip(self.metric_names, self.cv_train, self.cv_validation):    
                    if metric == 'learning rate':
                        continue
                    elif metric == 'loss':
                        print('Mean training loss:{:.4f} ... Mean validation loss: {:.4f}'.format(train, val))
                    elif metric == 'Matthew Correlation Coefficient':
                        print('Mean training MCC: {:.5f} ... Mean validation MCC: {:.5f}'.format(train, val))
                    elif self._is_regression:
                        print("Mean training {}: {:.5f} ... Mean validation {}: {:.5f}".format(metric, train, metric, val))
                    else:
                        print('Mean training {}: {:.3f}% ... Mean validation {}: {:.3f}%'.format(metric, train, metric, val))
                print('-'*100)
        else:
            return self.cv_validation
        
        print('Done!')
        
    def fit(self, training_set, validation_set = None):
        """
        Starts the training process.
        
        : training_set (pd.DataFrame): training set used for training. If randomised, the set should not be translated
        : validation_set (pd.DataFrame, optional): validation set used. If randomised, the set should not be translated
        
        
        """
        
        # Constructing the model
        self._model.build()
        
        # Allowing pretrain
        if self._pretraining_model is not None:
            self.pretrain()
       
        # Scheduling
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._model.optimiser, patience = 10, factor = 0.1)
        
        # Setting the training and validation sets
        self.training_set = training_set
        self.validation_set = validation_set
        
        # Creating lists for statistics collection
        self.stats_epoch_train, self.stats_epoch_val = [], []

        # Instantiating early stopping if required
        if self._early_stopping:
            early_stopping = EarlyStopping(dataset_name = self._dataset_name, verbose = True)
        
        # Starting the training process
        for self.epoch in range(1,self._epochs+1):
            
            if self._verbose > 0:
                print('\n')
                print('Epoch: {}/{}'.format(self.epoch, self._epochs))
            
            # Creation of dataloaders
            self.train_dataloader = self._initialise_dataloader(self.training_set, self._augmentation_factor_train)
            
            if self.validation_set is not None:
                self.val_dataloader = self._initialise_dataloader(self.validation_set, self._augmentation_factor_val, evaluate = True)
            else:
                self.val_dataloader = None
           
            # Running each epoch
            self._run_epoch()
            
            # Retrieve loss and pass it through the scheduler
            self.scheduler.step(self.stats_train[0])
        
            # Display metrics in jupyter if required
            if self._verbose > 0:
                self.display_metrics()
            
            # Collect statistics for every epoch
            self.stats_epoch_train.append(self.stats_train)
            
            if self.validation_set is not None:
                self.stats_epoch_val.append(self.stats_val)
            
            # Early stopping if validation loss does not decrease 
            if self._early_stopping:
                early_stopping(self.stats_val[0], self._model)

                if early_stopping.early_stop:
                    print('Early stopping')
                    break
    
                    # Load the last checkpoint with the best model
                    self._model.load_state_dict(torch.load(str(self._dataset_name) + 'checkpoint.pt'))
        
        dump_pickle(zip(*self.stats_epoch_val),'/projects/../../PythonNotebooks/fit_statistics', str(self._dataset_name) + '_1_fold')
        
        # Plot learning curves if required
        if self._verbose > 1:
            self.plot_metrics()
    
    @torch.no_grad()
    def predict(self, data, model = None, which = 'validation', save = True):
        """
        Predicts the binary similarity label for a given pair with the current trained model.

        : which (str): indicates which set is fed onto the system

        """
        
        if self._is_regression:
            raise NotImplementedError('Not implemented for regression tasks.')
            
        # Selects which dataloader will be used
        if which == 'validation':
            self.mode = 'val'
            augmentation = self._augmentation_factor_val
        elif which == 'training':
            self.mode = 'train'
            augmentation = self._augmentation_factor_train
        
        if model is not None:
            self._model = model
               
        # Storage lists
        predictions, ground_truth = [], []
        smi_1, smi_2 = [], []
        
        # Generate dataloader
        dataloader = self._initialise_dataloader(data, augmentation)
        
        # Running iterations
        for inputs1, inputs2, inputs1_lens, inputs2_lens, labels in tqdm(dataloader):
            similarity = self._run_iteration(inputs1, inputs2, inputs1_lens, inputs2_lens, labels, get_metrics = False)
            
            # Store predictions and ground truth
            predictions.append(np.round(similarity.cpu().numpy()))
            ground_truth.append(labels.cpu().numpy())
            
            # Store SMILES strings from each pair
            smi_1.append(inputs1.cpu().numpy())
            smi_2.append(inputs2.cpu().numpy())
        
        predictions, ground_truth, smi_1, smi_2 = list(map(itertools.chain.from_iterable, [predictions, ground_truth, smi_1, smi_2]))
        self.predictions, self.ground_truth, self.smi_1, self.smi_2 = list(predictions), list(ground_truth), list(smi_1), list(smi_2)

        # Select mispredictions
        self.get_mispredictions()
        
        if save:
            self.df_mispredictions.to_csv('/projects/../../PythonNotebooks/mispredictions/' + str(self._dataset_name) + '.csv')
        
        return self.df_mispredictions  
        
    def get_mispredictions(self):
        """
        Selects false positive and false negative predictions from binary similarity inference task.

        """
        
        df = pd.DataFrame(zip(self.smi_1, self.smi_2, self.predictions, self.ground_truth), 
                          columns = ['SMILES_1', 'SMILES_2', 'predictions','ground_truth'])
        
        # Selection of mispredictions
        mispredictions = df[df.iloc[:,2] != df.iloc[:,-1]]
        
        # Decoding tokenised SMILES
        decoded_pair_1, decoded_pair_2 = list(map(main_decoder, [mispredictions['SMILES_1'],mispredictions['SMILES_2']]))
         
        # Reconverting randomised SMILES strings
        # into canonical SMILES
        if self._randomising_times is not None:
            mol_1, mol_2 = [[Chem.MolFromSmiles(smi) for smi in i] for i in [decoded_pair_1, decoded_pair_2]]
            decoded_pair_1, decoded_pair_2 = [[Chem.MolToSmiles(x, doRandom = False, isomericSmiles = False) for x in smi] for smi in [mol_1, mol_2]]
        
        self.df_mispredictions = pd.DataFrame(zip(decoded_pair_1, decoded_pair_2, mispredictions['predictions'], mispredictions['ground_truth']),
                                        columns = ['SMILES_1','SMILES_2','predictions','ground_truth'])
    
    def _initialise_dataloader(self, dataset, augmentation_factor, shuffle = True, evaluate = True):
        """
        Creates the dataloader needed for generating batches from the given data.
        
        : dataset (pd.DataFrame):
        : augmentation_factor (int):
        
        """       
        
        if evaluate:
            shuffle = False
        
        if self._is_mlp:
            fps = calculate_circular_fp(dataset.loc[:,'SMILES'])
            
            if self._is_regression:
                class_label = dataset.loc[:,'pXC50']
            else:
                class_label = dataset.loc[:,'Activity']
            
            dataset = pd.concat([fps, class_label], axis = 1)
            
            dataset = Dataset(dataset, transform = transforms.Compose([PairGenerator(augmentation_factor, 
                                                                                     fp = self._is_mlp,
                                                                                     is_regression = self._is_regression),
                                                                       Randomiser(randomising_times = self._randomising_times,
                                                                                  is_mlp = self._is_mlp),
                                                                       ToTensor(is_mlp = self._is_mlp)]))
        else:   
            dataset = Dataset(dataset, transform = transforms.Compose([PairGenerator(augmentation_factor,
                                                                                    is_regression = self._is_regression),
                                                                       Randomiser(randomising_times = self._randomising_times,
                                                                                 is_mlp = self._is_mlp),
                                                                       ToTensor()]))
     
        return tud.DataLoader(dataset = dataset, shuffle = shuffle, batch_size = self._batch_size, 
                            collate_fn = dataset.collate_fn, drop_last = True, pin_memory = True)
            
    def _run_epoch(self):
        """
        Runs a complete epoch running the validation set, too.
        
        """
        statistics_train = []
        
        self.episode = 0
   
        self.mode = 'train'
        
        for inputs1, inputs2, inputs1_lens, inputs2_lens, labels in tqdm(self.train_dataloader):
            metrics = self._run_iteration(inputs1, inputs2, inputs1_lens, inputs2_lens, labels)
            statistics_train.append(metrics)
            
            self.episode += 1
           
            if self.val_dataloader is not None:
                self.evaluate()
            
        # Average of metrics across iteration
        self.stats_train = list(map(np.mean,zip(*statistics_train)))
        
    def _run_iteration(self, inputs1, inputs2, inputs1_lens, inputs2_lens, labels, get_metrics = True):
        """
        Runs an iteration, that is, runs a batch through the model.
        
        : inputs1 ():
        : inputs2 ():
        : inputs1_lens ():
        : inputs2_lens ():
        : labels ():
        
        """
        
        if self.mode == 'train':
            # Grad zero and mode change
            self._model.optimiser.zero_grad()
            self._model.train()

        if self.mode == 'val':
            self._model.eval()
                   
        inputs1, inputs2, labels = inputs1.to(self.device), inputs2.to(self.device), labels.to(self.device) 

        # Forward pass
        metrics, similarity = self._model(inputs1, inputs2, inputs1_lens, inputs2_lens, labels, predict = True)
        
        # Get loss
        loss = self._model.get_loss(similarity, labels)
        
        if get_metrics:
            metrics.insert(0,loss.item())
        
        if self.mode == 'train':
            # Backpropagate loss
            loss.backward()

            # Clip exploding gradients
            tnnu.clip_grad_norm_(self._model.get_model_params()[1], self._clip)

            # Optimise model
            self._model.optimiser.step()
            
        if get_metrics:
            return metrics
        else:
            return similarity
    
    def evaluate(self):
        """
        Evaluates the performance of the model with the validation set (if present).
        
        """
        
        statistics_val = []
        
        # Starting evaluation
        if self.episode == len(self.train_dataloader) - 1:
            self.mode = 'val'
            
            for inputs1, inputs2, inputs1_lens, inputs2_lens, labels in tqdm(self.val_dataloader):   
                val_metrics = self._run_iteration(inputs1, inputs2, inputs1_lens, inputs2_lens, labels)
                
                statistics_val.append(val_metrics)
            
            # Average of metrics across iteration
            #print(f'Statistics_val: {statistics_val}')
            #print(f'list(zip(*statistics_val)): {list(zip(*statistics_val))}')
            #print(f'map(mean,zip(*statistics_val)): {map(mean,zip(*statistics_val))}')
            self.stats_val = list(map(np.mean,zip(*statistics_val)))
            
    def save_model(self, name):
        """
        Saves the trained model.
        
        : name (str): name to give to the model
        
        """
            
        torch.save(self._model.state_dict(), '/projects/../../PythonNotebooks/saved_models/' + str(name) + '.pt')

    def display_metrics(self):
        """
        Prints out learning rate, loss, accuracy, F1-score, precision, recall, false negative rate and false positive rate
        for every epoch, both for training and validation sets.
        
        """
               
        # Prepend learning rate
        self.stats_train.insert(0, self._model.optimiser.param_groups[0]['lr'])
                
        if self.validation_set is not None:
            # Filling in the first position to have lists with the same length
            self.stats_val.insert(0, np.nan)
            
            for metric, train, val in zip(self.metric_names, self.stats_train, self.stats_val):
                if metric=='learning rate':
                    print('The learning rate is: {:.6f}'.format(train))
                elif metric=='loss':
                    print('Training loss: {:.4f} ... Validation loss: {:.4f}'.format(train, val))
                elif metric == 'Matthew Correlation Coefficient':
                    print('Training MCC: {:.5f} ... Validation MCC: {:.5f}'.format(train, val))
                elif self._is_regression:
                    print("Mean training {}: {:.5f} ... Mean validation {}: {:.5f}".format(metric, train, metric, val))
                else:
                    print('Training {}: {:.3f}% ... Validation {}: {:.3f}%'.format(metric, train, metric, val))
            print('-'*100)
        else:
            for metric, train in zip(self.metric_names, self.stats_train):
                if metric == 'learning rate':
                    print('The learning rate is: {:.6f}'.format(train))
                elif metric == 'loss':
                    print('Training loss: {:.4f}'.format(train))
                elif metric == 'Matthew Correlation Coefficient':
                    print('Training {}: {:.5f}'.format(metric, train))
                else:
                    print('Training {}: {:.3f}%'.format(metric, train))
            print('-'*100)
    
    def plot_metrics(self, ncols = 1, nrows = 5):
        """
        Displays graphs with the evolution of metrics (learning rate, accuracy, F1-score, precision, recall, 
        false positive rate, false negative rate) 
        
        : ncols (int): number of columns to display in the graph
        : nrows (int): numbre of rows to display in the graph
        
        """
        
        target_dict_train = {t[0]:t[1] for t in zip(self.metric_names, zip(*self.stats_epoch_train))}
        
        if self.stats_epoch_val is not None:
            target_dict_val = {t[0]:t[1] for t in zip(self.metric_names, zip(*self.stats_epoch_val))}
            ncols = 2
        
        if self._is_regression:
            nrows = 2
        
        fig, ax = plt.subplots(ncols = ncols, nrows = nrows, sharex = 'col', figsize = (15,10))
        fig.delaxes(ax[-1,-1])
        
        for axes, (key_train, value_train) in zip(ax.flatten(), target_dict_train.items()):    
            if key_train == 'learning rate':
                axes.set_yscale('log')
            
            if key_train == 'loss':
                dump_pickle(value_train, '/projects/../../PythonNotebooks/cv_statistics',str(self._dataset_name) + 'training_loss_1_fold')
    
            if key_train == 'Matthew Correlation Coefficient' or key_train == "Pearson's r":
                axes.set_ylim(-1,1)
                axes.axhline(y = 0, color = 'green', linestyle = '--', label = 'random guessing')
                
            if key_train not in ['learning rate','loss',"Pearson's r",'Matthew Correlation Coefficient','False negative rate','False positive rate']:
                axes.axhline(y = 50, color = 'green', linestyle = '--', label = 'random guessing')
                 
            # Display label in x-axis only in the last two graphs
            if key_train in self.metric_names[-2:]:
                axes.set_xlabel('epochs / N', size = 11)
            
            axes.plot(np.arange(len(value_train)), value_train, label = 'training set')
            
            axes.set_title(str(key_train.capitalize()) + ' for Siamese Neural Network', size = 12.5)
            axes.set_ylabel(str(key_train.capitalize()), size = 11)
            axes.grid()
            
        for axes, (key_val, value_val) in zip(ax.flatten(), target_dict_val.items()):
            axes.plot(np.arange(len(value_val)), value_val, label = 'validation set')
            
            if key_val != 'learning rate':
                axes.legend(loc = 'lower right', fontsize = 11)
            
            if key_val == 'loss':
                dump_pickle(value_val, '/projects/../../PythonNotebooks/cv_statistics', str(self._dataset_name) + 'val_loss_1_fold')
            
        plt.tight_layout()
        plt.show()
        
    def plot_kfolds(self):
        """
        Generates boxplots to visualise performance over cross-validation folds.
        
        """
        
        target_dict = {t[0]:t[1] for t in zip(self.metric_names, zip(*self.cross_val_validation))}
        target_dict_metrics = {key:value for key, value in target_dict.items() if key not in ['learning rate','loss','false positive rate', 'false negative rate','Matthew Correlation Coefficient']}
        target_dict_mispredictions = {key:value for key, value in target_dict.items() if key in ['false positive rate', 'false negative rate']}
        target_dict_MCC = {key:value for key, value in target_dict.items() if key is 'Matthew Correlation Coefficient'}
        
        fig, axes = plt.subplots(ncols = 3, figsize = (12.5, 7.5))

        for ax, dictionary in zip(axes.flatten(), [target_dict_metrics, target_dict_mispredictions, target_dict_MCC]):
            ax.boxplot(dictionary.values(), showmeans = True, meanprops = dict(marker= '+', markeredgecolor = 'black'))
            ax.set_xticklabels(dictionary.keys(), rotation = 60, size = 11)
            
            ax.set_ylabel('metric (%)', size = 10.5)

            if 'accuracy' in dictionary.keys():
                ax.set_title(str(self._dataset_name) + ' performance validation metrics boxplots \n for ' + str(self.kfolds) +'-folds on Siamese Neural Network', size = 11.5)
            elif 'Matthew Correlation Coefficient' in dictionary.keys():
                ax.set_ylabel('Matthews Correlation Coefficient', size = 10.5)
                ax.set_ylim(-1,1)
                ax.set_yticks(np.linspace(-1,1,21))
                ax.set_title(str(self._dataset_name) + ' Matthew Correlation Coefficient validation boxplot \n for ' + str(self.kfolds) +'-folds on Siamese Neural Network', size = 11.5)
            else: 
                ax.set_title(str(self._dataset_name) + ' misprediction validation ratios boxplots \n for ' + str(self.kfolds) +'-folds on Siamese Neural Network', size = 11.5)
        
        plt.tight_layout()
        plt.savefig('/projects/../../PythonNotebooks/cv_boxplots/' + str(self._dataset_name) + '.svg', format = 'svg')
        
"""
Implementation of early stopping method to avoid overfitting.

"""

class EarlyStopping(object):
    """
    Early stops the training if validation loss does not improve after a given patience.
    Adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    
    """
    def __init__(self, dataset_name, patience = 5, wait = 20, delta = 0, verbose = False):
        """       
        Args:
        
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            wait (int): How many epochs to wait until EarlyStopping patience can start.
                        Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        
        self.dataset_name = dataset_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_val_min = np.PZERO
        self.delta = delta
        self.wait = wait

    def __call__(self, loss_val, model):
        
        # Due to instability in the first epochs
        if self.wait == 0:
            score = loss_val
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(loss_val, model)

            # If loss increases or stabilises, set counter on
            elif score >= self.best_score + self.delta:
                self.counter += 1
                print(f'Validation loss patience counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(loss_val, model)
                self.counter = 0
        else:
            self.wait -= 1

    def save_checkpoint(self, loss_val, model):
        """
        Saves model when accuracy in validation set increases
        
        : accuracy_val:
        : model:
        """
        if self.verbose:
            if loss_val - self.loss_val_min > 0 and self.loss_val_min != 0:
                percentage = ((loss_val - self.loss_val_min)/self.loss_val_min)*100
                
                print(f'Validation loss increased {percentage:.2f} %. Saving model...')
                print('-'*75)
            else:         
                print(f'Validation loss changed from {self.loss_val_min:.4f} to {loss_val:.4f}. Saving model ...')
                print('-'*75)
        
        torch.save(model.state_dict(), '/projects/../../PythonNotebooks/saved_models/' + str(dataset_name) + 'checkpoint.pt')
        self.loss_val_min = loss_val