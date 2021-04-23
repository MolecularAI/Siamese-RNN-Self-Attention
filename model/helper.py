"""
Implementation of miscellaneous functions to provide statistics, visualisations, etc.

"""

import matplotlib.pyplot as plt # Visualisation
import math # mathematical operations
#import multiprocessing as mp # CPU multiprocessing
import numpy as np # Linear algebra
import os # Directories handling
import pandas as pd # Data wrangling
import random # randomness
import scipy.stats
import statistics # statistics
import subprocess
import sys

import torch # Deep learning
import pickle

# Visualisation
from matplotlib import cm
from matplotlib.colors import Normalize 

from scipy.interpolate import interpn # Statistics

# Machine learning
from sklearn.metrics import roc_curve, auc

# Custo, dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from fingerprints import *
from pair_generator import *
from translator import *
from main_translator import *


template = """#!/bin/sh
#SBATCH --job-name={}
#SBATCH --chdir={}
#SBATCH --mem={}gb
#SBATCH --cpus-per-task={}
#SBATCH --time={}-{}:{}:{}
#SBATCH -e {}
#SBATCH -o {}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={}
"""

def create_slurm_file(job_name, environment, work_dir = None, email = None, save_location = None, memory = 50, 
                      cpu_per_task = 1, days = 6, hours=0, minutes=0, seconds=0, partition='gpu', gpu_type='any', 
                      n_gpus=1, verbose=True):
    """
    Creates a .sh file for SLURM submission.
    
    : job_name (str): name given to the current job
    : environment (str): environment with Python packages
    : work_dir (str): current working directory
    : email (str): user email, username@company.com
    : save_location (str): specify where the SLURM .sh file is to be saved, otherwise saved in default folder
    : memory (int): memory to use in GB
    : cpu_per_task (int): CPUs required
    : days (int): days requested for the .py file to run
    : hours (int): hours requested for the .py file to run
    : minutes (int): minutes requested for the .py file to run
    : seconds (int): seconds requested for the .py file to run
    : partition (str): GPU or core partition
    : gpu_type(str): GPU type (any, volta or tesla)
    : n_gpus (int): number of gpus requested
    : verbose (bool): regulates verbosity of outputs
    
    """
    
    if days==0 and hours == 0 and minutes == 0 and seconds == 0:
        raise ValueError('Need to specify non-zero time using args: days, hours, minutes, seconds')

    times = []
    
    for t in [hours, minutes, seconds]:
        str_time = str(t)
        zero_filled = str_time.zfill(2)
        times.append(zero_filled)
    
    errorfile = os.path.join(work_dir,'submission_files/logfile',job_name.split('.')[0] + '_%j.logfile')
    outputfile = os.path.join(work_dir,'submission_files/output',job_name.split('.')[0] + '_%j.output')
    
    slurm_string = template.format(job_name, work_dir, memory, cpu_per_task, days, times[0], times[1], times[2], 
                                  errorfile, outputfile, email)

    if partition == 'gpu':
        if gpu_type == 'any':
            slurm_string += "\n#SBATCH --gres=gpu:{}".format(n_gpus)
        elif gpu_type == 'volta' or gpu_type == 'tesla':
            slurm_string += "\n#SBATCH --gres=gpu:{}:{}".format(gpu_type, n_gpus)
        else:
            raise ValueError("gpu_type arg must be 'any', 'volta' or 'tesla'")
        slurm_string += '\n#SBATCH -p gpu'
    else:
        if partition != 'core':
            raise ValueError("partition arg must be 'core' or 'gpu'")
    
    slurm_string += '\n\n'
    
    slurm_string += 'source ~/.bashrc'
    slurm_string += '\n\n'
    slurm_string += 'conda activate ' + str(environment)
    
    slurm_string += '\n\n'
    slurm_string += 'echo "Starting job..."'
    slurm_string += '\necho "== Starting run at $(date)"'
    
    script = os.path.join(work_dir, 'model/main_trainer.py $1 $2')
    
    slurm_string += '\npython ' + script
    slurm_string += '\necho "Done!"'
    slurm_string += '\necho "==Finished run at $(date)"'
    
    if save_location is None:
        save_location = os.path.join(work_dir, 'submission_files/files',job_name + '.sh')
        with open(save_location, 'w') as f:
            f.write(slurm_string) 
    if verbose:
        print('SLURM file created')  
    return slurm_string, errorfile

def dump_pickle(file, path, name):
    """
    Pickles a file.
    
    : file (str): file to be pickled
    : path (str): path where the file is going to be pickled
    : name (str): name assigned to the file
    """
    
    with open(os.path.join(path,name) + '.pkl', 'wb') as f:
        pickle.dump(file, f)
        
def load_pickle(filename, path):
    """
    Loads a pickled file.
    
    : filename (str): name of the .pkl file
    : path (str): path to file
    """
    
    with open(os.path.join(path, filename) + '.pkl', 'rb') as f:
        file = pickle.load(f)
    return file

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculates 95% confidence intervals (CI).
    Taken from: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    
    : data (pd.DataFrame):
    : confidence (float):
    
    """
    
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
    return m, m-h, m+h

def select_gpu(number):
    """
    Selects GPU.
    
    : number (int): assigned GPU number
    
    """
    
   # subprocess.run(nvidia-smi, stderr=subprocess.STDOUT)
    
    try:
        torch.cuda.set_device(number)
    except:
        raise RuntimeError('GPU not available, choose another one.')
        
    # Better trace back messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(number)
        
    if torch.cuda.is_available():
        print('\n')
        print('Using GPU number {}!'.format(torch.cuda.current_device()))

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

def activity_classification(DataFrame):
    """
    Plots bar and pie chars with the activity label proportions.
    
    """
    
    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (12,7.5))

    group = DataFrame['Activity'].value_counts()
    prelabels = ['inactive','active']
                         
    ax[0].bar(prelabels,group, color = ['#1f77b4','orange'], alpha = 0.8)
    ax[0].set_xticklabels(['inactive','active'], rotation = 60)
    ax[0].set_xlabel('activity')
    ax[0].set_ylabel('number of molecules')
    
    sizes, perc = group.tolist(), (group / group.sum()*100).tolist()
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(prelabels, perc)]
    colours = ['#1f77b4', 'orange']
    ax[1].axis('equal')
    ax[1].pie(sizes, colors = colours, wedgeprops = {'alpha':0.8})
    ax[1].legend(labels,title = 'Activity', loc = 'right', bbox_to_anchor = (1.4, 0, 0, 1))

    plt.suptitle('Compound activity', y = 1.05, size = 13)
    plt.tight_layout()
    plt.show()

def activity_boxplot(DataFrame):
    """
    Plots the distribution of pXC50 according to binary activity labels with a boxplot.
    
    """
        
    inactives = DataFrame[DataFrame['Activity']==0]['pXC50']
    actives = DataFrame[DataFrame['Activity']==1]['pXC50']
    
    plt.figure(figsize = (10, 7.5))
    meanpointprops = dict(marker = '+', markeredgecolor = 'black')
    plt.boxplot([inactives, actives], vert = False, widths = 0.6, showmeans = True, meanprops = meanpointprops, labels = ['inactives','actives'])
    plt.xlabel(r'pXC$_{50}$', size = 11)
    plt.title(r'pXC$_{50}$ distribution over class', size = 13)
    plt.show()

def calculate_circular_fps(smi, useCounts=True, radius=2, size=""):
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    mol = Chem.MolFromSmiles(smi)
    
    if (useCounts==True and size == ""):
        fp = AllChem.GetMorganFingerprint(mol, radius)
    elif (useCounts==True and size != ""):
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, size)
    else:
        if (size == ""):
            size = 2048
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, size) # this is hashed too
    
    return (fp)

def calculate_bit_similarity(fp1, fp2, metric='tanimoto'):
    
    from rdkit import DataStructs
    
    metricsAvailable = {
        "allbit": DataStructs.AllBitSimilarity,  # bit
        "asymmetric": DataStructs.AsymmetricSimilarity,  # bit
        "braunblanquet": DataStructs.BraunBlanquetSimilarity, # bit
        "cosine": DataStructs.CosineSimilarity,  # bit
        "dice": DataStructs.DiceSimilarity, # bit or int
        "kulczynski": DataStructs.KulczynskiSimilarity,  # bit
        "mcconnaughey": DataStructs.McConnaugheySimilarity, # bit
        "rogotgoldberg": DataStructs.RogotGoldbergSimilarity,  # bit
        "russel": DataStructs.RusselSimilarity,  # bit
        "sokal": DataStructs.SokalSimilarity,  # bit
        "tanimoto": DataStructs.TanimotoSimilarity, # bit or int
        "tversky": DataStructs.TverskySimilarity # bit or int
    }

    if metric.lower() not in metricsAvailable:
        print ("The given metric is unknown! Selecting 'tanimoto' as default")
        metric = 'tanimoto'
    
    ### Calculate Fingerprint similarity Matrix
    return(metricsAvailable[metric.lower()](fp1, fp2))

def cartesian_product(DataFrame, dissimilar = True):
    
    inactives = DataFrame[DataFrame['Activity']==0][['SMILES','pXC50']]
    actives = DataFrame[DataFrame['Activity']==1][['SMILES','pXC50']]

    if dissimilar:        
        df = inactives.assign(key = 1).merge(actives.assign(key = 1), on = 'key').drop('key',1)
       
    else:
        df1 = actives.assign(key = 1).merge(actives.assign(key = 1), on = 'key').drop('key',1)
        df2 = inactives.assign(key = 1).merge(inactives.assign(key = 1), on = 'key').drop('key',1)
        df = pd.concat([df1, df2], axis = 0)
        df.reset_index(drop = True, inplace = True)
    
    df.columns = ['SMILES_1','pXC50_1','SMILES_2','pXC50_2']
    df['potency_difference'] = abs(df['pXC50_1']-df['pXC50_2'])
        
    return df

def get_activity_cliffs(DataFrame, augmentation_factor, dataset_name = None):
    """
    Generates pairs to be used in the SNN (since PairGenerator has a seed). 
    pXC50 differences and Tanimoto distances amongst pairs are estimated
    Saves the statistics.
    
    : DataFrame (pd.DataFrame): DataFrame containing SMILES strings, pXC50 and Activity label
    : augmentation_facotr (int): number of times to repeat pairing process
    : dataset_name (str, optional): name with which the dataframe will be saved
    
    """
    
    paired = pair(DataFrame, 1)
    paired.columns = ['SMILES_1', 'SMILES_2','similarity']
    paired.columns = ['SMILES_1', 'SMILES_2','similarity']
    paired = paired.merge(DataFrame, left_on = 'SMILES_1', right_on = 'SMILES', how = 'left')
    paired = paired.merge(DataFrame, left_on = 'SMILES_2', right_on = 'SMILES', how = 'left')
    paired['potency_difference'] = abs(paired['pXC50_x'] - paired['pXC50_y'])
    
    pool = mp.Pool(processes = 6)
    paired['fp_1'] = [pool.apply(calculate_circular_fps, args=(x,)) for x in paired['SMILES_1']]
    paired['fp_2'] = [pool.apply(calculate_circular_fps, args=(x,)) for x in paired['SMILES_2']]
    paired['Tanimoto'] = paired.apply(lambda x: calculate_bit_similarity(x.fp_1, x.fp_2), axis = 1)
    paired.drop(columns = ['SMILES_x','Activity_x', 'SMILES_y', 'Activity_y'], inplace = True)
    paired.columns = ['SMILES_1','SMILES_2','similarity','pXC50_1','pXC50_2','potency_difference','fp_1','fp_2','Tanimoto']
    
    paired.to_csv('/projects/../../PythonNotebooks/density_scatter/datasets/' + str(dataset_name) + '.csv')
    
    return paired

def density_scatter(x1, y1,
                    x2, y2,
                    x3, y3,
                    x4, y4,
                    name = None, sort = True, bins = 20, **kwargs):
    """
    Scatter plot colored by 2d histogram in https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    by Gillaume
    
    : x: Tanimoto bit similarity
    : y: potency difference
    
    """
    
    fig , ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize = (15, 12.5), nrows = 2, ncols = 2)
    
    for (x,y), ax in zip([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], [ax1, ax2, ax3, ax4]):
        
        x = np.array(x)
        y = np.array(y)


        data, x_e, y_e = np.histogram2d(x, y, bins = bins, density = True)
        z = interpn((0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x,y]).T, method = "splinef2d", bounds_error = False)

        # Sort the points by density, so that the densest points are plotted last
        if sort :
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        ax.scatter(x, y, c=z, **kwargs)
        ax.set_xticks(np.arange(0,1.1, step = 0.2))
        x_label = ax.get_xticks().tolist()
        y_label = ax.get_yticks().tolist()

        ax.set_yticklabels(np.round(y_label, 1), fontsize = 15)
        ax.set_xticklabels(np.round(x_label,1), fontsize = 15)
    
    ax1.set_ylabel(r'$\Delta$pXC$_{50}$', size = 18)
    ax3.set_ylabel(r'$\Delta$pXC$_{50}$', size = 18)
    ax3.set_xlabel('Tanimoto coefficient', size = 18)
    ax4.set_xlabel('Tanimoto coefficient', size = 18)
       
    plt.savefig('/projects/ct/ml_ai/projects/Segmentation/comp' + str(name) + '.png', dpi = 150, format = 'png')
    plt.show()

def get_descriptors(mol):
    """
    Estimates logP, TPSA, molecular weight, HBA, HBD and QED descriptors.
    Tweaked (removed active probability) from 
    https://www.wildcardconsulting.dk/master-your-molecule-generator-2-direct-steering-of-conditional-recurrent-neural-networks-crnns/
    
    : mol (rdkit.Chem.rdchem.Mol): molecule from which descriptors are estimated
    
    """
    
    logp  = Descriptors.MolLogP(mol)
    tpsa  = Descriptors.TPSA(mol)
    molwt = Descriptors.ExactMolWt(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    qed   = QED.qed(mol)
    
    return [logp, tpsa, molwt, qed, hba, hbd]

def compare_models(rf, rf_fp, svm, svm_fp, snn_mlp, snn, snn_internal, dataset = None):
    """
    Plots boxplots comparing the metrics for each of the models.
    
    : rf (list):
    : svm (list):
    : snn (list):
    : protein (str): name to be given to the SVG image
    
    """
    
    metrics = ['accuracy','F1-score','MCC','Recall','Precision','False Negative Rate','False Positive Rate']
    snn_values = list(zip(*snn))[2:]
    snn_internal_values = list(zip(*snn_internal))[2:]
    snn_mlp_values = list(zip(*snn_mlp))[2:]
    svm_values = list(zip(*svm))[:-1]
    svm_fp_values = list(zip(*svm_fp))[:-1]
    rf_values = list(zip(*rf))[:-1]
    rf_fp_values = list(zip(*rf_fp))[:-1]

    fig, ax = plt.subplots(ncols = 2, nrows = 4, figsize = (10,15))

    fig.delaxes(ax[3,1])
    
    for axes, metric, snn_fold, snn_internal_fold, snn_mlp_fold, svm_fold, svm_fp_fold, rf_fold, rf_fp_fold in list(zip(ax.flatten(), metrics, snn_values, snn_internal_values, snn_mlp_values, svm_values, svm_fp_values, rf_values, rf_fp_values)):

        axes.boxplot([snn_fold, snn_internal_fold, snn_mlp_fold, svm_fold, svm_fp_fold, rf_fold, rf_fp_fold], showmeans = True, meanprops = dict(marker= '+', markeredgecolor = 'black'))
        axes.set_title(str(metric), size = 12.5)
        axes.set_xticklabels(['SNN\n(self-attention)','SNN\n(internal processing)','MLP','SVM','SVM ECFP6','RF', 'RF ECFP6'], rotation = 60)
        if metric == 'MCC':
            axes.set_ylim(-1,1)
        else:
            if metric == 'False Negative Rate' or metric == 'False Positive Rate':
                axes.set_ylim(0,100)
            else:
                axes.set_ylim(40,100)
        plt.tight_layout()
        plt.savefig('/projects/../../PythonNotebooks/cv_boxplots/' + str(dataset) + '.svg', format = 'svg')
        
def filter_tokens(data, filter = True):
    
    # Translation of SMILES strings
    translated_data = translate_sets(data, verbose = 0)
    
    # Grouping tokens by absolute frequency
    tokens = pd.DataFrame(translated_data['Molecule'].tolist()).apply(pd.value_counts, axis = 1).sum()
    
    tokens.drop(labels = [0, 4, 62], inplace = True) # Drop 0 paddings, SOS and EOS tokens
    
    # Percentage of tokens
    tokens_pcn = tokens/tokens.sum() * 100

    # Identification of outlying molecules
    # (unfrequent tokens present in the SMILES string)
    if filter:
        
        accepted_characters = tokens_pcn.sort_values(ascending = False).head(25).index.tolist()
        accepted_characters.extend([0,4,62])
        #tokens_pcn[tokens_pcn < filtering_percentage].index.tolist()
        rejected_mols = []
        translation = [rejected_mols.append(molecule) if any(atom not in accepted_characters for atom in molecule) else True for molecule in translated_data['Molecule']]
        
        rejected_mols = pd.Series(rejected_mols)
        decoded_rejected_mols = main_decoder(rejected_mols, columns = False)
        
        print('Rejected molecules:')
        for mol in decoded_rejected_mols:
            print(mol)
            
        # Filtering
        data = pd.concat([data,pd.Series(translation)], axis = 1).dropna()
        data.reset_index(drop = True, inplace = True)
        data.drop(columns = 0, inplace = True)

        return data, tokens_pcn
    else:
        return tokens_pcn

def tokens_frequency(percentage_tokens, standard_dictionary, special_characters_dictionary):
   
    percentage_tokens.index = [standard_dictionary[i] for i in percentage_tokens.index]
    percentage_tokens.rename(index = {'A':'Cl','D':'Br'}, inplace = True)
    percentage_tokens.index = [special_characters_dictionary[i] if i in special_characters_dictionary.keys() else i for num, i in enumerate(percentage_tokens.index)]
    percentage_tokens = percentage_tokens.sort_values(ascending = True)
    
    plt.figure(figsize = (12.5,6))
    plt.bar(percentage_tokens.index, height = percentage_tokens, align = 'center', color = 'orange')
    plt.xticks(rotation = 30)
    plt.title('Relative frequency of tokens', size = 12.5)
    plt.xlabel('token', size = 10.5)
    plt.ylabel('frequency (%)', size = 10.5)
    plt.show()
    
def get_means(model, metric, siamese_network = False):
    """
    Provides means and std for a given metric.
    
    : model (list): contains folds with metrics
    : metric (str): metric to provide means and std
    : siamese_network (bool): whether the model is based on SNN or not
    """
    
    if siamese_network:
        list_metrics = list(zip(*model))[2:]
    else:
        list_metrics = list(zip(*model))
    
    metrics = ['accuracy','F1-score','MCC','Recall','Precision','False Negative Rate','False Positive Rate']
    
    dict_metrics = dict(zip(metrics, list_metrics))
    
    values = dict_metrics[metric]
    print(statistics.mean(values))
    print(statistics.stdev(values))

def plot_roc(name, N, k, **kwargs):
    """
    
    
    : name (str):
    : N (int):
    : k (int):
    : kwargs (dict):
    
    """
    
    color = color=plt.get_cmap('Set1')
    color = color(np.linspace(0,1,len(kwargs.items())))
    lw = 1.5
    
    for c, (label, prob) in zip(color,kwargs.items()):
        fpr, tpr, _ = roc_curve(test['Activity'], prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color = c,
                 lw=lw, label= str(label) + ' ROC curve (area = %0.2f)' % roc_auc)
        
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for ' + str(name))
    leg = plt.legend(loc="lower right", title = 'Hyperparameters: N = ' + str(N) + ' k = ' + str(k))
    leg._legend_box.align = "left"
    plt.show()
    
def get_mispredictions(DataFrame, strategy):
    
    FN = DataFrame[(DataFrame['Activity']==1) & (DataFrame[str(strategy)]==0)]['pXC50']
    FP = DataFrame[(DataFrame['Activity']==0) & (DataFrame[str(strategy)]==1)]['pXC50']
    
    return FP, FN

def plot_pXC50(test,kwargs):
    
    fig, ax = plt.subplots(ncols= 2, figsize = (15,8))
    
    for i, (label, misprediction) in enumerate(kwargs.items()):
        
        if label == 'false positives':
            ax[i].hist(test[test['Activity']==0]['pXC50'], color = 'orange', alpha = 0.6)
        elif label == 'false negatives':
            ax[i].hist(test[test['Activity']==1]['pXC50'], color = 'orange', alpha = 0.6)
        else:
            raise ValueError('Keys should be either "false positives" or "false negatives"')
            
        ax[i].hist(misprediction, color = 'royalblue')
        ax[i].set_title('Distribution of pXC$_{50}$ for ' + str(label))
        ax[i].set_xlabel('pXC$_{50}$')
        ax[i].set_ylabel('absolute frequency (N)')
        ax[i].grid(which = 'both')
        ax[i].set_xlim(0,10)
        
        if label == 'false negatives':
            ax[i].legend(labels = ['True positives','False negatives'])
        else:
            ax[i].legend(labels = ['True negatives','False positives'])
            
    plt.tight_layout()
    plt.show()
    
def mcc_dataframe(**kwargs):
    """
    Retrieves Matthew's Correlation Coefficient from a given list with metrics.
    
    : kwargs (dict): keys and metrics values for each algorithm
    
    """
    
    from re import search

    df = pd.DataFrame()

    for name, algorithm in kwargs.items():

        if search('RSNN', name) or search('MLP', name):
            algorithm = list(zip(*algorithm))[4]
        else:
            algorithm = list(zip(*algorithm))[2]

        algorithm = pd.DataFrame(algorithm, columns = [str(name)])

        df = pd.concat([df, algorithm], axis = 1)

    df = df.melt()
    
    return df

def plot_ci(pxc50, **kwargs):
    """
    Plots barchart with 95% confidence intervals for Matthew's Correlation Coefficient.
    
    : kwargs (dict): keys and MCC values for each algorithm
    : pxc50 (int): value of the pXC50 to display in title
    """
    
    import seaborn as sns
    
    df = mcc_dataframe(kwargs)
    
    labels = ['RSNN \n(self-attention)','RSNN \n(internal processing)','MLP','SVM','SVM ECPF6','RF','RF ECFP6']

    plt.figure(figsize = (7,5))
    sns.barplot(x = 'variable', y = 'value', data = df, ci = 95, capsize = .1)
    plt.xticks(range(0,len(labels)),labels, rotation = 60)
    plt.title("Matthew's correlation coefficient by algorithm pXC$_{50}$ = " + str(pxc50), size = 12)
    plt.ylabel("Matthew's Correlation Coefficient (MCC)")
    plt.xlabel('')
    plt.ylim(-1,1)
    plt.show()