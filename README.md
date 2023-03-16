**Please note: this repository is no longer being maintained.**

# Siamese-RNN-Self-Attention
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.76](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-373/)

___

Code for the purposes of Siamese Recurrent Neural Network with a Self-Attention Mechanism for Bioactivity Prediction.
___

Activity prediction plays an essential role in drug discovery by directing search of drug candidates in the relevant chemical space. Despite being applied successfully to image recognition and semantic similarity, the Siamese neural network has rarely been explored in drug discovery where modelling faces challenges such as insufficient data and class imbalance. Here, we present a Siamese recurrent neural network model (SiameseCHEM) based on bidirectional long short-term memory architecture with a self-attention mechanism, which can automatically learn discriminative features from the SMILES representations of small molecules. Subsequently, it is used to categorize bioactivity of small molecules via N-shot learning. Trained on random SMILES strings, it proves robust across five different datasets for the task of binary or categorical classification of bioactivity. Benchmarking against two baseline machine learning models which use the chemistry-rich ECFP fingerprints as input, the deep learning model outperforms on three datasets and achieves comparable performance on the other two. The failure of both baseline methods on SMILES strings highlights that the deep learning model may learn task-specific chemistry features encoded in SMILES strings.
___

### Installation
- Clone the repo and navigate to it.
- Create a predefined Python3.7 conda environment by `conda env create -f environment.yml`.
- Run `pip install .` to install remaining dependencies and add the package to the Python path.

### Usage
``` bash
conda activate siamese
```

```python
from model import Trainer, FewShotLearner
```

### Methods
#### Trainer
- `fit()`: fit a Siamese Neural Network to a given dataset (validation set can be provided).
- `cross_validate()`: performs k-fold cross-validation.
- `predict()`: generates similarity prediction for a test set.

#### N-shot learning
Implemented with `__call__` method. 

## Contributions
Contributions are welcome in the form of issues or pull requests. To report a bug, please submit an issue. Thank you to everyone who has used the code and provided feedback thus far.
