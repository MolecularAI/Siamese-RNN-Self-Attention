"""
Implements the Recurrent Siamese Neural Network (RSNN) model.

"""

import numpy as np # Linear algebra
import pandas as pd # Data wrangling

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as tnni
import torch.nn.utils.rnn as tnnur
import sys

from torch import optim # Deep learning optimiser

from tqdm import tqdm_notebook as tqdm # progress bar

# Custom dependencies
sys.path.append('/projects/../../PythonNotebooks/model/')
from helper import *
from metrics_calculator import *

import numpy as np # Linear algebra
import pandas as pd # Data wrangling

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as tnni
import torch.nn.utils.rnn as tnnur

# Interact with the system
import sys

from torch import optim # Deep learning optimiser

from tqdm import tqdm_notebook as tqdm # progress bar

class SNN(nn.Module):
    def __init__(self, **kwargs):
        
        """
        Initialiser.
        
        
        : hidden_size (int): size of the hidden dimension of the RNN
        
        : num_layers (int): number of RNN layers
        : expansion_size (int): 
        : dist_fn (str): energy function to calculate distance between the two SNN arms
        : cell_type (str): establish the RNN type (LSTM or GRU)
        : embedding_dropout (float): dropout for the embedding layer
        : dropout (float): dropout for the RNN layers
        : learning_rate (float): 
        : weight_decay (float): L2 regularisation penalty (Ridge)
        : bidirectional (bool): toggle to regulate uni- or bidirectional RNN
        : init_weights (bool): toggle to regulate weight initialisation
        : embedding_dimensions (int, optional): size of the embedding layer dimension
        : embedding (bool): toggle to regulate the presence of an embedding
             
        """
        
        super(SNN, self).__init__()
        
        # Ensuring reproducibility
        set_seed()
        
        # Defining the layers depending on SNN MLP or RSNN
        self._is_mlp = kwargs.get('is_mlp', False)
        
        if self._is_mlp:
            self._hidden_size = kwargs.get('hidden_size', 512) 
            self._output_size = kwargs.get('output_size', 256)
            self._dist_fn = kwargs.get('dist_fn', 'cos')
            self._loss = kwargs.get('loss','mse')
            self._similarity_fn = kwargs.get('similarity_fn',None)
            self._initialisation_process = kwargs.get('initialisation_process',None)
            self._input_size = kwargs.get('input_size', 2048)
            self._learning_rate = kwargs.get('learning_rate', 0.005)
            
            if 'hidden_states_processor' in list(kwargs.keys()):
                raise NotImplementedError('No hidden states processor is used for SNN MLP.')
            
        else:
            self._hidden_size = kwargs.get('hidden_size',128)
            self._n_layers = kwargs.get('n_layers',3)
            self._dist_fn = kwargs.get('dist_fn','cos')
            self._similarity_fn = kwargs.get('similarity_fn','clamp')
            self._dist_fn = kwargs.get('dist_fn','cos')
            self._cell_type = kwargs.get('cell_type','LSTM')
            self._loss = kwargs.get('loss','logcosh')
            self._initialisation_process = kwargs.get('initialisation_process','kaiming_normal')
            self._input_size = kwargs.get('input_size',150)
            self._bidirectional = kwargs.get('bidirectional',True)
            self._normalisation = kwargs.get('normalisation', None)
            self._embedding_dimensions = kwargs.get('embedding_dimensions', 128)
            self._embedding_dropout_p = kwargs.get('embedding_dropout',0.05)
            self._learning_rate = kwargs.get('learning_rate', 0.0001)
            
            # Hidden states processing mechanism is defined in another class
            self._hidden_states_processor = kwargs.get('hidden_states_processor',None)
            
             # Avoiding dropout if the model only has one layer
            if self._n_layers > 1:
                self._dropout = kwargs.get('dropout',0.05)
            else:
                self._dropout = 0
                
        # Shared hyperparameters across RSNN and SNN MLP 
        self._weight_decay = kwargs.get('weight_decay',0.00)
        self._is_regression = kwargs.get('is_regression', False)
        
        # Additional 'this' entities 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._eps = 10e-8 # to avoid division by 0
    
    def build(self):
        """
        Instantiates layers.
        
        """
        
        if not self._is_mlp:
            
            # Embedding
            if self._embedding_dimensions is not None:
                self.embedding = nn.Embedding(self._input_size, self._embedding_dimensions, padding_idx = 0).to(self.device)
                self.input_rnn = self._embedding_dimensions
                self.embedding_dropout = nn.Dropout(p = self._embedding_dropout_p)
            else:
                self.input_rnn = self._input_size

            # Defining the RNN layers
            if self._cell_type.lower() == 'gru':
                self.rnn = nn.GRU(self.input_rnn, self._hidden_size, self._n_layers, 
                                    dropout = self._dropout, batch_first = True, bidirectional = self._bidirectional).to(self.device)
            elif self._cell_type.lower() == 'lstm':    
                self.rnn = nn.LSTM(input_size = self.input_rnn, hidden_size = self._hidden_size, 
                                    num_layers = self._n_layers, dropout = self._dropout, batch_first = True, 
                                    bidirectional = self._bidirectional).to(self.device)
            else:
                raise ValueError("Value of the parameter should be 'GRU' or 'LSTM'")

            # Tuning batch normalisation and fully connected layer depending on directionality of RNN
            if self._bidirectional:
                self.directions = 2
            else:
                self.directions = 1

            if self._normalisation == 'batch':
                self.norm0 = nn.BatchNorm1d(self.input_size).to(self.device)
                self.norm1 = nn.BatchNorm1d(self.hidden_size*self.directions).to(self.device)
            elif self._normalisation == 'layer':
                self.norm0 = nn.LayerNorm(self.input_size).to(self.device)
                self.norm1 = nn.LayerNorm(self.hidden_size*self.directions).to(self.device)
            elif self._normalisation is None:
                pass
            else:
                raise NotImplementedError('Normalisation should be batch, layer or None.')
            
            # Builds hidden state processing mechanism
            if self._hidden_states_processor is not None:
                if self._hidden_states_processor == 'attention':
                    self._hidden_states_processor = SelfAttention()
                elif self._hidden_states_processor == 'internal_processing':
                    self._hidden_states_processor = InternalProcessing()
                self._hidden_states_processor.build()
            print('Built RSNN model!')
            
        else:
            # Layers, dropout and non-linear activation functions are instantiated
            self.mlp = nn.Sequential(nn.Linear(self._input_size, self._hidden_size),
                                     nn.Dropout(p = 0.05),
                                     nn.LeakyReLU(),
                                     nn.Linear(self._hidden_size,self._output_size),
                                     nn.Dropout(p = 0.05),
                                     nn.LeakyReLU()).to(self.device)
            
            print('Built SNN MLP model!')
            
        if self._dist_fn != 'cos':
            # Linear layer after distance estimation
            self.dist_fc = nn.Linear(self._hidden_size*self.directions,1).to(self.device)

        # Weight initialisation
        if self._initialisation_process is not None:
            self.initialise_weights()
            
        # Get params and register optimiser
        info, params = self.get_model_params()
        
        self.optimiser = optim.Adam(params, lr = self._learning_rate, weight_decay = self._weight_decay)

    def forward_once(self, inputs, inputs_lens = None):
        """
        Performs the forward pass for the arms of the Siamese Neural Network.
        
        : inputs (torch.Tensor):
        : inputs_lens (torch.Tensor, optional):
        
        """
        # Determination of batch size
        batch_size = inputs.size(0)
    
        # Initialisation of hidden states
        h0 = self.initialise_hidden(batch_size)
    
        if self._embedding_dimensions is not None:
            embedded_inputs = self.embedding(inputs)
            inputs = self.embedding_dropout(embedded_inputs)
            
            if self._normalisation is not None: 
                inputs = self.norm0(inputs)
                
        inputs_packed = tnnur.pack_padded_sequence(inputs, inputs_lens, batch_first = True, enforce_sorted = False)
        
        # RNN cell type
        if self._cell_type.lower() == 'lstm':
            output_packed, hidden = self.rnn(inputs_packed, h0)
        elif self._cell_type.lower() == 'gru':
            output_packed, hidden = self.rnn(inputs_packed)
        
        output, output_lens = tnnur.pad_packed_sequence(output_packed, batch_first = True, total_length = self._input_size)
        
        if self._hidden_states_processor is not None:
            # Applies relevant processing to last hidden states
            ht = self._hidden_states_processor(output)
        else:
            # Extract only last hidden state from last BiLSTM layer
            output_fw = output[:,-1,0:self._hidden_size]
            output_bw = output[:,0,self._hidden_size:]

            ht = torch.cat((output_fw, output_bw),-1)
       
        if self._normalisation is not None:
            ht = self.norm1(ht)
        
        return ht

    def forward(self, inputs1, inputs2, inputs1_lens = None, inputs2_lens = None, labels = None, predict = False): 
        """
        Performs the computation of the last hidden states of the two inputs in parallel.
        
        : inputs1 (torch.Tensor): tokenised input from the left-hand arm of the Siamese Neural Network
        : inputs2 (torch.Tensor): tokenised input form the right-hand arm of the Siamese Neural Network
        : inputs1_lens (np.array, optional): 
        : inputs2_lens (np.array, optional): 
        : labels (torch.Tensor, optional): class labels
        : predict (bool): enables display of performance metrics
        
        """
        
        if self._is_mlp:
            output1 = self.mlp(inputs1.float())
            output2 = self.mlp(inputs2.float())
        else: 
            output1 = self.forward_once(inputs1, inputs1_lens)
            output2 = self.forward_once(inputs2, inputs2_lens)
        
        # Energy function
        similarity = self.distance_layer(output1, output2, self._dist_fn)

        # Evaluation metrics
        if predict:
            metrics_calculator = CalculateMetrics(similarity, labels, is_regression = self._is_regression)
            metrics = metrics_calculator()
                
            return metrics, similarity
        else:
            return similarity
            
    def distance_layer(self, output1, output2, distance):
        """
        Energy function. Estimates the distance between the two outputs of the Siamese Neural Network according to a given distance metric.
        
        : output1 (torch.Tensor):
        : output2 (torch.Tensor):
        : distance (str): metric to calculate the distance between outputs.
        
        """
        
        # Check definition in http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        # Redefined with L1 as per http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf
        if self._loss == 'contrastive loss':
            distance = 'l1'
        
        try:
            assert distance in ['cos','l1','l2']
        except:
            print('Similarity metric must be cosine, L1 or L2 distances')
               
        if distance == 'cos':
            distance = F.cosine_similarity(output1, output2, dim = -1, eps = self._eps)
        elif distance == 'l1':
            distance = self.dist_fc(torch.abs(output1 - output2)).squeeze(1)
        elif distance == 'l2':
            distance = self.dist_fc(torch.abs(output1 - output2) ** 2).squeeze(1)
        
        if self._loss != 'contrastive loss':
            # Passing the distance vector through a similarity function to squish it between 0 and 1
            if self._similarity_fn == 'sigmoid':
                distances = torch.sigmoid(distance)
            elif self._similarity_fn == 'exp':
                distances = torch.exp(-torch.abs(distance))
            elif self._similarity_fn == 'clamp':
                distances = torch.clamp(distance, min = 0.0)
            elif self._similarity_fn is None:
                distances = distance
                
        return distances
    
    def get_loss(self, outputs, labels):
        """
        Computes the specified loss function.
        
        : outputs (torch.Tensor):
        : labels (torch.Tensor):
        
        """
        try:
            assert self._loss in ['mse','mae','l1','l2','huber','logcosh','bce','contrastive'], 'Specify correct loss function'
        except AssertionError as msg:
                sys.exit(msg)
        
        if self._loss == 'mse' or self._loss == 'l2':
            # L2 loss function
            self.criterion = lambda x,y: torch.pow(x - y,2)
            loss = self.criterion(outputs, labels)
            
            # Adding up the losses (L1 loss) or meaning the losses (MAE loss)
            # of all batch instances
            if self._loss == 'mse':
                loss = torch.mean(loss)
            elif self._loss == 'l2':
                loss = torch.sum(loss)
            
        elif self._loss == 'mae' or self._loss == 'l1':
            # L1 loss function
            self.criterion = lambda x,y: torch.abs(x - y)
            loss = self.criterion(outputs, labels)
            
            # Adding up the losses (L1 loss) or meaning the losses (MAE loss)
            # of all batch instances
            if self._loss == 'mae':
                loss = torch.mean(loss)
            elif self._loss == 'l1':
                loss = torch.sum(loss)
        
        elif self._loss == 'huber':
            # Huber loss function
            self.criterion = torch.nn.SmoothL1Loss()
            loss = self.criterion(outputs.float(), labels.float())
            
            # Adding up the losses of all batch instances
            loss = torch.mean(loss)
        
        elif self._loss == 'logcosh':
            # Log-cosh loss function
            loss = torch.log(torch.cosh(outputs.float() - labels.float()))
            
            # Adding up the losses of all batch instances
            loss = torch.sum(loss)           
        
        elif self._loss == 'bce':
            if self._dist_fn == 'cos':
                self.criterion =  nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.BCELoss()
            loss = self.criterion(outputs.float(), labels.float())
        
        elif self._loss == 'contrastive':
            margin = 1
            loss = torch.sum((1-labels) * torch.pow(outputs,2)+ labels * torch.pow(torch.clamp(margin - outputs, min = 0.0),2))

        return loss

    def get_model_params(self):
        """
        
        """
                
        params = []
        total_size = 0
        
        def multiply_iter(parameter_list):
            """
            
            : parameter_list ():
            """
            
            out = 1
            for parameter in parameter_list:
                out *= parameter
            return out

        for parameter in self.parameters():
            if parameter.requires_grad:
                params.append(parameter)
                total_size += multiply_iter(parameter.size())

        return '{}\nparam size: {:,}\n'.format(self, total_size), params
    
    def initialise_hidden(self, batch_size):
        """
        Initialisation of hidden states and cell states of LSTM to zero.
        Creation of two new tensors with sizes n_layers x batch_size x n_hidden.
        
        : batch_size (int): specified size of the batch
        
        """
        
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self._n_layers*self.directions, batch_size, self._hidden_size).zero_().to(self.device),
        weight.new(self._n_layers*self.directions, batch_size, self._hidden_size).zero_().to(self.device))
                  
        return hidden 
    
    def initialise_weights(self):
        """
        Initialisation of weights and biases for the embedding layer and the RNN layers.
        
        """       
        
        def initialise_process(param):
        
            """
            Initialises weights of a given parameter following either Xavier or Kaiming uniform or normal processes.
                
            : param (torch.Tensor):
            
            """
            
            if self._initialisation_process == 'xavier_uniform':
                tnni.xavier_uniform_(param.data)
            elif self._initialisation_process == 'xavier_normal':
                tnni.xavier_normal_(param.data)
            elif self._initialisation_process == 'kaiming_uniform':
                tnni.kaiming_uniform_(param.data)
            elif self._initialisation_process == 'kaiming_normal':
                tnni.kaiming_normal_(param.data)
                
        if self._initialisation_process is not None:
            for m in self.modules():
                # Embedding
                if type(m) is nn.Embedding:
                    tnni.normal_(self.embedding.weight)
                # RNN
                elif type(m) in [nn.GRU, nn.LSTM, nn.RNN]:  
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            initialise_process(param)
                            #torch.nn.init.kaiming_normal_(param.data)
                        elif 'weight_hh' in name:
                            tnni.orthogonal_(param.data)
                        elif 'bias' in name:
                            # Bias initialised with zero will get the bias from
                            # the forget gate
                            param.data.fill_(0.0)
                            param.data[self._hidden_size:self.directions*self._hidden_size].fill_(1.0)
                # Attention linear layer
                elif type(m) is nn.Linear:
                    for name, param in m.named_parameters():
                        if 'weight' in name:
                            initialise_process(param.data)
                        elif 'bias' in name:
                            param.data.normal_()

"""
Implements attention as described in 'Indentifying Structure-Property Relationships through SMILES Syntax 
Analysis with Self-Attention Mechanism'

"""
                        
class SelfAttention(nn.Module):    
    def __init__(self, **kwargs):
        """
        Initialiser.
        
        : expansion_size (int): adjustable hyperparameter. Intermediate dimension to yield attention weights
        : hidden_size (int): hidden units after concatenating the hidden units for both directions
        : attention_layers (int):
        : seqlen (int): length of the padded SMILES string
        
        """
        
        super(SelfAttention, self).__init__()
        
        set_seed()

        self._hidden_size = kwargs.get('hidden_size',128)
        self._expansion_size = kwargs.get('expansion_size',1024)
        self._attention_layers = kwargs.get('attention_layers',512)
        self._activation_fn = kwargs.get('activation_fn','leaky ReLU')
        self._seqlen = kwargs.get('seqlen',150)
        
        # Device agnostic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def build(self):
        """
        Generates matrixes and layers to implement internal processing.
        
        """
        
        # Defining the layers
        self.w1 = nn.Linear(self._hidden_size*2, self._expansion_size, bias = False).to(self.device)
        self.tanh = nn.Tanh()
        self.w2 = nn.Linear(self._expansion_size, self._attention_layers, bias = False).to(self.device)
        self.softmax = nn.Softmax(dim = 2)
        self.fc = nn.Linear(self._attention_layers, 1).to(self.device)
        
        if self._activation_fn == 'tanh' or isinstance(self._activation_fn, torch.nn.modules.activation.Tanh):
            self._activation_fn = nn.Tanh()
        elif self._activation_fn == 'sigmoid' or isinstance(self._activation_fn, torch.nn.modules.activation.Sigmoid):
            self._activation_fn = nn.Sigmoid()
        elif self._activation_fn == 'leaky ReLU' or isinstance(self._activation_fn, torch.nn.modules.activation.LeakyReLU):
            self._activation_fn = nn.LeakyReLU()
        else:
            raise NotImplementedError('Non-linear activation function must be "tanh", "sigmoid" or "leaky ReLU"')
            
        # Passing it onto the relevant device
        self._activation_fn = self._activation_fn.to(self.device)
    
    def forward(self, output):
        """
        Forward pass of the bidirectional hidden states.
        
        : output (torch.Tensor): matrix with hidden states and cell states
        
        """
        
        hidden_states = self.extract_hidden_states(output)
        
        # Obtaining the attention weights
        weighted_states = self.w1(hidden_states)
        activated_states = self.tanh(weighted_states)
        score_weights = self.w2(activated_states)
        attention_weights = self.softmax(score_weights)
        
        # Applying attention to the matrix with hidden states
        attentional_vector = torch.bmm(torch.transpose(attention_weights,2,1),hidden_states)   
        attentional_vector = self.fc(torch.transpose(attentional_vector,2,1)).squeeze(2)
        attentional_vector = self._activation_fn(attentional_vector)
    
        return attentional_vector
    
    def extract_hidden_states(self, output):
        """
        Extracts last hidden states from both directions.
        
        : output (torch.Tensor): matrix with hidden states and cell states
        
        """
        
        # Extracting the forward and backward hidden states from the last BiLSTM layer
        # output (batch_size, sequence length, 2 * hidden dim)
        output_fw = output[:,:,0:self._hidden_size]
        output_bw = output[:,:,self._hidden_size:]
        
        hidden_states = torch.cat((output_fw, output_bw),-1)
        
        return hidden_states
    
"""
Implements internal processing (matrices with weights initialised N~(0,1))

"""
     
class InternalProcessing(nn.Module):
    def __init__(self, **kwargs):
        """
        
        : expansion_size (int):
        : hidden_size (int):
        
        """
        
        super(InternalProcessing, self).__init__()
        
        set_seed()
        
        self._hidden_size = kwargs.get('hidden_size',128)
        self._expansion_size = kwargs.get('expansion_size',128)
        self._activation_fn = kwargs.get('activation_fn','sigmoid')
        self._seqlen = kwargs.get('seqlen',150)
                
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def build(self, weight = 0.5):
        """
        Generates matrixes and layers to implement internal processing.
        
        : batch_size (int):
        
        """
        
        self.weight = weight
        
        # Defining weighting matrixes
        self.processing_fw = torch.randn((self._hidden_size, self._expansion_size), requires_grad = True).to(self.device)
        self.processing_bw = torch.randn((self._hidden_size, self._expansion_size), requires_grad = True).to(self.device)
        self.processing_last_ht = torch.randn((self._hidden_size*2, self._hidden_size*2), requires_grad = True).to(self.device)
        
        # These will only be applied to the intermediate hidden states
        self.linear_fw = nn.Linear(self._seqlen - 1, 1).to(self.device)
        self.linear_bw = nn.Linear(self._seqlen - 1, 1).to(self.device)
        
        self.compression = torch.randn((self._expansion_size*2, self._hidden_size*2), requires_grad = True).to(self.device)
        
        if self._activation_fn == 'tanh' or isinstance(self._activation_fn, torch.nn.modules.activation.Tanh):
            self._activation_fn = nn.Tanh()
        elif self._activation_fn == 'sigmoid' or isinstance(self._activation_fn, torch.nn.modules.activation.Sigmoid):
            self._activation_fn = nn.Sigmoid()
        elif self._activation_fn == 'leaky ReLU' or isinstance(self._activation_fn, torch.nn.modules.activation.LeakyReLU):
            self._activation_fn = nn.LeakyReLU()
        else:
            raise ValueError('Non-linear activation function must be "tanh", "sigmoid" or "leaky ReLU"')
            
        # Passing it onto the relevant device
        self._activation_fn = self._activation_fn.to(self.device)
        
    def forward(self, output):
        """
        Forward pass of the bidirectional hidden states.
        
        : output (torch.Tensor): matrix with hidden states and cell states
        
        """
        
        last_ht, output_fw_intermediate, output_bw_intermediate = self.extract_hidden_states(output)
        
        # Intermediate hidden state internal processing
        output_reduced_fw = self.implement_processing(output_fw_intermediate, self.processing_fw)
        output_reduced_bw = self.implement_processing(output_bw_intermediate, self.processing_bw, forward = False)
        
        # Concatenation of intermediate hidden state outputs
        output_reduced = torch.cat((output_reduced_fw, output_reduced_bw),-1)
        
        # Reduction of concatenated outputs dimension to match last hidden state
        intermediate_ht = torch.matmul(output_reduced, self.compression)
        
        # Last hidden state internal processing
        last_ht = torch.matmul(last_ht, self.processing_last_ht)
        last_ht = (1-torch.tensor(self.weight))*last_ht
        
        # Weighted sum of hidden states
        ht = last_ht.add(intermediate_ht)
        
        return ht 
    
    def extract_hidden_states(self, output):
        """
        Extracts intermediate and last hidden states
        
        : output (torch.Tensor):
        
        """
        # Intermediate hidden states
        output_fw_intermediate = output[:,:-1,0:self._hidden_size]
        output_bw_intermediate = output[:,1:,self._hidden_size:] 
        
        # Last hidden states
        output_fw = output[:,-1,0:self._hidden_size]
        output_bw = output[:,0,self._hidden_size:]
        last_ht = torch.cat((output_fw, output_bw), -1)
        
        return last_ht, output_fw_intermediate, output_bw_intermediate
    
    def implement_processing(self, output_intermediate, processing_matrix, forward = True):
        """
        Carries out internal processing for each hidden state output direction.
        
        : output_intermediate (torch.Tensor): unidirectional intermediate hidden states
        : processing_matrix (torch.Tensor): matrix with weights ~ N(0,1)
        : forward (bool): toggle to regulate output direction
        
        """
        
        # Attention implementation
        output_intermediate = torch.matmul(output_intermediate, processing_matrix)
        output_intermediate = torch.transpose(output_intermediate,2,1)
            
        # Linear layer output reduction 
        # from [batch, hidden_size, input_size - 1] to [batch, hidden_size,1]
        # different linear layers required to ensure good performance
        if forward:
            output_reduced = self.linear_fw(output_intermediate).squeeze(2)
        else:
            output_reduced = self.linear_bw(output_intermediate).squeeze(2)
        
        output_reduced = self._activation_fn(output_reduced)
        
        return output_reduced