import os
from collections import defaultdict
from torch import randperm
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch._utils import _accumulate
from dataset import TwitterFileArchiveDataset
from gru import GRUCell
from utils import init_weights, argmax, cuda, variable, get_sequence_from_indices
from dataset import Vocab
import random
import matplotlib.pyplot as plt
import plotGraph 
    
class NeuralLanguageModel(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, init_token, eos_token, teacher_forcing=0.7):
        super(NeuralLanguageModel,self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.init_token = init_token
        self.eos_token = eos_token
        self.vocab_size = vocab_size
        ##############################
        ### Insert your code below ###
        # create an embedding layer, a GRU cell, and the output projection layer
        ##############################
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.bi_gru_doc = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=True)
        self.bi_gru_query = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=True)
        
        #self.reverse_gru = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=False)
        ###############################
        ### Insert your code above ####
        ###############################

    def cell_zero_state(self, batch_size):
        """
        Create an initial hidden state of zeros
        :param batch_size: the batch size
        :return: A tensor of zeros of the shape of (batch_size x hidden_size)
        """
        weight = next(self.parameters()).data
        hidden = Variable(weight.new(1,batch_size, self.hidden_size).zero_())
        return hidden

    def forward(self, document_input,query_input):
        
        document_embedding = self.embedding(document_input)
        query_embedding = self.embedding(query_input)
        
        document_output,document_hidden = self.gru_doc(document_embedding,d_hidden)
        query_output,query_hidden = self.gru_query(query_embedding, q_hidden)

        q_hidden = query_hidden
        d_hidden = document_hidden

        
        
        


    


        
