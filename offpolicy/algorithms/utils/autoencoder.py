import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from offpolicy.utils.util import adj_init, get_clones, init

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,use_orthogonal, use_ReLU,device):
        super(Autoencoder, self).__init__()
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        #active_func = nn.Sigmoid()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])
        #import pdb;pdb.set_trace()
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),gain=gain)
        
        '''self.encoder = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_dim[0])), active_func ,
            init_(nn.Linear(hidden_dim[0], hidden_dim[1])), active_func ,
            init_(nn.Linear(hidden_dim[1], output_dim)))
        self.decoder = nn.Sequential(
            init_(nn.Linear(output_dim, hidden_dim[1])),active_func, 
            init_(nn.Linear(hidden_dim[1], hidden_dim[0])), active_func ,
            init_(nn.Linear(hidden_dim[0], input_dim)))'''
        
        self.encoder = nn.Sequential(
            init_(nn.Linear(input_dim, output_dim)))
        '''self.encoder = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_dim)), active_func ,init_(nn.Linear(hidden_dim, output_dim)))'''
        '''self.decoder = nn.Sequential(
            init_(nn.Linear(output_dim, hidden_dim)),active_func, init_(nn.Linear(hidden_dim, input_dim)))'''

    def forward(self, x):
        embedding = self.encoder(x)
        '''output = self.decoder(embedding)'''
        return embedding, None

