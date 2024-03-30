import torch.nn as nn
import torch
from offpolicy.utils.util import adj_init

class RNNBase(nn.Module):
    """ Identical to rnn_agent, but does not compute value/probability for each action, only the hidden state. """
    def __init__(self, args, input_shape, hidden_size, out_shape, device=torch.device("cuda:0")):
        nn.Module.__init__(self)
        self.args = args
        self.use_ReLU = self.args.use_ReLU
        self.use_orthogonal = self.args.use_orthogonal
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.kaiming_uniform_][self.use_orthogonal]
        def init_(m):
            return adj_init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.fc1 = nn.Sequential(nn.Linear(input_shape, hidden_size), active_func)
        #self.fc1 = nn.Sequential(init_(nn.Linear(input_shape, hidden_size)), active_func)
        #self.fc1 = nn.Sequential(init_(nn.Linear(input_shape, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.rnn = nn.GRU(hidden_size, out_shape)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self.use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        #self.norm = nn.LayerNorm(out_shape)
        self.to(device) 

    def forward(self, inputs, rnn_states):
        no_sequence = False
        if len(inputs.shape) == 2:

            inputs = inputs[None]
        if len(rnn_states.shape) == 2:
            rnn_states = rnn_states[None]
        x = self.fc1(inputs)
        self.rnn.flatten_parameters()
        x, hid = self.rnn(x, rnn_states)
        #x = self.norm(x)
        return x, hid[0,:,:], no_sequence
