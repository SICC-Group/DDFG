import torch
import torch.nn as nn
from offpolicy.utils.util import adj_init
from offpolicy.utils.util import to_torch
from offpolicy.algorithms.utils.mlp import MLPBase
from offpolicy.algorithms.utils.act import ACTLayer

class AgentQFunction(nn.Module):
    """
    Individual agent q network (MLP).
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """
    def __init__(self, args, input_dim, hidden_dim, act_dim, device):
        super(AgentQFunction, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_ReLU = args.use_ReLU
        self.use_orthogonal = args.use_orthogonal
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.kaiming_uniform_][self.use_orthogonal]
        gain = args.gain
        def init_(m):
            return adj_init(m, init_method, lambda x: nn.init.constant_(x, 0))
          
        '''self.input_layer = nn.Sequential(init_(nn.Linear(input_dim,hidden_dim)), active_func)
        self.output_layer = nn.Sequential(init_(nn.Linear(hidden_dim,act_dim)))'''
        self.output_layer = nn.Sequential(init_(nn.Linear(input_dim,act_dim)))
        self.to(device) 

    def forward(self, x, no_sequence):
        """
        Compute q values for every action given observations and rnn states.
        :param x: (torch.Tensor) observations from which to compute q values.

        :return q_outs: (torch.Tensor) q values for every action
        """
        # make sure input is a torch tensor
        x = to_torch(x).to(**self.tpdv)
       
        #x =self.input_layer(x)
        
        '''for layer in self.hidden_layers:
            x = layer(x)
        # pass outputs through linear layer'''
        q_value = self.output_layer(x)

        if no_sequence:
                q_value = q_value[0, :, :]
        return q_value
