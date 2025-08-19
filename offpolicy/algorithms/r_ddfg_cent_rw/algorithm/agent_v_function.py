import torch
import torch.nn as nn
from offpolicy.utils.util import init, adj_init
from offpolicy.utils.util import to_torch
<<<<<<< HEAD
=======
import torch.nn.functional as F
>>>>>>> 53301a4 (20250819)


class AgentVFunction(nn.Module):
    """
    Individual agent q network (MLP).
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """
<<<<<<< HEAD
    def __init__(self, args, input_dim, hidden_dim, act_dim, device):
=======
    def __init__(self, args, input_dim, state_dim, num_orders, act_dim, device):
>>>>>>> 53301a4 (20250819)
        super(AgentVFunction, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_ReLU = args.use_ReLU
        self.use_orthogonal = args.use_orthogonal
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        gain = args.gain
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),gain=gain)
<<<<<<< HEAD
        self.output_layer = nn.Sequential(init_(nn.Linear(input_dim,act_dim)))
        self.to(device) 

    def forward(self, x, no_sequence):
=======
        self.hidden_dim = input_dim
        self.state_dim = state_dim
        self.num_orders = num_orders

        self.output_layer = nn.Sequential(init_(nn.Linear(input_dim*num_orders,act_dim)))
       
        self.to(device) 

    def forward(self, x, state, no_sequence):
>>>>>>> 53301a4 (20250819)
        """
        Compute q values for every action given observations and rnn states.
        :param x: (torch.Tensor) observations from which to compute q values.

        :return q_outs: (torch.Tensor) q values for every action
        """
        # make sure input is a torch tensor
<<<<<<< HEAD
        x = to_torch(x).to(**self.tpdv)

        # pass outputs through linear layer
        v_value = self.output_layer(x)
        if no_sequence:
                v_value = q_value[0, :, :]

        return v_value
=======
        bs = x.shape[0]
        #self.num_orders = num_orders
        #x = to_torch(x).to(**self.tpdv).reshape(bs, self.num_orders,-1)
        x = to_torch(x).to(**self.tpdv).reshape(bs,-1)
        
        v_value = self.output_layer(x)
        #v_value = self.output_layer(torch.cat([x.sum(1),self.hidden_layer(state)],dim=1))

        if no_sequence:
                v_value = v_value[0, :, :]


        return v_value

>>>>>>> 53301a4 (20250819)
