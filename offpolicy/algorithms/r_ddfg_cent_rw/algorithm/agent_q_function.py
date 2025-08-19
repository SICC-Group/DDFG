import torch
import torch.nn as nn
from offpolicy.utils.util import init, adj_init
from offpolicy.utils.util import to_torch
import torch.nn.functional as F


class AgentQFunction(nn.Module):
    """
    Individual agent q network (MLP).
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """
    def __init__(self, args, obs_dim, input_dim, num_orders, act_dim, device):
        super(AgentQFunction, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_ReLU = args.use_ReLU
        self.use_orthogonal = args.use_orthogonal
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        gain = args.gain
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),gain=gain)
        self.hidden_dim = input_dim
        self.num_orders = num_orders
        self.act_dim = act_dim

        self.output_layer = nn.Sequential(init_(nn.Linear(input_dim*num_orders,act_dim*num_orders)))
        self.to(device) 

    def forward(self, x, rnn_obs, no_sequence):
        """
        Compute q values for every action given observations and rnn states.
        :param x: (torch.Tensor) observations from which to compute q values.

        :return q_outs: (torch.Tensor) q values for every action
        """
        # make sure input is a torch tensor
        bs = x.shape[0]

        x = to_torch(x).to(**self.tpdv).reshape(bs*self.num_orders,-1)
         #[bs*order,obs_dim]
        #rnn_obs = rnn_obs.reshape(bs, self.num_orders,-1)
        rnn_obs = rnn_obs.reshape(bs, -1)
        
        #q_value = self.output_layer(rnn_obs).reshape(bs, -1)
        q_value = self.output_layer(rnn_obs)
        
        norm_q = torch.abs(q_value+ 1e-8) ** (1-1/self.num_orders) 

        q_value_norm = q_value / norm_q
        #q_value_norm = q_value
        #[bs,1, hidden_dim] * [bs,hidden_dim, order*act_dim] -> [bs,1,  order*act_dim]-> [bs,order*act_dim]
        
        if no_sequence:
                q_value_norm = q_value_norm[0, :, :]

        return q_value_norm


