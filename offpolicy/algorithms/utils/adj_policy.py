import torch
import torch.nn as nn
import torch.nn.functional as F
from offpolicy.utils.util import init, to_torch

class AdjPolicy(nn.Module):
    """
    Computes total Q values given agent q values and global states.
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param num_agents: (int) number of agents in env
    :param cent_obs_dim: (int) dimension of the centralized state
    :param device: (torch.Device) torch device on which to do computations.
    :param multidiscrete_list: (list) list of each action dimension if action space is multidiscrete
    """
    def __init__(self, args, obs_dim ,hidden_dim, device, use_ReLU):
        super(AdjPolicy, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.num_variable = args.num_agents
        self.num_factor = args.num_factor
        self._use_orthogonal = args.use_orthogonal
        self.gain = args.gain
        
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        def act_init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),gain=self.gain)
        # 2 layer hypernets: output dimensions are same as above case
        #nn.ReLU()
        self.hyper_input_dim = self.hidden_dim * self.num_variable
        '''self.hyper_obs = Hypernet(
            input_dim=obs_dim, hidden_dim=self.hidden_dim,
            main_input_dim=obs_dim, main_output_dim=self.hidden_dim
        )'''
        self.mlp_obs = init_(nn.Linear(obs_dim, self.hidden_dim))  
        
        self.hyper_w1 = Hypernet(
            input_dim=self.hyper_input_dim, hidden_dim=self.hidden_dim*self.hidden_dim//2,
            main_input_dim=self.hidden_dim, main_output_dim=self.hidden_dim//2
        )
  
        self.hyper_w2 = Hypernet(
            input_dim=self.hyper_input_dim, hidden_dim=self.num_factor*self.hidden_dim//2,
            main_input_dim=self.hidden_dim//2, main_output_dim=self.num_factor
        )
        
        self.hyper_b1 = init_(nn.Linear(self.hyper_input_dim, self.num_variable * self.hidden_dim//2))  
        
        self.hyper_b2 = Hypernet(
            input_dim=self.hyper_input_dim, hidden_dim=self.num_variable * self.hidden_dim//2,
            main_input_dim=self.num_variable, main_output_dim=self.num_factor
        )
        
        self.bn1 = nn.BatchNorm1d(self.num_variable)
        

    def forward(self, obs, rnn_obs, use_adj_init,dones):
        """
         Computes Q_tot using the individual agent q values and global state.
         :param agent_q_inps: (torch.Tensor) individual agent q values
         :param states: (torch.Tensor) state input to the hypernetworks.
         :return Q_tot: (torch.Tensor) computed Q_tot values
         """
        batch_size = obs.size(0)
        
        h_obs = self.mlp_obs(obs)
        
        global_obs = h_obs.reshape(batch_size,-1)
        w1 = self.hyper_w1(global_obs)
        b1 = self.hyper_b1(global_obs).reshape(batch_size,self.num_variable,-1)
        hidden_layer = torch.tanh(torch.matmul(rnn_obs, w1) + b1)
        
        w2 = self.hyper_w2(global_obs)
        b2 = self.hyper_b2(global_obs).reshape(batch_size,self.num_variable,-1)
        #use_adj_init = False
        if use_adj_init:
            out = torch.matmul(hidden_layer, w2*0.01) + b2 - b2.detach()
        else:
            out = torch.matmul(hidden_layer, w2) + b2
        
        out_pre = torch.where(dones,-1e10*torch.ones_like(out),out)
        out = F.softmax(out_pre,dim=1)  
        
        log_probs = F.log_softmax(out_pre,dim=1)  

        
        return out, log_probs

      
class Hypernet(nn.Module):
    def __init__(self, input_dim, hidden_dim, main_input_dim, main_output_dim):
        super(Hypernet, self).__init__()

        # the output dim of the hypernet
        output_dim = main_input_dim * main_output_dim
        # the output of the hypernet will be reshaped to [main_input_dim, main_output_dim]
        self.main_input_dim = main_input_dim
        self.main_output_dim = main_output_dim
        init_method = nn.init.orthogonal_
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.multihead_nn = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, output_dim)),
        )

    def forward(self, x):
        # [...,  main_output_dim + main_output_dim + ... + main_output_dim]
        # [bs, main_input_dim, n_heads * main_output_dim]
        #import pdb;pdb.set_trace()
        return self.multihead_nn(x).view([-1, self.main_input_dim, self.main_output_dim])
