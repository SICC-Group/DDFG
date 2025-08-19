import torch
import torch.nn as nn
from offpolicy.utils.util import to_torch
from offpolicy.algorithms.utils.mlp import MLPBase
from offpolicy.algorithms.utils.rnn import RNNBase
from offpolicy.algorithms.utils.act import ACTLayer

class AgentQFunction(nn.Module):
    """
    Individual agent q network (RNN).
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """
    def __init__(self, args, input_dim, act_dim, device):
        super(AgentQFunction, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self._use_rnn_layer = args.use_rnn_layer
        self._gain = args.gain
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)


        self.mlp = MLPBase(args, input_dim)

        self.q = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, gain=self._gain)

        self.to(device)

    def forward(self, obs):
        """
        Compute q values for every action given observations and rnn states.
        :param obs: (torch.Tensor) observations from which to compute q values.
        :param rnn_states: (torch.Tensor) rnn states with which to compute q values.

        :return q_outs: (torch.Tensor) q values for every action
        :return h_final: (torch.Tensor) new rnn states
        """
        obs = to_torch(obs).to(**self.tpdv)

        no_sequence = False
        if len(obs.shape) == 2:
            # this means we're just getting one output (no sequence)
            no_sequence = True
            obs = obs[None]
            # obs is now of shape (seq_len, batch_size, obs_dim)

        inp = obs

        rnn_outs = self.mlp(inp)

        #import pdb;pdb.set_trace()
        # pass outputs through linear layer
        q_outs = self.q(rnn_outs, no_sequence)

        return q_outs
  
   