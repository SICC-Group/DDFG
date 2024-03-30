import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from offpolicy.utils.util import init, to_torch


class QMixerCentralFF(nn.Module):
    def __init__(self, args, num_agents, state_dim, device):
        super(QMixerCentralFF, self).__init__()

        self.args = args
        self.tpdv = dict(dtype=th.float32, device=device)
        self.n_agents = num_agents
        self.state_dim = state_dim

        self.input_dim = self.n_agents * self.args.central_action_embed + self.state_dim
        self.embed_dim = args.central_mixing_embed_dim

        non_lin = nn.ReLU

        self.net = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, 1))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               non_lin(),
                               nn.Linear(self.embed_dim, 1))
        self.to(device)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(1)
        states =  to_torch(states).to(**self.tpdv).reshape(-1, bs, self.state_dim)
        agent_qs = to_torch(agent_qs).to(**self.tpdv).reshape(-1, bs, self.n_agents * self.args.central_action_embed)

        inputs = th.cat([states, agent_qs], dim=2)

        advs = self.net(inputs)
        vs = self.V(states)

        y = advs + vs

        q_tot = y.view(-1,bs, 1, 1)
        return q_tot
