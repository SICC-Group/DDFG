import numpy as np
import torch
import time
import torch_scatter
import copy
from offpolicy.algorithms.casec.algorithm.obs_reward_encoder import ObsRewardEncoder
from offpolicy.algorithms.casec.algorithm.agent_q_function import AgentQFunction
from offpolicy.algorithms.casec.algorithm.adj_generator import Adj_Generator
from torch.distributions import Categorical, OneHotCategorical
from offpolicy.utils.util import get_dim_from_space, is_discrete, is_multidiscrete, make_onehot, DecayThenFlatSchedule, avail_choose, to_torch, to_numpy
from offpolicy.algorithms.base.mlp_policy import MLPPolicy
from offpolicy.algorithms.casec.algorithm.rnn import RNNBase

class R_CASECPolicy(MLPPolicy):
    """
    QMIX/VDN Policy Class to compute Q-values and actions (MLP). See parent class for details.
    :param config: (dict) contains information about hyperparameters and algorithm configuration
    :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
    :param train: (bool) whether the policy will be trained.
    """
    def __init__(self, config, policy_config, train=True):
        self.args = config["args"]
        self.device = config['device']
        self.obs_space = policy_config["obs_space"]
        self.n_agents = config["num_agents"]
        self.num_factor = self.args.num_factor
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.act_space)
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        self.central_obs_dim = policy_config["cent_obs_dim"]
        self.discrete_action = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)
        self.hidden_size = self.args.hidden_size
        self.lamda = self.args.lamda
        if self.args.prev_act_inp:
            # this is only local information so the agent can act decentralized
            self.rnn_network_input_dim = self.obs_dim + self.act_dim
        else:
            self.rnn_network_input_dim = self.obs_dim
        self.rnn_out_dim = self.args.hidden_size
        self.rnn_hidden_size = self.args.hidden_size
        self.independent_p_q = self.args.independent_p_q
        self.q_hidden_size = [32,64]
        self.highest_orders = self.args.highest_orders
        self.use_action_repr = self.args.use_action_repr
        self.pair_rnn_hidden_dim = self.args.pair_rnn_hidden_dim
        self.p_lr = self.args.p_lr
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        if self.independent_p_q:
            self.q_network_input_dim = [self.rnn_out_dim,self.pair_rnn_hidden_dim * 2]
        else:
            self.q_network_input_dim = [self.rnn_out_dim,self.rnn_out_dim * 2]
        if self.use_action_repr:
            self.q_out_dim = [self.act_dim,2 * self.args.action_latent_dim]
        else:
            self.q_out_dim = [self.act_dim,self.act_dim*self.act_dim]
            
        self.zeros = torch.zeros([1, self.n_agents, self.n_agents]).to(self.device)
        self.eye2 = torch.eye(self.n_agents).bool()
        self.eye2 = self.eye2.to(self.device)   
        self.action_repr = torch.ones(self.act_dim, self.args.action_latent_dim).to(self.device)
        input_i = self.action_repr.unsqueeze(1).repeat(1, self.act_dim, 1)
        input_j = self.action_repr.unsqueeze(0).repeat(self.act_dim, 1, 1)
        self.p_action_repr = torch.cat([input_i, input_j], dim=-1).view(self.act_dim * self.act_dim,-1).t().unsqueeze(0)
        wo_diag = torch.ones(self.n_agents, self.n_agents)
        diag = torch.ones(self.n_agents)
        diag = torch.diag(diag, 0)
        self.wo_diag = (wo_diag - diag).to(self.device).unsqueeze(0)
        self.edges_from = torch.tensor([[i] * self.n_agents for i in range(self.n_agents)]).view(-1).unsqueeze(0).to(self.device)
        self.edges_to = torch.tensor([[i for i in range(self.n_agents)] * self.n_agents]).view(-1).unsqueeze(0).to(self.device)
        self.message = torch.zeros(2, 1, self.act_dim).to(self.device)
        self.action_repr = torch.ones(self.act_dim, self.args.action_latent_dim).to(self.device)
        self.action_encoder = ObsRewardEncoder(self.args,self.n_agents,self.act_dim,self.central_obs_dim,self.obs_dim, self.device)
        # Local recurrent q network for the agent
        self.rnn_network = RNNBase(self.args, self.rnn_network_input_dim, self.rnn_hidden_size, self.rnn_out_dim, self.device)
        self.q_network = {num_orders :AgentQFunction(self.args, self.q_network_input_dim[num_orders-1], self.q_hidden_size[num_orders-1], self.q_out_dim[num_orders-1], self.device) for num_orders in range(1,self.highest_orders+1)}
        if self.independent_p_q:
            self.p_rnn_network = RNNBase(self.args, self.rnn_network_input_dim, self.pair_rnn_hidden_dim, self.pair_rnn_hidden_dim, self.device)
            
        if train:
            self.exploration = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish, self.args.epsilon_anneal_time,decay="linear")
               
    def _variance(self, a, mask):
        mean = (a * mask).sum(-1) / mask.sum(-1)
        var = ((a - mean.unsqueeze(-1)) ** 2 * mask).sum(-1) / mask.sum(-1)
        return var
      
    def get_hidden_states(self, obs, prev_actions, rnn_states):
        if self.args.prev_act_inp:
            prev_action_batch = to_torch(prev_actions).type(torch.float32)
            input_batch = torch.cat((to_torch(obs), prev_action_batch), dim=-1)
        else:
            input_batch = to_torch(obs)
        
        q_batch, new_rnn_states, no_sequence = self.rnn_network(input_batch.to(**self.tpdv), to_torch(rnn_states).to(**self.tpdv))
        return q_batch,new_rnn_states,no_sequence
      
    def get_p_hidden_states(self, obs, prev_actions, rnn_states):
        if self.args.prev_act_inp:
            prev_action_batch = to_torch(prev_actions).type(torch.float32)
            input_batch = torch.cat((to_torch(obs), prev_action_batch), dim=-1)
        else:
            input_batch = to_torch(obs)
        
        q_batch, new_rnn_states, no_sequence = self.p_rnn_network(input_batch.to(**self.tpdv), to_torch(rnn_states).to(**self.tpdv))
        return q_batch,new_rnn_states,no_sequence
      
    def get_rnn_batch(self,obs_batch,batch_size,p_rnn_states=None,no_sequence=False):
        
        agent_outs = self.q_network[1](obs_batch,no_sequence).view(-1, self.n_agents, self.act_dim)
        f_i = agent_outs.clone()
        
        if self.independent_p_q:
            p_hidden_states = p_rnn_states.view(batch_size,self.n_agents, -1)
        else:
            p_hidden_states = obs_batch.clone().view(batch_size, self.n_agents, -1)
        input_i = p_hidden_states.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        input_j = p_hidden_states.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
        if self.independent_p_q:
            inputs = torch.cat([input_i, input_j], dim=-1).view(-1, self.n_agents*self.n_agents, 2 * self.args.pair_rnn_hidden_dim)
        else:
            inputs = torch.cat([input_i, input_j], dim=-1).view(-1, self.n_agents*self.n_agents, 2 * self.args.hidden_size)
        history_cos_similarity = self.zeros.repeat(batch_size, 1, 1)
        if self.use_action_repr:
            key = self.q_network[2](inputs,no_sequence).view(batch_size, self.n_agents*self.n_agents, -1)
            f_ij = torch.bmm(key, self.p_action_repr.repeat(batch_size, 1, 1)) / self.args.action_latent_dim / 2
        else:
            f_ij = self.q_network[2](inputs,no_sequence)
        
        f_ij = f_ij.view(batch_size, self.n_agents, self.n_agents, self.act_dim, self.act_dim)
        f_ij = (f_ij + f_ij.permute(0, 2, 1, 4, 3).detach()) / 2.
        
        f_i_expand_j = f_i.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_agents, 1, self.act_dim, 1)
        f_i_expand_i = f_i.unsqueeze(dim=2).unsqueeze(dim=-1).repeat(1, 1, self.n_agents, 1, self.act_dim)
        q_ij = f_i_expand_i.detach() + f_i_expand_j.detach() + f_ij

        f_ij[:, self.eye2] = 0
        q_ij[:, self.eye2] = 0
        
        return f_i, f_ij, q_ij, history_cos_similarity
      
    def caller_ip_q(self, obs_batch, p_rnn_states=None,no_sequence=False):
        # Calculate the utilities of each agent i and the incremental matrix delta for each agent pair (i&j).
        # (bs,n,|A|), (bs,n,n,|A|,|A|) = (b,T,n,|A|)
        if len(obs_batch.shape) == 3:
            batch_size = obs_batch.shape[0]
        else:
            batch_size = 1
        f_i, delta_ij, q_ij, his_cos_similarity  = self.get_rnn_batch(obs_batch,batch_size,p_rnn_states,no_sequence)

        # return individual and pair-wise q function
        return f_i, delta_ij, q_ij * self.p_lr, his_cos_similarity
      
    def get_q_values(self, obs_batch, action_batch, p_rnn_states=None,no_sequence=False):
        """
        Computes q values using the given information.
        :param obs_batch: (np.ndarray) agent observations from which to compute q values
        :param action_batch: (np.ndarray) if not None, then only return the q values corresponding to actions in action_batch

        :return q_values: (torch.Tensor) computed q values
        """
        if len(obs_batch.shape) == 3:
            batch_size = obs_batch.shape[0]
        else:
            batch_size = 1
        f_i, delta_ij, q_ij, _  = self.get_rnn_batch(obs_batch,batch_size,p_rnn_states,no_sequence)
        # Gather individual Qs
        f_i_gather = torch.gather(f_i, index=action_batch, dim=-1)  # (bs,n,1)

        # Gather pairwise Qs
        agent_actions_gather_i = action_batch.unsqueeze(dim=2).unsqueeze(dim=-1).repeat(1, 1, self.n_agents, 1,self.act_dim)
        agent_actions_gather_j = action_batch.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_agents, 1, 1, 1)
        edge_attr = torch.gather(q_ij, index=agent_actions_gather_i, dim=-2)
        edge_attr = torch.gather(edge_attr, index=agent_actions_gather_j, dim=-1)
        edge_attr = edge_attr.squeeze()  # * self.adj  # (bs,n,n)

        agent_outs = f_i_gather.squeeze(dim=-1).mean(dim=-1) + edge_attr.sum(dim=-1).sum(dim=-1) / self.n_agents / (self.n_agents - 1) * self.p_lr

        agent_outs.unsqueeze_(dim=-1)
        return agent_outs, f_i, delta_ij, q_ij
        
      
    def max_sum(self, avail_actions, f_i=None, delta_ij=None, q_ij=None, his_cos_sim=None, atten_ij=None,
                target_delta_ij=None, target_q_ij=None, target_his_cos_sim=None, target_atten_ij=None,batch_size=None):
        # Calculate the utilities of each agent i and the incremental matrix delta for each agent pair (i&j).
        x, adj, edge_attr, q_ij = self.construction(f_i, delta_ij, q_ij, his_cos_sim, atten_ij,
                                                    target_delta_ij, target_q_ij, target_his_cos_sim, target_atten_ij,
                                                    available_actions=avail_actions,batch_size=batch_size)

        # (bs,n,|A|) = (b,n,|A|), (b,n,n), (b,E,|A|,|A|)
        x_out = self.MaxSum_faster(x.detach(), adj.detach(), q_ij.detach(), batch_size,available_actions=avail_actions)
        return x_out
      
    def construction(self, f_i, delta_ij, q_ij, his_cos_sim, atten_ij, target_delta_ij=None,
                     target_q_ij=None, target_his_cos_sim=None, target_atten_ij=None, available_actions=None,batch_size=None):
        # available_actions: (bs,n,|A|)

        available_actions_i = available_actions.detach().unsqueeze(dim=2).repeat(1, 1, self.n_agents, 1).view(-1, self.n_agents * self.n_agents, self.act_dim)
        available_actions_j = available_actions.detach().unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_agents, 1, self.act_dim, 1).view(-1, self.n_agents * self.n_agents, self.act_dim, self.act_dim)
        available_actions_ij = (available_actions_i.unsqueeze(dim=-1).repeat(1, 1, 1, self.act_dim) * available_actions_j).view(-1, self.n_agents * self.n_agents, self.act_dim * self.act_dim)
        x = f_i.clone()

        if target_q_ij is not None:
            indicator = self._variance(target_q_ij.detach().view(-1, self.n_agents * self.n_agents, self.act_dim, self.act_dim), available_actions_j)
            indicator = (indicator * available_actions_i).max(-1)[0]
        else:
            indicator = self._variance(q_ij.detach().view(-1, self.n_agents * self.n_agents, self.act_dim, self.act_dim), available_actions_j)
            indicator = (indicator * available_actions_i).max(-1)[0]
        adj_tensor = indicator.masked_fill(self.wo_diag.repeat(batch_size, 1, 1).view(-1, self.n_agents * self.n_agents) == 0, -99999999)

        adj_tensor_topk = torch.topk(adj_tensor, int(self.n_agents * self.n_agents * self.args.threshold // 2 * 2), dim=-1)[1]
        adj = self.zeros.repeat(batch_size, 1, 1).view(-1, self.n_agents * self.n_agents)
        adj.scatter_(1, adj_tensor_topk, 1)
        adj = adj.view(-1, self.n_agents, self.n_agents).detach()
        adj[:, self.eye2] = 1.

        return x, adj, None, q_ij * self.p_lr
      
    def MaxSum_faster(self, x, adj, q_ij, batch_size=None, available_actions=None, k=5):
        # (bs,n,|A|), (bs,n,n), (bs,n,n,|A|,|A|), (bs,n,|A|) -> (bs,n,|A|)
        adj[:, self.eye2] = 0.
        num_edges = int(adj[0].sum(-1).sum(-1))  # Samples in the batch should have the same number of edges
        edges_from = self.edges_from.repeat(x.shape[0], 1)[adj.view(-1, self.n_agents ** 2) == 1].view(-1, num_edges)
        edges_to = self.edges_to.repeat(x.shape[0], 1)[adj.view(-1, self.n_agents ** 2) == 1].view(-1, num_edges)
        nodes = torch.cat([edges_from, edges_to], dim=1)  # (bs,2|E|)

        x = x / self.n_agents
        q_ij = q_ij / num_edges

        q_ij_new = q_ij[adj == 1].view(-1, num_edges, self.act_dim, self.act_dim)
        # q_left_down = self.message.clone().unsqueeze(0).repeat(self.bs, 1, num_edges, 1)
        r_down_left = self.message.clone().unsqueeze(0).repeat(batch_size, 1, num_edges, 1)
        # (bs,2,|E|,|A|,|A|),(bs,2,|E|,|A|), (bs,2,|E|,|A|)

        # Unavailable actions have a utility of -inf, which propagates throughout message passing
        if available_actions is not None:
            x = x.masked_fill(available_actions == 0, -99999999)
            available_actions_new = torch.gather(available_actions, dim=1,index=nodes.unsqueeze(-1).repeat(1, 1, self.act_dim)).view(-1, 2, num_edges, self.act_dim)

        for _ in range(k):
            # Message from variable node i to function node g:
            q_left_down_sum = torch_scatter.scatter_add(src=r_down_left.view(-1, 2 * num_edges, self.act_dim), \
                    index=nodes, dim=1, dim_size=self.n_agents)
            q_left_down_sum += x
            q_left_down = torch.gather(q_left_down_sum, dim=1,\
                    index=nodes.unsqueeze(-1).repeat(1, 1, self.act_dim)).view(-1, 2, num_edges, self.act_dim)
            q_left_down -= r_down_left
            # Normalize
            if available_actions is not None:
                q_left_down -= (q_left_down * available_actions_new).sum(-1, keepdim=True) / available_actions_new.sum(-1, keepdim=True)
            else:
                q_left_down -= q_left_down.mean(dim=-1, keepdim=True)

            # Message from function node g to variable node i:
            r_down_left[:, 0] = (q_ij_new + q_left_down[:, 1].unsqueeze(-2)).max(dim=-1)[0]
            r_down_left[:, 1] = (q_ij_new + q_left_down[:, 0].unsqueeze(-1)).max(dim=-2)[0]

        # Calculate the z value
        z = torch_scatter.scatter_add(src=r_down_left.view(-1, 2 * num_edges, self.act_dim), \
                                      index=nodes, dim=1, dim_size=self.n_agents)
        z += x
        return z      
      
    def get_actions(self, obs_batch, available_actions=None, t_env=None, explore=False,p_rnn_states = None,no_sequence = False):
        if len(obs_batch.shape) == 3:
            batch_size = obs_batch.shape[0]
        else:
            batch_size = 1

        if len(available_actions.shape) == 2:
            available_actions = available_actions.unsqueeze(0).to(self.device)
        f_i, delta_ij, q_ij, his_cos_sim = self.get_rnn_batch(obs_batch,batch_size,p_rnn_states,no_sequence)
        agent_outputs = self.max_sum(available_actions, f_i=f_i, delta_ij=delta_ij, q_ij=q_ij,
                                     his_cos_sim=his_cos_sim.detach(),batch_size=batch_size)

        actions = agent_outputs.squeeze()
        # mask the available actions by giving -inf q values to unavailable actions
        if self.multidiscrete:
            onehot_actions = []
            for i in range(len(self.act_dim)):
                greedy_action = actions[i]
                if explore:
                    eps = self.exploration.eval(t_env)
                    rand_number = np.random.rand(batch_size*self.n_agents)
                    # random actions sample uniformly from action space
                    random_action = Categorical(logits=torch.ones(batch_size*self.n_agents, self.act_dim[i])).sample().numpy()
                    take_random = (rand_number < eps).astype(int)
                    action = (1 - take_random) * to_numpy(greedy_action) + take_random * random_action
                    onehot_action = make_onehot(action, self.act_dim[i])
                else:
                    onehot_action = make_onehot(greedy_action, self.act_dim[i])

                onehot_actions.append(onehot_action)

            onehot_actions = np.concatenate(onehot_actions, axis=-1)
        else:
            if explore:
                eps = self.exploration.eval(t_env)
                rand_numbers = np.random.rand(batch_size*self.n_agents)
                # random actions sample uniformly from action space
                logits = avail_choose(torch.ones(batch_size*self.n_agents, self.act_dim), available_actions.squeeze())
                random_actions = Categorical(logits=logits).sample().numpy()
                take_random = (rand_numbers < eps).astype(int)
                #take_random表示在每一条轨迹初始多采取随机动作，后面多采取指定动作
                actions = (1 - take_random) * to_numpy(actions.max(dim=1)[1]) + take_random * random_actions
                onehot_actions = make_onehot(actions, self.act_dim)
            else:
                onehot_actions = make_onehot(actions.max(dim=1)[1], self.act_dim)
                
        return onehot_actions, None,None,None
    def get_random_actions(self, obs, available_actions=None):
        """See parent class."""
        batch_size = obs.shape[0]

        if self.multidiscrete:
            random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().numpy() for i in
                                range(len(self.act_dim))]
            random_actions = np.concatenate(random_actions, axis=-1)
        else:
            if available_actions is not None:
                logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                random_actions = OneHotCategorical(logits=logits).sample().numpy()
            else:
                random_actions = OneHotCategorical(logits=torch.ones(batch_size, self.act_dim)).sample().numpy()
        
        return random_actions

    def init_hidden(self, num_agents, batch_size):
        if num_agents == -1:
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(batch_size*num_agents, self.hidden_size)
          
    def init_p_hidden(self, num_agents, batch_size):
        if num_agents == -1:
            return torch.zeros(batch_size, self.pair_rnn_hidden_dim)
        else:
            return torch.zeros(batch_size*num_agents, self.pair_rnn_hidden_dim)

    def parameters(self):
        parameters_sum = []
        parameters_sum += self.rnn_network.parameters()
        if self.independent_p_q:
            parameters_sum += self.p_rnn_network.parameters()
        for num_orders in range(1,self.highest_orders+1):
            parameters_sum += self.q_network[num_orders].parameters()

        return parameters_sum
    def action_encoder_params(self):
        return self.action_encoder.parameters()
          
    def load_state(self, source_policy):
        self.rnn_network.load_state_dict(source_policy.rnn_network.state_dict())
        for num_orders in range(1,self.highest_orders+1):
            self.q_network[num_orders].load_state_dict(source_policy.q_network[num_orders].state_dict())
        if self.independent_p_q:
            self.p_rnn_network.load_state_dict(source_policy.p_rnn_network.state_dict())
        self.action_encoder.load_state_dict(source_policy.action_encoder.state_dict())
        self.action_repr = copy.deepcopy(source_policy.action_repr)
        self.p_action_repr = copy.deepcopy(source_policy.p_action_repr)
        
    def action_repr_forward(self, obs, actions_onehot):
        return self.action_encoder.predict(obs, actions_onehot)
      
    def update_action_repr(self):
        action_repr = self.action_encoder()

        self.action_repr = action_repr.detach().clone()

        # Pairwise Q (|A|, al) -> (|A|, |A|, 2*al)
        input_i = self.action_repr.unsqueeze(1).repeat(1, self.act_dim, 1)
        input_j = self.action_repr.unsqueeze(0).repeat(self.act_dim, 1, 1)
        self.p_action_repr = torch.cat([input_i, input_j], dim=-1).view(self.act_dim * self.act_dim,
                                                                     -1).t().unsqueeze(0)
            
            
            
            
