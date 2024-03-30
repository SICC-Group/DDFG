import numpy as np
import torch
import time
import torch_scatter
from offpolicy.algorithms.sopcg.algorithm.agent_q_function import AgentQFunction
from offpolicy.algorithms.sopcg.algorithm.adj_generator import Adj_Generator
from torch.distributions import Categorical, OneHotCategorical
from offpolicy.utils.util import get_dim_from_space, is_discrete, is_multidiscrete, make_onehot, DecayThenFlatSchedule, avail_choose, to_torch, to_numpy
import offpolicy.algorithms.utils.constructor as constructor
from offpolicy.algorithms.base.mlp_policy import MLPPolicy
from offpolicy.algorithms.sopcg.algorithm.rnn import RNNBase

class R_SOPCGPolicy_Parallel(MLPPolicy):
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
        self.q_network_input_dim = self.rnn_out_dim
        self.q_hidden_size = [32,64,128]
        self.highest_orders = self.args.highest_orders
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        # Local recurrent q network for the agent
        self.solver = constructor.TreeSolver(self.args)
        self.constructor = constructor.Constructor(self.args,self.n_agents,self.act_dim)
        self.rnn_network = RNNBase(self.args, self.rnn_network_input_dim, self.rnn_hidden_size, self.rnn_out_dim, self.device)
        self.q_network = {num_orders :AgentQFunction(self.args, self.q_network_input_dim*num_orders, self.q_hidden_size[num_orders-1], self.act_dim**num_orders, self.device) for num_orders in range(1,self.highest_orders+1)}
        
        if train:
            self.exploration = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish, self.args.epsilon_anneal_time,decay="linear")
               

    def get_hidden_states(self, obs, prev_actions, rnn_states):
        if self.args.prev_act_inp:
            prev_action_batch = to_torch(prev_actions).type(torch.float32)
            input_batch = torch.cat((to_torch(obs), prev_action_batch), dim=-1)
        else:
            input_batch = to_torch(obs)

        #if len(input_batch.shape) == 3:
            #input_batch = input_batch.reshape(-1,input_batch.shape[2])
        
        q_batch, new_rnn_states, no_sequence = self.rnn_network(input_batch.to(**self.tpdv), to_torch(rnn_states).to(**self.tpdv))
        #import pdb;pdb.set_trace()
        return q_batch,new_rnn_states,no_sequence
      
    def get_rnn_batch(self,obs_batch,batch_size,available_actions=None,no_sequence=False,select_graph=False):
        #import pdb;pdb.set_trace()
        f = self.q_network[1](obs_batch,no_sequence).view(-1, self.n_agents, self.act_dim)
        
        input_i = obs_batch.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        input_j = obs_batch.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
        inputs = torch.cat([input_i, input_j], dim=-1).view(-1, 2 * self.q_network_input_dim)
        g = self.q_network[2](inputs,no_sequence)
        g = g.view(-1, self.n_agents, self.n_agents, self.act_dim, self.act_dim)
        g = (g + g.permute(0, 2, 1, 4, 3)) / 2.
        
        if select_graph == False:
            return f, g

        graphs = self.solver.solve(f, g, available_actions, self.device)
        return f, g, graphs
      
    def get_q_values(self, obs_batch, action_batch, adj_input=None,no_sequence=False):
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

        q_batch, idx_node_order, _, _ = self.get_rnn_batch(obs_batch, batch_size, adj_input,no_sequence)

        values = self.q_values(q_batch, action_batch.type(torch.int64), idx_node_order, batch_size)
        
        return values

    def get_actions(self, obs_batch, available_actions=None, t_env=None, explore=False,adj_input = None,no_sequence = False):
        if len(obs_batch.shape) == 3:
            batch_size = obs_batch.shape[0]
        else:
            batch_size = 1

        f, g, graphs = self.get_rnn_batch(obs_batch,batch_size,available_actions.unsqueeze(0),no_sequence,select_graph=True)
        #import pdb;pdb.set_trace()
        actions = self.constructor.compute_outputs(f, g, available_actions.unsqueeze(0), graphs)
        actions = actions.squeeze()
        #import pdb;pdb.set_trace()
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
                logits = avail_choose(torch.ones(batch_size*self.n_agents, self.act_dim), available_actions)
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

    def parameters(self):
        parameters_sum = []
        parameters_sum += self.rnn_network.parameters()
        for num_orders in range(1,self.highest_orders+1):
            parameters_sum += self.q_network[num_orders].parameters()

        return parameters_sum

    def load_state(self, source_policy):
        self.rnn_network.load_state_dict(source_policy.rnn_network.state_dict())
        for num_orders in range(1,self.highest_orders+1):
            self.q_network[num_orders].load_state_dict(source_policy.q_network[num_orders].state_dict())
