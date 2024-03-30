import numpy as np
import torch
import time
import torch_scatter
from offpolicy.algorithms.r_mfg_cent.algorithm.agent_q_function import AgentQFunction
from offpolicy.algorithms.r_mfg_cent.algorithm.adj_generator import Adj_Generator
from torch.distributions import Categorical, OneHotCategorical
from offpolicy.utils.util import get_dim_from_space, is_discrete, is_multidiscrete, make_onehot, DecayThenFlatSchedule, avail_choose, to_torch, to_numpy
from offpolicy.algorithms.base.mlp_policy import MLPPolicy
from offpolicy.algorithms.r_mfg_cent.algorithm.rnn import RNNBase

class R_MFGPolicy_Parallel(MLPPolicy):
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

    '''def get_q_batch(self,obs_batch,prev_actions,rnn_states,batch_size,adj_input=None):
        _,new_rnn_states,no_sequence = self.get_hidden_states(obs_batch,prev_actions,rnn_states)
        obs_batch = new_rnn_states.reshape(batch_size,-1,new_rnn_states.shape[1])
        list_obs_batch = [[] for i in range(self.highest_orders)]
        idx_node_order = [[] for i in range(self.highest_orders)]
        num_edges = 0
        q_batch = []

        for i in range(adj_input.shape[1]):
            idx_node_order[torch.sum(adj_input[:,i]==1)-1].append(torch.cat((torch.tensor([i]),torch.where(adj_input[:,i]==1)[0])))

        import pdb;pdb.set_trace()
        for i in range(self.highest_orders):
            len_i = len(idx_node_order[i])
            if len_i != 0:
                temp_obs = torch.zeros((batch_size,len_i,obs_batch.shape[2]*(i+1)))
                for j in range(len_i):
                    
                    temp_obs[:,j] = obs_batch[:,idx_node_order[i][j][1:]].reshape(batch_size,-1)
                list_obs_batch[i] = temp_obs

        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                q_batch.append(self.q_network[i+1](list_obs_batch[i],no_sequence).double().cpu())
            else:
                q_batch.append([])

        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                num_edges += q_batch[i].shape[1]*(i+1)

        return q_batch, idx_node_order, adj_input, num_edges, new_rnn_states'''
      
    def get_rnn_batch(self,obs_batch,batch_size,adj_input=None,no_sequence=False):
        list_obs_batch = [[] for i in range(self.highest_orders)]
        idx_node_order = [[] for i in range(self.highest_orders)]
        num_edges = 0
        q_batch = []
        #if len(obs_batch.shape)==2:
            #obs_batch = np.expand_dims(obs_batch,axis=0)
        for i in range(adj_input.shape[1]):
            idx_node_order[torch.sum(adj_input[:,i]==1)-1].append(torch.cat((torch.tensor([i]).to(self.device),torch.where(adj_input[:,i]==1)[0])))
            #idx_node_order[np.sum(adj_input[:,i]==1)-1].append(np.append(i,np.where(adj_input[:,i]==1)))
        
        for i in range(self.highest_orders):
            len_i = len(idx_node_order[i])
            if len_i != 0:
                temp_obs = torch.zeros((batch_size,len_i,obs_batch.shape[2]*(i+1)))
                for j in range(len_i):
                    temp_obs[:,j] = obs_batch[:,idx_node_order[i][j][1:]].reshape(batch_size,-1)
                list_obs_batch[i] = temp_obs

        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                q_batch.append(self.q_network[i+1](list_obs_batch[i],no_sequence))
            else:
                q_batch.append([])

        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                num_edges += q_batch[i].shape[1]*(i+1)
        #import pdb;pdb.set_trace()
        return q_batch, idx_node_order, adj_input, num_edges
      
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
        #import pdb;pdb.set_trace()
        values = self.q_values(q_batch, action_batch.type(torch.int64), idx_node_order, batch_size)
        
        return values

    def q_values(self, f_q, actions, idx_node_order, batch_size):
        """ Computes the Q-values for given utilities, payoffs and actions (Algorithm 2 in Boehmer et al., 2020). """
        idx_a_of_Q = []

        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                idx_a_of_Q.append(torch.stack(idx_node_order[i])[:,1:].t().squeeze())
            else:
                idx_a_of_Q.append([])
        
        # Use the utilities for the chosen actions
        values = 0
        if len(idx_node_order[0]) != 0:
            edge_actions_1 = actions[:,idx_a_of_Q[0]]
            if len(edge_actions_1.shape) == 2:
                edge_actions_1 = edge_actions_1.unsqueeze(dim=-1)
            #values += f_q[0].gather(dim=-1, index=edge_actions_1).squeeze(dim=-1).sum(dim=-1)
            values = f_q[0].gather(dim=-1, index=edge_actions_1).squeeze(dim=-1).mean(dim=-1)

        #import pdb;pdb.set_trace()
        if len(idx_node_order[1]) != 0:
            #f_ij = f_ij.view(n_batches, len(self.edges_from), self.n_actions * self.n_actions)
            edge_actions_2 = actions[:,idx_a_of_Q[1][0]] * self.act_dim + actions[:,idx_a_of_Q[1][1]]
            if len(edge_actions_2.shape) == 2:
               edge_actions_2 = edge_actions_2.unsqueeze(dim=-1)
            values += f_q[1].gather(dim=-1, index=edge_actions_2).squeeze(dim=-1).mean(dim=-1)

        if len(idx_node_order) > 2 and len(idx_node_order[2]) != 0:
            edge_actions_3 = (actions[:,idx_a_of_Q[2][0]] * self.act_dim + actions[:,idx_a_of_Q[2][1]]) \
                * self.act_dim + actions[:,idx_a_of_Q[2][2]]
            if len(edge_actions_3.shape) == 2:
                edge_actions_3 = edge_actions_3.unsqueeze(dim=-1)
            values += f_q[2].gather(dim=-1, index=edge_actions_3).squeeze(dim=-1).mean(dim=-1)
        # Return the Q-values for the given actions
        return values

    def greedy(self, adj, q_batch, idx_node_order,available_actions,num_edges,batch_size):
        """ Finds the maximum Q-values and corresponding greedy actions for given utilities and payoffs.
            (Algorithm 3 in Boehmer et al., 2020)"""
        # All relevant tensors should be double to reduce accumulating precision loss
        lamda = self.lamda
        adj_f = torch.full([self.num_factor,self.highest_orders],self.n_agents).to(self.device)
        adj_edge = torch.where(adj==1)
        #num_edges = 0
        in_q_batch = []
        for i in range(adj.shape[1]):
            adj_f[i,:len(torch.where(adj[:,i]==1)[0])] = torch.where(adj[:,i]==1)[0]
        
        f_q = torch.zeros((batch_size,self.num_factor,self.act_dim,self.act_dim,self.act_dim)).to(**self.tpdv)
        for i in range(len(idx_node_order)):
            in_q_batch.append(q_batch[i])
            if len(idx_node_order[i]) != 0:
                q_batch[i] = q_batch[i].unsqueeze(-1).repeat(1,1,1,self.act_dim**(2-i)) / q_batch[i].shape[1]
                q_batch[i] = q_batch[i].reshape((-1,q_batch[i].shape[1],self.act_dim,self.act_dim,self.act_dim))
                f_q[:,torch.stack(idx_node_order[i])[:,0]] = q_batch[i]
        # Unavailable actions have a utility of -inf, which propagates throughout message passing
        # Initialize best seen value and actions for anytime-extension
        best_value = torch.empty(batch_size).fill_(-float('inf')).to(**self.tpdv) #[1] device=self.device
        best_actions = torch.empty(best_value.shape[0], self.n_agents, 1, dtype=torch.int64).to(self.device) #[1,8,1]
        # Without edges (or iterations), CG would be the same as VDN: mean(f_i)
        utils_Q = best_value.new_zeros(best_value.shape[0], self.num_factor, self.act_dim, self.act_dim, self.act_dim).to(**self.tpdv) #[1,5,5]
        utils_a = best_value.new_zeros(best_value.shape[0], self.n_agents, self.act_dim).to(**self.tpdv) #[1,8,5]
        avail_a = best_value.new_zeros(best_value.shape[0], self.n_agents, self.act_dim).to(**self.tpdv)
        if available_actions is not None:
            avail_a = avail_a.masked_fill(available_actions == 0, -float('inf'))  
        idx_dim = torch.where((adj_f[adj_edge[1]] == adj_edge[0].unsqueeze(dim=-1)))[1]
        idx_type = []
        idx_type.append(torch.where(idx_dim ==0))
        idx_type.append(torch.where(idx_dim ==1))
        idx_type.append(torch.where(idx_dim ==2))
        num_dim = torch.tensor([[1,2],[0,2],[0,1]])+2
        # Perform message passing for self.iterations: [0] are messages to *edges_to*, [1] are messages to *edges_from*   
        #import pdb;pdb.set_trace()
        if num_edges > 0 and self.args.msg_iterations > 0:
            messages_a2Q = best_value.new_zeros(batch_size, num_edges, self.act_dim, self.act_dim, self.act_dim).to(**self.tpdv) #[1,11,5]
            messages_Q2a = best_value.new_zeros(batch_size, num_edges, self.act_dim).to(**self.tpdv) #[1,11,5] 
            zeros_Q2a = torch.full_like(messages_Q2a, 0).to(**self.tpdv)
            zeros_a2Q = torch.full_like(messages_a2Q, 0).to(**self.tpdv)
            inf_joint_Q2a = best_value.new_zeros(batch_size, num_edges, self.act_dim, self.act_dim, self.act_dim).fill_(-float('inf')).to(**self.tpdv)
            for iteration in range(self.args.msg_iterations):
                joint_Q2a = utils_Q[:,adj_edge[1]] - messages_a2Q + f_q[:,adj_edge[1]]      
                
                joint_Q2a = torch.where(torch.isnan(joint_Q2a), inf_joint_Q2a, joint_Q2a)
                #[1,16,5,5,5]
                # Maximize the joint Q-value over the action of the sender
                '''for i in range(num_edges):
                    idx_dim = torch.where((adj_f[adj_edge[1][i]] != adj_edge[0][i]))[0]+1
                    messages_Q2a[:,i] = joint_Q2a[:,i].max(dim=idx_dim[0])[0].max(dim=idx_dim[1]-1)[0]'''
                for i in range(3):
                    if min(idx_type[i][0].shape) != 0:
                        messages_Q2a[:,idx_type[i][0]] = joint_Q2a[:,idx_type[i][0]].max(dim=num_dim[i][0])[0].max(dim=num_dim[i][1]-1)[0]
                # Normalization as in Kok and Vlassis (2006) and Wainwright et al. (2004)
                
                if self.args.msg_normalized:
                    messages_Q2a -= torch.where(torch.isinf(messages_Q2a), zeros_Q2a, messages_Q2a).mean(dim=-1, keepdim=True)

                utils_a = avail_a + torch_scatter.scatter_add(src=messages_Q2a, index=adj_edge[0], dim=1, dim_size=self.n_agents)
                # Anytime extension (Kok and Vlassis, 2006)
                if iteration%2 == 0:
                    joint_a2Q = utils_a[:,adj_edge[0]] - messages_Q2a #[batch_size,num_edge,act_dim] = [1,16,5] 
                else:
                    joint_a2Q = lamda * joint_a2Q + (1-lamda) * (utils_a[:,adj_edge[0]] - messages_Q2a)
                    
                for i in range(3):
                    if min(idx_type[i][0].shape) != 0:
                        #import pdb;pdb.set_trace()
                        messages_a2Q[:,idx_type[i][0]] = joint_a2Q[:,idx_type[i][0]].unsqueeze(num_dim[i][0]).unsqueeze(num_dim[i][1])
                if self.args.msg_normalized:
                    messages_a2Q -= torch.where(torch.isfinite(messages_a2Q), messages_a2Q, zeros_a2Q).mean(dim=-1, keepdim=True)
                utils_Q = torch_scatter.scatter_add(src=messages_a2Q, index=adj_edge[1], dim=1, dim_size=self.num_factor)           
                if self.args.msg_anytime:
                    # Find currently best actions and the (true) value of these actions
                    actions = utils_a.max(dim=-1, keepdim=True)[1]
                    value = self.q_values(in_q_batch, actions,idx_node_order,batch_size)
                    #mport pdb;pdb.set_trace()
                    # Update best_actions only for the batches that have a higher value than best_value
                    change = value > best_value
                    best_value[change] = value[change]
                    best_actions[change] = actions[change]
        # Return the greedy actions and the corresponding message output averaged across agents
        #import pdb;pdb.set_trace()
        if not self.args.msg_anytime or num_edges == 0 or self.args.msg_iterations <= 0:
            _, best_actions = utils_a.max(dim=-1, keepdim=True)
        return best_actions, best_value

    def get_actions(self, obs_batch, available_actions=None, t_env=None, explore=False,adj_input = None,no_sequence = False):
        if len(obs_batch.shape) == 3:
            batch_size = obs_batch.shape[0]
        else:
            batch_size = 1

        q_batch, idx_node_order, adj, num_edges = self.get_rnn_batch(obs_batch,batch_size,adj_input,no_sequence)
        
        actions, best_value = self.greedy(adj,q_batch,idx_node_order,available_actions,num_edges,batch_size)
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
                actions = (1 - take_random) * to_numpy(actions) + take_random * random_actions
                onehot_actions = make_onehot(actions, self.act_dim)
            else:
                onehot_actions = make_onehot(actions, self.act_dim)
                
        return onehot_actions, best_value,None,None
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
