import numpy as np
import torch
import torch_scatter
import time
from offpolicy.algorithms.r_ddfg_cent_rw.algorithm.agent_q_function import AgentQFunction
from offpolicy.algorithms.r_ddfg_cent_rw.algorithm.agent_v_function import AgentVFunction
from offpolicy.algorithms.r_ddfg_cent_rw.algorithm.adj_generator import Adj_Generator
from torch.distributions import Categorical, OneHotCategorical
from offpolicy.utils.util import get_dim_from_space, is_discrete, is_multidiscrete, make_onehot, DecayThenFlatSchedule, avail_choose, to_torch, to_numpy
from offpolicy.algorithms.base.mlp_policy import MLPPolicy
from offpolicy.algorithms.r_ddfg_cent_rw.algorithm.rnn import RNNBase

class R_DDFGPolicy(MLPPolicy):
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
        self.num_factor = config["num_agents"] + self.args.num_factor
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.act_space)
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        self.central_obs_dim = policy_config["cent_obs_dim"]
        self.discrete_action = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)
        self.hidden_size = self.args.hidden_size
        self.lamda = self.args.lamda
        self.num_rank = self.args.num_rank
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
        self.q_out_size = []
        self.q_out_size.append(self.act_dim)
        for num_orders in range(2,self.highest_orders+1):
            self.q_out_size.append(num_orders*self.num_rank*self.act_dim)
        
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.use_vfunction = self.args.use_vfunction
        # Local recurrent q network for the agent
        self.rnn_network = RNNBase(self.args, self.rnn_network_input_dim, self.rnn_hidden_size, self.rnn_out_dim, self.device)
        self.q_network = {num_orders :AgentQFunction(self.args, self.q_network_input_dim*num_orders, self.q_hidden_size[num_orders-1], self.q_out_size[num_orders-1], self.device) for num_orders in range(1,self.highest_orders+1)}
        if self.use_vfunction:
            self.v_network = {num_orders :AgentVFunction(self.args, self.q_network_input_dim*num_orders, self.rnn_hidden_size, 1, self.device) for num_orders in range(1,self.highest_orders+1)}
        
        if train:
            self.exploration = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish, self.args.epsilon_anneal_time,decay="linear")
               

    def get_hidden_states(self, obs, prev_actions, rnn_states):
        if self.args.prev_act_inp:
            prev_action_batch = to_torch(prev_actions)
            input_batch = torch.cat((obs, prev_action_batch), dim=-1)
        else:
            input_batch = to_torch(obs)
        
        q_batch, new_rnn_states, no_sequence = self.rnn_network(input_batch.to(**self.tpdv), to_torch(rnn_states).to(**self.tpdv))
        #import pdb;pdb.set_trace()
        return q_batch,new_rnn_states,no_sequence
      
    def get_v_batch(self,obs_batch,adj_input=None,batch_size=1,no_sequence=False,dones=None):
                
        list_obs_batch = [[] for i in range(self.highest_orders)]
        num_edges = 0
        q_batch = []
        idx_node_order = []
        #obs_batch = obs_batch.repeat(2,1,1)
        adj_input = adj_input.transpose(1,2)
        for i in range(1,self.highest_orders+1):
            idx = torch.where(torch.sum(adj_input,dim=2)==i)
            tmp = torch.cat([idx[0].unsqueeze(-1),idx[1].unsqueeze(-1),torch.where(adj_input[idx])[1].reshape(-1,i)],dim=-1)
            idx_node_order.append(tmp)           

        for i in range(self.highest_orders):
            tmp = idx_node_order[i]
            len_i = len(tmp)
            if len_i != 0:
                list_obs_batch[i] = obs_batch[tmp[:,:1],tmp[:,2:]].reshape(len_i,-1)
        
        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                q_batch.append(self.v_network[i+1](list_obs_batch[i],no_sequence))
            else:
                q_batch.append([])               
        
        num_edges = adj_input.sum()
        idx_factor = torch.sum(adj_input.transpose(1,2),dim=1)
        for i in range(len(idx_node_order)):
            if len(idx_node_order[i]) != 0:
                q_batch[i] = q_batch[i] / torch.sum(idx_factor==i+1,dim=-1)[idx_node_order[i][:,0]].unsqueeze(-1)
        #n_f = self.args.num_factor
        #q_batch[0] = torch.where((idx_node_order[0][:,1].unsqueeze(-1)>n_f) & (dones[idx_node_order[0][:,0],torch.clamp(idx_node_order[0][:,1]-n_f,min=-1)]==1),torch.zeros_like(q_batch[0]),q_batch[0])     
        return q_batch, idx_node_order, adj_input.transpose(1,2), num_edges   
      
    def get_v_values(self, obs_batch,adj_input=None,no_sequence=False,dones=None):

        if len(obs_batch.shape) == 3:
            batch_size = obs_batch.shape[0]
        else:
            batch_size = 1
            
        v_batch, idx_node_order, adj , num_edges= self.get_v_batch(obs_batch, adj_input,batch_size,no_sequence,dones) #0.00358

        if batch_size == 1:
            values = self.v_local_values(v_batch, idx_node_order)
        else:
            f_v = torch.zeros((batch_size,self.num_factor,1)).to(self.device)
            for i in range(len(idx_node_order)):
                if len(idx_node_order[i]) != 0:
                    f_v[idx_node_order[i][:,0],idx_node_order[i][:,1]] = v_batch[i]
            values =  f_v.sum(dim=1)
           
        return values

    def v_local_values(self, f_q, idx_node_order):
        """ Computes the Q-values for given utilities, payoffs and actions (Algorithm 2 in Boehmer et al., 2020). """
        num_f = self.num_factor     
        # Use the utilities for the chosen actions
        values_f = torch.zeros((1,num_f,1)).to(self.device)
        if len(idx_node_order[0]) != 0:
            values_f[0,idx_node_order[0][:,1]] = f_q[0]

        if self.highest_orders > 1 and len(idx_node_order[1]) != 0:         
            values_f[0,idx_node_order[1][:,1]] = f_q[1]

        if self.highest_orders == 3 and len(idx_node_order[2]) != 0:
            values_f[0,idx_node_order[2][:,1]] = f_q[2]
        # Return the Q-values for the given actions
        return values_f #vadj_inputalues_f[:,:self.num_factor-self.n_agents]
      
    def get_rnn_batch(self,obs_batch,adj_input=None,batch_size=1,no_sequence=False,dones=None):
                
        list_obs_batch = [[] for i in range(self.highest_orders)]
        num_edges = 0
        q_batch = []
        idx_node_order = []
        adj_input = adj_input.transpose(1,2)
        for i in range(1,self.highest_orders+1):
            idx = torch.where(torch.sum(adj_input,dim=2)==i)
            tmp = torch.cat([idx[0].unsqueeze(-1),idx[1].unsqueeze(-1),torch.where(adj_input[idx])[1].reshape(-1,i)],dim=-1)
            idx_node_order.append(tmp)

        for i in range(self.highest_orders):
            tmp = idx_node_order[i]
            len_i = len(tmp)
            if len_i != 0:
                list_obs_batch[i] = obs_batch[tmp[:,:1],tmp[:,2:]].reshape(len_i,-1)
        
        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                q_batch.append(self.q_network[i+1](list_obs_batch[i],no_sequence))
            else:
                q_batch.append([])
                
        if self.highest_orders > 1 and len(idx_node_order[1]) != 0:
            dim = list(q_batch[1].shape[:-1])
            tmp = q_batch[1].view(*[np.prod(dim) , self.num_rank, 2, self.act_dim])
            q_batch[1] = torch.einsum('abi,abj->aij',tmp[:,:,0],tmp[:,:,1]).reshape(np.prod(dim),-1)
        if self.highest_orders == 3 and len(idx_node_order[2]) != 0:
            dim = list(q_batch[2].shape[:-1])
            tmp = q_batch[2].view(*[np.prod(dim) , self.num_rank, 3, self.act_dim])
            q_batch[2] = torch.einsum('abi,abj,abk->aijk',tmp[:,:,0],tmp[:,:,1],tmp[:,:,2]).reshape(np.prod(dim),-1)
        
        num_edges = adj_input.sum()
        idx_factor = torch.sum(adj_input.transpose(1,2),dim=1)
        for i in range(len(idx_node_order)):
            if len(idx_node_order[i]) != 0:
                q_batch[i] = q_batch[i] / torch.sum(idx_factor==i+1,dim=-1)[idx_node_order[i][:,0]].unsqueeze(-1)
    
        return q_batch, idx_node_order, adj_input.transpose(1,2), num_edges
      
    def get_q_values(self, obs_batch, action_batch, adj_input=None,no_sequence=False,dones=None):
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
            
        q_batch, idx_node_order, adj , num_edges= self.get_rnn_batch(obs_batch, adj_input,batch_size,no_sequence,dones) #0.00358
        values = self.q_values(q_batch, action_batch.type(torch.int64), idx_node_order,batch_size)  #0.0002

        return values

    def q_values(self, f_q, actions, idx_node_order,batch_size):
        """ Computes the Q-values for given utilities, payoffs and actions (Algorithm 2 in Boehmer et al., 2020). """
        idx_a_of_Q = []
        
        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                idx_a_of_Q.append(torch.cat([idx_node_order[i][:,:1],idx_node_order[i][:,2:]],dim=1).t().squeeze())
            else:
                idx_a_of_Q.append([])
        
        # Use the utilities for the chosen actions
        values = torch.zeros((3,batch_size)).to(self.device)
        if len(idx_node_order[0]) != 0:
            edge_actions_1 = actions[idx_a_of_Q[0][0],idx_a_of_Q[0][1]]
            if len(edge_actions_1.shape) == 1:
                edge_actions_1 = edge_actions_1.unsqueeze(dim=-1)
            tmp1 = torch.cat([f_q[0].gather(dim=-1, index=edge_actions_1),torch.zeros((1,1)).to(self.device)],dim=0)
            if len(idx_a_of_Q[0][0].shape) == 0:
                index1 = torch.cat([idx_a_of_Q[0][0].unsqueeze(0),torch.tensor([batch_size]).to(self.device)])
            else:
                index1 = torch.cat([idx_a_of_Q[0][0],torch.tensor([batch_size]).to(self.device)]) 
            values[0] = torch_scatter.scatter_add(src=tmp1, index=index1, dim=0)[:-1].reshape(-1,batch_size)

            #values = f_q[0].gather(dim=-1, index=edge_actions_1).squeeze(dim=-1).mean(dim=-1)
        if self.highest_orders > 1 and len(idx_node_order[1]) != 0:
            edge_actions_2 = actions[idx_a_of_Q[1][0],idx_a_of_Q[1][1]] * self.act_dim + actions[idx_a_of_Q[1][0],idx_a_of_Q[1][2]]
            if len(edge_actions_2.shape) == 1:
               edge_actions_2 = edge_actions_2.unsqueeze(dim=-1)
            tmp2 = torch.cat([f_q[1].gather(dim=-1, index=edge_actions_2),torch.zeros((1,1)).to(self.device)],dim=0)
            if len(idx_a_of_Q[1][0].shape) == 0:
                index2 = torch.cat([idx_a_of_Q[1][0].unsqueeze(0),torch.tensor([batch_size]).to(self.device)])
            else:
                index2 = torch.cat([idx_a_of_Q[1][0],torch.tensor([batch_size]).to(self.device)])
            values[1] = torch_scatter.scatter_add(src=tmp2, index=index2, dim=0)[:-1].reshape(-1,batch_size)

        if self.highest_orders == 3 and len(idx_node_order[2]) != 0:
            edge_actions_3 = (actions[idx_a_of_Q[2][0],idx_a_of_Q[2][1]] * self.act_dim + actions[idx_a_of_Q[2][0],idx_a_of_Q[2][2]]) * self.act_dim + actions[idx_a_of_Q[2][0],idx_a_of_Q[2][3]]
            if len(edge_actions_3.shape) == 1:
                edge_actions_3 = edge_actions_3.unsqueeze(dim=-1)
            tmp3 = torch.cat([f_q[2].gather(dim=-1, index=edge_actions_3),torch.zeros((1,1)).to(self.device)],dim=0)
            if len(idx_a_of_Q[2][0].shape) == 0:
                index3 = torch.cat([idx_a_of_Q[2][0].unsqueeze(0),torch.tensor([batch_size]).to(self.device)])
            else:
                index3 = torch.cat([idx_a_of_Q[2][0],torch.tensor([batch_size]).to(self.device)])   
            values[2] = torch_scatter.scatter_add(src=tmp3, index=index3, dim=0)[:-1].reshape(-1,batch_size)
        # Return the Q-values for the given actions
        return values.sum(0)

    def q_local_values(self, f_q, actions, idx_node_order):
        """ Computes the Q-values for given utilities, payoffs and actions (Algorithm 2 in Boehmer et al., 2020). """
        num_f = f_q.shape[1]
        f = f_q.reshape(1,num_f,-1)
        idx_a_of_Q = []
        
        for i in range(self.highest_orders):
            if len(idx_node_order[i]) != 0:
                idx_a_of_Q.append(torch.cat([idx_node_order[i][:,:1],idx_node_order[i][:,2:]],dim=1).t().squeeze())
            else:
                idx_a_of_Q.append([])
        
        # Use the utilities for the chosen actions
        values_f = torch.zeros((1,num_f,1)).to(self.device)
        if len(idx_node_order[0]) != 0:
            edge_actions_1 = (actions[idx_a_of_Q[0][0],idx_a_of_Q[0][1]] * self.act_dim + actions[idx_a_of_Q[0][0],idx_a_of_Q[0][1]]) * self.act_dim + actions[idx_a_of_Q[0][0],idx_a_of_Q[0][1]]
            if len(edge_actions_1.shape) == 1:
                edge_actions_1 = edge_actions_1.unsqueeze(dim=-1)
            values_f[0,idx_node_order[0][:,1]] = f[0][idx_node_order[0][:,1]].gather(dim=-1, index=edge_actions_1)

        if self.highest_orders > 1 and  len(idx_node_order[1]) != 0:         
            edge_actions_2 = (actions[idx_a_of_Q[1][0],idx_a_of_Q[1][1]] * self.act_dim + actions[idx_a_of_Q[1][0],idx_a_of_Q[1][2]]) * self.act_dim + actions[idx_a_of_Q[1][0],idx_a_of_Q[1][2]]
            if len(edge_actions_2.shape) == 1:
                edge_actions_2 = edge_actions_2.unsqueeze(dim=-1)
            values_f[0,idx_node_order[1][:,1]] = f[0][idx_node_order[1][:,1]].gather(dim=-1, index=edge_actions_2)

        if self.highest_orders == 3 and len(idx_node_order[2]) != 0:
            edge_actions_3 = (actions[idx_a_of_Q[2][0],idx_a_of_Q[2][1]] * self.act_dim + actions[idx_a_of_Q[2][0],idx_a_of_Q[2][2]]) * self.act_dim + actions[idx_a_of_Q[2][0],idx_a_of_Q[2][3]]
            if len(edge_actions_3.shape) == 1:
                edge_actions_3 = edge_actions_3.unsqueeze(dim=-1)
            values_f[0,idx_node_order[2][:,1]] = f[0][idx_node_order[2][:,1]].gather(dim=-1, index=edge_actions_3)
        # Return the Q-values for the given actions
        return values_f
      
    def greedy(self, adj, q_batch, idx_node_order,available_actions,num_edges,batch_size):
        """ Finds the maximum Q-values and corresponding greedy actions for given utilities and payoffs.
            (Algorithm 3 in Boehmer et al., 2020)"""
        # All relevant tensors should be double to reduce accumulating precision loss
        #batch_size = 2
        #available_actions = available_actions.repeat(2,1,1)
        lamda = self.lamda
        adj_f = torch.full([batch_size,self.num_factor,self.highest_orders],self.n_agents,dtype=torch.int64).to(self.device)
        adj_edge = torch.where(adj)
        adj_tmp = adj.reshape(-1,self.num_factor)
        var_edge,  f_edge = torch.where(adj_tmp)
        f_add_dim = torch.zeros_like(adj)
        f_add_dim[:] = (torch.arange(0,batch_size,1) * self.num_factor).unsqueeze(-1).unsqueeze(-1)
        f_edge += f_add_dim.reshape(-1,self.num_factor)[adj_tmp==1]

        idx_factor = torch.sum(adj,dim=1)
        for i in range(1,self.highest_orders+1):
            tmp = torch.where(idx_factor==i)
            if len(tmp) != 0:
                adj_f[tmp[0],tmp[1],:i] = torch.where(adj.transpose(1,2)[tmp[0],tmp[1]])[1].reshape(-1,i)
        
        idx_dim = torch.where(adj_f[adj_edge[0],adj_edge[2]] == adj_edge[1].unsqueeze(dim=-1))[1]
        idx_type = []
        idx_type.append(torch.where(idx_dim ==0))
        idx_type.append(torch.where(idx_dim ==1))
        idx_type.append(torch.where(idx_dim ==2))
        num_dim = torch.tensor([[1,2],[0,2],[0,1]])+1
        in_q_batch = []
        f_q = torch.zeros((batch_size,self.num_factor,self.act_dim,self.act_dim,self.act_dim)).to(self.device)
        for i in range(len(idx_node_order)):
            in_q_batch.append(q_batch[i])
            if len(idx_node_order[i]) != 0:
                q_batch[i] = q_batch[i].unsqueeze(-1).repeat(1,1,self.act_dim**(2-i))
                q_batch[i] = q_batch[i].reshape((q_batch[i].shape[0],self.act_dim,self.act_dim,self.act_dim))
                f_q[idx_node_order[i][:,0],idx_node_order[i][:,1]] = q_batch[i]
        # Unavailable actions have a utility of -inf, which propagates throughout message passing
        # Initialize best seen value and actions for anytime-extension
        best_value = torch.empty(batch_size, dtype=torch.float64).fill_(-float('inf')).to(**self.tpdv) #[1] device=self.device     
        best_f_value = torch.empty(batch_size, self.num_factor,1, dtype=torch.float64).fill_(-float('inf')).to(**self.tpdv)
        best_actions = torch.empty(batch_size,self.n_agents, 1, dtype=torch.int64).to(self.device) #[1,8,1]
        # Without edges (or iterations), CG would be the same as VDN: mean(f_i)
        utils_Q = best_value.new_zeros(batch_size,self.num_factor, self.act_dim, self.act_dim, self.act_dim).to(**self.tpdv) #[1,5,5]
        utils_a = best_value.new_zeros(batch_size,self.n_agents, self.act_dim).to(**self.tpdv) #[1,8,5]
        avail_a = best_value.new_zeros(batch_size,self.n_agents, self.act_dim).to(**self.tpdv)
        if available_actions is not None:
            avail_a = avail_a.masked_fill(available_actions == 0, -float('inf'))
        # Perform message passing for self.iterations: [0] are messages to *edges_to*, [1] are messages to *edges_from*   
        if num_edges > 0 and self.args.msg_iterations > 0:
            messages_a2Q = best_value.new_zeros(num_edges, self.act_dim, self.act_dim, self.act_dim).to(**self.tpdv) #[1,11,5]
            messages_Q2a = best_value.new_zeros(num_edges, self.act_dim).to(**self.tpdv) #[1,11,5] 
            zeros_Q2a = torch.full_like(messages_Q2a, 0).to(**self.tpdv)
            zeros_a2Q = torch.full_like(messages_a2Q, 0).to(**self.tpdv)
            inf_joint_Q2a = best_value.new_zeros(num_edges, self.act_dim, self.act_dim, self.act_dim).fill_(-float('inf')).to(**self.tpdv) 
            for iteration in range(self.args.msg_iterations):               
                # Recompute messages: joint utility for each edge: "sender Q-value"-"message from receiver"+payoffs/
                #update Q->a
                #import pdb;pdb.set_trace()
                joint_Q2a = utils_Q[adj_edge[0],adj_edge[2]] - messages_a2Q + f_q[adj_edge[0],adj_edge[2]]
                joint_Q2a = torch.where(torch.isnan(joint_Q2a), inf_joint_Q2a, joint_Q2a) 
                # Maximize the joint Q-value over the action of the sender
                for i in range(3):
                    if min(idx_type[i][0].shape) != 0:
                        messages_Q2a[idx_type[i]] = joint_Q2a[idx_type[i]].max(dim=num_dim[i][0])[0].max(dim=num_dim[i][1]-1)[0]
                # Normalization as in Kok and Vlassis (2006) and Wainwright et al. (2004)
                if self.args.msg_normalized:
                    messages_Q2a -= torch.where(torch.isinf(messages_Q2a), zeros_Q2a, messages_Q2a).mean(dim=-1, keepdim=True)
                # Create the current utilities of all agents, based on the messages 
                utils_a = avail_a + torch_scatter.scatter_add(src=messages_Q2a, index=var_edge, dim=0).reshape(batch_size, self.n_agents,-1)
                
                #update a->Q
                if iteration%2 == 0:
                    joint_a2Q = utils_a[adj_edge[0],adj_edge[1]] - messages_Q2a #[batch_size,num_edge,act_dim] = [1,16,5] 
                else:
                    joint_a2Q = lamda * joint_a2Q + (1-lamda) * (utils_a[adj_edge[0],adj_edge[1]] - messages_Q2a)

                for i in range(3):
                    if min(idx_type[i][0].shape) != 0:
                        messages_a2Q[idx_type[i][0]] = joint_a2Q[idx_type[i][0]].unsqueeze(num_dim[i][0]).unsqueeze(num_dim[i][1])
                '''if self.args.msg_normalized:
                    messages_a2Q -= torch.where(torch.isfinite(messages_a2Q), messages_a2Q, zeros_a2Q).mean(dim=-1, keepdim=True)'''

                utils_Q = torch_scatter.scatter_add(src=messages_a2Q, index=f_edge, dim=0).reshape(batch_size, self.num_factor,self.act_dim, self.act_dim, self.act_dim)   
                # Anytime extension (Kok and Vlassis, 2006)
                if self.args.msg_anytime:
                    # Find currently best actions and the (true) value of these actions 
                    actions = utils_a.max(dim=-1, keepdim=True)[1]
                    
                    value = self.q_values(in_q_batch, actions,idx_node_order,batch_size)
                    # Update best_actions only for the batches that have a higher value than best_value
                    change = value > best_value
                    if batch_size == 1:
                        f_value = self.q_local_values(f_q, actions,idx_node_order)
                        best_f_value[change] = f_value[change]     
                    best_value[change] = value[change]
                    best_actions[change] = actions[change]
                    #best_margin_value[change] = margin_value[change]      
        # Return the greedy actions and the corresponding message output averaged across agents        
        if not self.args.msg_anytime or num_edges == 0 or self.args.msg_iterations <= 0:
            _, best_actions = utils_a.max(dim=-1, keepdim=True)
        return best_actions, best_value, None, best_f_value
      
    def get_actions(self, obs_batch, available_actions=None, t_env=None, explore=False,adj_input = None,no_sequence = False,dones=None):
        if len(obs_batch.shape) == 3:
            batch_size = obs_batch.shape[0]
        else:
            batch_size = 1
            
        q_batch, idx_node_order, adj, num_edges = self.get_rnn_batch(obs_batch,adj_input,batch_size,no_sequence,dones)
        actions, best_value,best_margin_value, best_f_value = self.greedy(adj,q_batch,idx_node_order,available_actions,num_edges,batch_size)

        actions = actions.squeeze()
        # mask the available actions by giving -inf q values to unavailable actions
        if self.multidiscrete:
            onehot_actions = []
            for i in range(len(self.act_dim)):
                greedy_action = actions[i]
                if explore:
                    eps = self.exploration.eval(t_env)
                    rand_number = np.random.rand(self.n_agents)
                    # random actions sample uniformly from action space
                    random_action = Categorical(logits=torch.ones(self.n_agents, self.act_dim[i])).sample().numpy()
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
                rand_numbers = np.random.rand(self.n_agents)
                # random actions sample uniformly from action space
                logits = avail_choose(torch.ones(self.n_agents, self.act_dim), available_actions)
                random_actions = Categorical(logits=logits).sample().numpy()
                take_random = (rand_numbers < eps).astype(int)
                #take_random表示在每一条轨迹初始多采取随机动作，后面多采取指定动作
                actions = (1 - take_random) * to_numpy(actions) + take_random * random_actions
                onehot_actions = make_onehot(actions, self.act_dim)
            else:
                onehot_actions = make_onehot(actions, self.act_dim)
                
        return onehot_actions, best_value, best_margin_value, best_f_value
      
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
            if self.use_vfunction:
                parameters_sum += self.v_network[num_orders].parameters()
        return parameters_sum

    def load_state(self, source_policy):
        self.rnn_network.load_state_dict(source_policy.rnn_network.state_dict())
        for num_orders in range(1,self.highest_orders+1):
            self.q_network[num_orders].load_state_dict(source_policy.q_network[num_orders].state_dict())
            if self.use_vfunction:
                self.v_network[num_orders].load_state_dict(source_policy.v_network[num_orders].state_dict())