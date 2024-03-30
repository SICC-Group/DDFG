import torch
import copy
from offpolicy.utils.util import soft_update, huber_loss, mse_loss, to_torch
import numpy as np
from offpolicy.utils.popart import PopArt
import time


class R_MFG:
    def __init__(self, args, num_agents, policies, adj_network, policy_mapping_fn, device=torch.device("cuda:0"), episode_length=25,vdn=False):
        """
        Trainer class for QMix with MLP policies. See parent class for more information.
        :param vdn: (bool) whether the algorithm in use is VDN.
        """
        self.args = args
        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta
        self.use_dyn_graph = self.args.use_dyn_graph
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = self.args.lr
        self.adj_lr = self.args.adj_lr
        self.tau = self.args.tau
        self.opti_eps = self.args.opti_eps
        self.opti_alpha = self.args.opti_alpha
        self.weight_decay = self.args.weight_decay
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.highest_orders = args.highest_orders
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}
        if self.use_popart:
            self.value_normalizer = {policy_id: PopArt(1) for policy_id in self.policies.keys()}

        multidiscrete_list = None
        if any([isinstance(policy.act_dim, np.ndarray) for policy in self.policies.values()]):
            # multidiscrete
            multidiscrete_list = [len(self.policies[p_id].act_dim) *
                                  len(self.policy_agents[p_id]) for p_id in self.policy_ids]

        # target policies/networks
        self.adj_network = adj_network
        self.target_policies = {p_id: copy.deepcopy(self.policies[p_id]) for p_id in self.policy_ids}
        #self.target_mixer = copy.deepcopy(self.mixer)

        # collect all trainable parameters: each policy parameters, and the mixer parameters
        self.policy_parameters = []
        self.adj_parameters = []
        for policy in self.policies.values():
            self.policy_parameters += policy.parameters()
        self.policy_optimizer = torch.optim.Adam(params=self.policy_parameters, lr=self.lr, eps=self.opti_eps)
        #Adam
        self.adj_parameters += self.adj_network.parameters()
        self.adj_optimizer = torch.optim.Adam(params=self.adj_parameters, lr=self.adj_lr, eps=self.opti_eps)

        if args.use_double_q:
            print("double Q learning will be used")

    def train_policy_on_batch(self, batch, use_same_share_obs=None):
        """See parent class."""

        obs_batch, cent_obs_batch, \
        act_batch, rew_batch, \
        dones_batch, dones_env_batch, \
        avail_act_batch, adj, \
        importance_weights, idxes = batch
        dones_env_batch = to_torch(dones_env_batch[self.policy_ids[0]]).to(**self.tpdv)
        # individual agent q values: each element is of shape (batch_size, 1)
        qs = []
        target_qs = []
        #import pdb;pdb.set_trace()
        for p_id in self.policy_ids:

            policy_qs = []
            target_policy_qs = []
            policy = self.policies[p_id]
            target_policy = self.target_policies[p_id]
            # get data related to the policy id
            
            rewards = to_torch(rew_batch[p_id][0]).to(**self.tpdv) #[25,32,1]
            curr_obs_batch = to_torch(obs_batch[p_id]).transpose(0,2) #[3,26,32,18] 
            curr_act_batch = to_torch(act_batch[p_id]).transpose(0,2).to(**self.tpdv)  #[3,25,32,5] 
            adj = to_torch(adj[p_id])[0][0].to(self.device)
            
            if avail_act_batch[p_id] is not None:
                curr_avail_act_batch = to_torch(avail_act_batch[p_id]).transpose(0,2).transpose(0,1).to(self.device)
            else:
                curr_avail_act_batch = None
            act_dim = curr_act_batch.shape[3]
            step = rewards.shape[0] + 1
            batch_size = curr_obs_batch.shape[0] 
            stacked_act_batch = torch.cat(list(curr_act_batch), dim=-2) #[25,256,5]
            stacked_obs_batch = torch.cat(list(curr_obs_batch), dim=-2) #[26,256,48]
            pol_prev_act_buffer_seq = torch.cat((torch.zeros(1, batch_size*self.num_agents, act_dim).to(**self.tpdv),stacked_act_batch)) #[26,256,5]
            stacked_act_batch_ind = stacked_act_batch.max(dim=-1)[1] #[25,256]
            
            adj_input = adj
            #torch.cat([adj[0][0],torch.eye(self.num_agents,dtype=torch.int64)],dim=1).to(self.device)
            rnn_states_1 = policy.init_hidden(self.num_agents,batch_size)
           # rnn_states_2 = policy.init_hidden(self.num_agents,batch_size)
            target_rnn_states = target_policy.init_hidden(self.num_agents,batch_size)
            rnn_obs_batch_1, _, no_sequence = policy.get_hidden_states(stacked_obs_batch,pol_prev_act_buffer_seq,rnn_states_1)
            #rnn_obs_batch_2, _, no_sequence = policy.get_hidden_states(stacked_obs_batch,pol_prev_act_buffer_seq,rnn_states_2)
            target_rnn_obs_batch, _, _ = target_policy.get_hidden_states(stacked_obs_batch,pol_prev_act_buffer_seq,target_rnn_states)
            
            curr_act_batch_ind = stacked_act_batch_ind.reshape((step-1)*batch_size,self.num_agents,-1)
            obs_1 = rnn_obs_batch_1.reshape(step,batch_size,self.num_agents,-1)
            #obs_2 = rnn_obs_batch_2.reshape(step,batch_size,self.num_agents,-1)
            #.reshape(step-1,batch_size,self.num_agents,-1)
            target_obs = target_rnn_obs_batch.reshape(step,batch_size,self.num_agents,-1)
            pol_all_q_out = policy.get_q_values(obs_1[:-1].reshape((step-1)*batch_size,self.num_agents,-1),curr_act_batch_ind,adj_input,no_sequence)
            policy_qs.append(pol_all_q_out)
            '''greedy, _,_ ,_= policy.get_actions(obs_1[1:].reshape((step-1)*batch_size,self.num_agents,-1), curr_avail_act_batch[1:].reshape((step-1)*batch_size,self.num_agents,-1), None, False, adj_input, no_sequence)
            curr_nact_batch_ind = torch.from_numpy(greedy).max(dim=-1)[1].to(self.device)
            targ_pol_next_qs = target_policy.get_q_values(target_obs[1:].reshape((step-1)*batch_size,self.num_agents,-1), curr_nact_batch_ind.unsqueeze(dim=-1), adj_input,no_sequence)
            target_policy_qs.append(targ_pol_next_qs)'''
            #import pdb;pdb.set_trace()
            new_bs = (step-1)*batch_size//4
            with torch.no_grad():  
                if self.args.use_double_q:
                    for i in range(4):
                        greedy, _,_ ,_= policy.get_actions(obs_1[1:,2*i:2*i+2].reshape(new_bs,self.num_agents,-1), curr_avail_act_batch[1:,2*i:2*i+2].reshape(new_bs,self.num_agents,-1), None, False, adj_input, no_sequence)
                        curr_nact_batch_ind = torch.from_numpy(greedy).max(dim=-1)[1].to(self.device)
                        targ_pol_next_qs = target_policy.get_q_values(target_obs[1:,2*i:2*i+2].reshape(new_bs,self.num_agents,-1), curr_nact_batch_ind.unsqueeze(dim=-1), adj_input,no_sequence)
                        target_policy_qs.append(targ_pol_next_qs.reshape(step-1,batch_size//4))
                else:
                    for n in range(batch_size):
                        _, targ_pol_next_qs = target_policy.get_actions(obs_2[1:,n], curr_avail_act_batch,adj_input[n],False)
                        target_policy_qs.append(targ_pol_next_qs)        
            qs.append(torch.stack(policy_qs, dim=0).transpose(0,1).reshape(step-1,batch_size,-1))
            target_qs.append(torch.cat(target_policy_qs, dim=1).unsqueeze(-1))
        # combine the agent q value sequences to feed into mixer networks
        curr_Q_tot = torch.cat(qs, dim=-1).to(self.device)
        next_step_Q_tot = torch.cat(target_qs, dim=-1).to(self.device)
        bad_transitions_mask = torch.cat((torch.zeros(1, batch_size, 1).to(**self.tpdv), dones_env_batch[:self.episode_length - 1, :, :]))
        # all agents must share reward, so get the reward sequence for an agent
        if self.use_popart:
            Q_tot_targets = rewards + (1 - dones_env_batch) * self.args.gamma * \
                self.value_normalizer[p_id].denormalize(next_step_Q_tot)
            Q_tot_targets = self.value_normalizer[p_id](Q_tot_targets)
        else:
            Q_tot_targets = rewards + (1 - dones_env_batch) * self.args.gamma * next_step_Q_tot

        error = (curr_Q_tot - Q_tot_targets.detach()) * (1 - bad_transitions_mask)
        
        if self.use_per:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).flatten()
            else:
                loss = mse_loss(error).flatten()
            loss = (loss * to_torch(importance_weights).to(**self.tpdv)).sum()  / ((1 - bad_transitions_mask).sum())
            # new priorities are a combination of the maximum TD error across sequence and the mean TD error across sequence
            new_priorities = error.abs().cpu().detach().numpy().flatten() + self.per_eps
        else:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).sum() / ((1 - bad_transitions_mask).sum())
            else:
                loss = mse_loss(error).sum() / ((1 - bad_transitions_mask).sum())
            new_priorities = None
            
        train_info = {}
        self.policy_optimizer.zero_grad()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_parameters, self.args.max_grad_norm)
        #train_info['grad_max_after'] = self.policy_parameters[0].grad.data.norm(2)
        self.policy_optimizer.step()

        train_info['loss'] = loss
        train_info['grad_norm'] = grad_norm
        train_info['Q_tot'] = curr_Q_tot.mean()
        return train_info, new_priorities, idxes
      
    def hard_target_updates(self):
        """Hard update the target networks."""
        print("hard update targets")
        
        for policy_id in self.policy_ids:
            self.target_policies[policy_id].load_state(
                self.policies[policy_id])

    def soft_target_updates(self):
        """Soft update the target networks."""
        for policy_id in self.policy_ids:
            soft_update(
                self.target_policies[policy_id], self.policies[policy_id], self.tau)

    def prep_training(self):
        """See parent class."""
        self.adj_network.train()
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.train()
            for num_orders in range(1,self.highest_orders+1):
                self.policies[p_id].q_network[num_orders].train()
                self.target_policies[p_id].q_network[num_orders].train()

    def prep_rollout(self):
        """See parent class."""
      
        self.adj_network.eval()
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.eval()
            for num_orders in range(1,self.highest_orders+1):
                self.policies[p_id].q_network[num_orders].eval()
                self.target_policies[p_id].q_network[num_orders].eval()
        #self.mixer.eval()
        #self.target_mixer.eval()
