import torch
import copy
from offpolicy.utils.util import soft_update, huber_loss, mse_loss, to_torch
import numpy as np
from offpolicy.utils.popart import PopArt
import time


class R_CASEC:
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
        self.independent_p_q = self.args.independent_p_q
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
        self.action_encoder_params = []
        for policy in self.policies.values():
            self.policy_parameters += policy.parameters()
            self.action_encoder_params += policy.action_encoder_params()
        #self.policy_optimizer = torch.optim.Adam(params=self.policy_parameters, lr=self.lr, eps=self.opti_eps)
        #Adam
        #self.action_encoder_optimiser = torch.optim.Adam(params=self.action_encoder_params, lr=self.lr,eps=self.opti_eps)
        
        self.policy_optimizer = torch.optim.RMSprop(params=self.policy_parameters, lr=self.lr, alpha=0.99, eps=0.00001)
        self.action_encoder_optimiser = torch.optim.RMSprop(params=self.action_encoder_params, lr=self.lr, alpha=0.99, eps=0.00001)
        
        self.adj_parameters += self.adj_network.parameters()
        #self.adj_optimizer = torch.optim.Adam(params=self.adj_parameters, lr=self.adj_lr, eps=self.opti_eps)
        self.adj_optimizer =torch.optim.RMSprop(params=self.adj_parameters, lr=self.adj_lr, alpha=0.99, eps=0.00001)

        if args.use_double_q:
            print("double Q learning will be used")

    def train_policy_on_batch(self, batch, action_repr_updating, use_same_share_obs=None):
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
            curr_obs_batch = to_torch(obs_batch[p_id]).transpose(0,2).to(**self.tpdv) #[3,26,32,18] 
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
            f_i_left, delta_ij_left, q_ij_left = [], [], []
            rnn_states = policy.init_hidden(self.num_agents,batch_size)
            p_rnn_states = policy.init_p_hidden(self.num_agents,batch_size)
            target_rnn_states = target_policy.init_hidden(self.num_agents,batch_size)
            target_p_rnn_states = target_policy.init_p_hidden(self.num_agents,batch_size)
            rnn_obs_batch, _, no_sequence = policy.get_hidden_states(stacked_obs_batch,pol_prev_act_buffer_seq,rnn_states)
            target_rnn_obs_batch, _, _ = target_policy.get_hidden_states(stacked_obs_batch,pol_prev_act_buffer_seq,target_rnn_states)
            if self.independent_p_q:
                p_rnn_obs_batch, _, _ = policy.get_p_hidden_states(stacked_obs_batch,pol_prev_act_buffer_seq,p_rnn_states)
                target_p_rnn_obs_batch, _, _ = target_policy.get_p_hidden_states(stacked_obs_batch,pol_prev_act_buffer_seq,target_p_rnn_states)
                p_obs = p_rnn_obs_batch.reshape(step,batch_size,self.num_agents,-1)
                target_p_obs = target_p_rnn_obs_batch.reshape(step,batch_size,self.num_agents,-1)
            curr_act_batch_ind = stacked_act_batch_ind.reshape(step-1,batch_size,self.num_agents,-1)
            obs = rnn_obs_batch.reshape(step,batch_size,self.num_agents,-1)
            target_obs = target_rnn_obs_batch.reshape(step,batch_size,self.num_agents,-1)
            
            for t in range(step-1):  
                if self.independent_p_q:
                    agent_outs, f_i, delta_ij, q_ij = policy.get_q_values(obs[t],curr_act_batch_ind[t],p_obs[t],no_sequence)
                else:
                    agent_outs, f_i, delta_ij, q_ij = policy.get_q_values(obs[t],curr_act_batch_ind[t],no_sequence=no_sequence)
                policy_qs.append(agent_outs)  # [t+1,(bs,1)]
                f_i_left.append(f_i)
                delta_ij_left.append(delta_ij)
                q_ij_left.append(q_ij)
            policy_qs = torch.stack(policy_qs, dim=1)
            
            f_i_left = torch.stack(f_i_left, dim=1)
            delta_ij_left = torch.stack(delta_ij_left, dim=1)
            q_ij_left_all = torch.stack(q_ij_left, dim=1)
            q_ij_left = torch.stack(q_ij_left, dim=1)[:, :-1]   
            
            target_f_i, target_delta_ij, target_q_ij, target_his_cos_sim = [], [], [], []
            for t in range(step):
                if self.independent_p_q:
                    f_i, delta_ij, q_ij, his_cos_similarity = target_policy.caller_ip_q(target_obs[t],target_p_obs[t],no_sequence=no_sequence)
                else:
                    f_i, delta_ij, q_ij, his_cos_similarity = target_policy.caller_ip_q(target_obs[t],no_sequence=no_sequence)
                target_f_i.append(f_i)  # [t+1,(bs,n,|A|)]
                target_delta_ij.append(delta_ij)  # [t+1,(bs,n,n,|A|,|A|)]
                target_q_ij.append(q_ij)  # [t+1,(bs,n,n,|A|,|A|)]
                target_his_cos_sim.append(his_cos_similarity)
                
            target_f_i = torch.stack(target_f_i[1:], dim=1)  # (bs,t,n,|A|)
            target_delta_ij_all = torch.stack(target_delta_ij, dim=1)  # (bs,t,n,n,|A|,|A|)
            target_q_ij_all = torch.stack(target_q_ij, dim=1)  # (bs,t,n,n,|A|,|A|)
            target_his_cos_sim_all = torch.stack(target_his_cos_sim, dim=1)  # (bs,t,n,n,|A|,|A|)
            target_q_ij = torch.stack(target_q_ij[1:], dim=1)  # (bs,t,n,n,|A|,|A|)
            
            for t in range(step-1):
                f_i = f_i_left[:, t].detach()
                delta_ij = delta_ij_left[:, t].detach()
                q_ij = q_ij_left_all[:, t].detach()
                target_agent_outs = policy.max_sum(curr_avail_act_batch[t], f_i=f_i, delta_ij=delta_ij, q_ij=q_ij,
                                                     target_delta_ij=target_delta_ij_all[:, t].detach(),
                                                     target_q_ij=target_q_ij_all[:, t].detach(),
                                                     target_his_cos_sim=target_his_cos_sim_all[:, t],batch_size=batch_size)  # (bs,n,|A|)
                target_policy_qs.append(target_agent_outs)  # [t,(bs,n,|A|)]
            
            target_policy_qs = torch.stack(target_policy_qs[1:], dim=1)
            target_policy_qs[curr_avail_act_batch[1:-1].permute(1, 0, 2, 3) == 0] = -9999999  # Q values
            
            target_policy_qs = target_policy_qs.clone().detach()  # return a new Tensor, detached from the current graph
            cur_max_actions = target_policy_qs.max(dim=3, keepdim=True)[1]  # indices instead of values
            target_f_i_gather = torch.gather(target_f_i[:,:-1], index=cur_max_actions, dim=-1)  # (bs,t,n,1)
            agent_actions_gather_i = cur_max_actions.unsqueeze(dim=3).unsqueeze(dim=-1).repeat(1, 1, 1,self.num_agents, 1,act_dim)
            agent_actions_gather_j = cur_max_actions.unsqueeze(dim=2).unsqueeze(dim=-2).repeat(1, 1, self.num_agents,1, 1, 1)
            target_q_ij_gather = torch.gather(target_q_ij[:,:-1], index=agent_actions_gather_i, dim=-2)
            target_q_ij_gather = torch.gather(target_q_ij_gather, index=agent_actions_gather_j, dim=-1)
            target_q_ij_gather = target_q_ij_gather.squeeze()  # * self.mac.adj  # (bs,t,n,n)
            target_max_qvals = target_f_i_gather.squeeze(dim=-1).mean(dim=-1) + target_q_ij_gather.sum(dim=-1).sum(dim=-1) / self.num_agents / (self.num_agents - 1)
            target_max_qvals.unsqueeze_(dim=-1)  # (bs,t,1)
            
            
            qs.append(policy_qs.transpose(0,1))
            target_qs.append(target_max_qvals.transpose(0,1))
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
            Q_tot_targets = rewards[:-1] + (1 - dones_env_batch[:-1]) * self.args.gamma * next_step_Q_tot

        error = (curr_Q_tot[:-1] - Q_tot_targets.detach()) * (1 - bad_transitions_mask[:-1])
        
        if self.use_per:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).flatten()
            else:
                loss = mse_loss(error).flatten()
            loss = (loss * to_torch(importance_weights).to(**self.tpdv)).sum()  / ((1 - bad_transitions_mask).sum())
            new_priorities = error.abs().cpu().detach().numpy().flatten() + self.per_eps
        else:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).sum() / ((1 - bad_transitions_mask).sum())
            else:
                loss = mse_loss(error).sum() / ((1 - bad_transitions_mask[:-1]).sum())
            new_priorities = None

        var_loss = (q_ij_left.view(-1, step - 2,self.num_agents * self.num_agents * act_dim,act_dim).var(-1)).sum(-1).unsqueeze(-1) / self.num_agents / (self.num_agents - 1) / act_dim
        masked_var_loss = var_loss.transpose(0,1) * (1 - bad_transitions_mask[:-1])
        loss = loss + 0.0001 * (masked_var_loss.sum() / (1 - bad_transitions_mask[:-1]).sum())
        train_info = {}
        self.policy_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_parameters, self.args.max_grad_norm)
        #train_info['grad_max_after'] = self.policy_parameters[0].grad.data.norm(2)
        self.policy_optimizer.step()
        
        pred_obs_loss = None
        pred_r_loss = None
        pred_grad_norm = None
        if action_repr_updating:
            no_pred = []
            r_pred = []
            obs_batch = stacked_obs_batch.reshape(step,batch_size,self.num_agents,-1)
            act_batch = stacked_act_batch.reshape(step-1,batch_size,self.num_agents,-1)
            for t in range(step-1):
                no_preds, r_preds = policy.action_repr_forward(obs_batch[t], act_batch[t])
                no_pred.append(no_preds)
                r_pred.append(r_preds)
            #import pdb;pdb.set_trace()
            no_pred = torch.stack(no_pred, dim=1).transpose(0,1)  # Concat over time
            r_pred = torch.stack(r_pred, dim=1).transpose(0,1)
            no = obs_batch[1:].detach().clone()
            repeated_rewards = rewards.detach().clone().unsqueeze(2).repeat(1, 1, self.num_agents, 1)

            pred_obs_loss = torch.sqrt(((no_pred - no) ** 2).sum(dim=-1)).mean()
            pred_r_loss = ((r_pred - repeated_rewards) ** 2).mean()
            pred_loss = pred_obs_loss + 10 * pred_r_loss
            self.action_encoder_optimiser.zero_grad()
            pred_loss.backward()
            pred_grad_norm = torch.nn.utils.clip_grad_norm_(self.action_encoder_params, self.args.max_grad_norm)
            self.action_encoder_optimiser.step()
        
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
            
    def update_action_repr(self):
         for policy_id in self.policy_ids:
            self.policies[policy_id].update_action_repr()
            
    def prep_training(self):
        """See parent class."""
        self.adj_network.train()
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.train()
            self.policies[p_id].action_encoder.train()
            self.target_policies[p_id].rnn_network.train()
            self.target_policies[p_id].action_encoder.train()
            if self.independent_p_q:
                self.policies[p_id].p_rnn_network.train()
                self.target_policies[p_id].p_rnn_network.train()
            for num_orders in range(1,self.highest_orders+1):
                self.policies[p_id].q_network[num_orders].train()
                self.target_policies[p_id].q_network[num_orders].train()

    def prep_rollout(self):
        """See parent class."""
      
        self.adj_network.eval()
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.eval()
            self.target_policies[p_id].action_encoder.eval()
            self.target_policies[p_id].rnn_network.eval()
            self.target_policies[p_id].action_encoder.eval()
            if self.independent_p_q:
                self.policies[p_id].p_rnn_network.eval()
                self.target_policies[p_id].p_rnn_network.eval()
            for num_orders in range(1,self.highest_orders+1):
                self.policies[p_id].q_network[num_orders].eval()
                self.target_policies[p_id].q_network[num_orders].eval()
        #self.mixer.eval()
        #self.target_mixer.eval()
