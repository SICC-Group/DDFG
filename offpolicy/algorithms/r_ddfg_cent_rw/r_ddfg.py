import torch
import copy
from offpolicy.utils.util import soft_update, huber_loss, mse_loss, to_torch, log_loss, update_linear_schedule
from offpolicy.utils.valuenorm import ValueNorm
import numpy as np
from offpolicy.utils.popart import PopArt
import time
from multiprocessing import Process,Pool,Queue
#from pathos.multiprocessing import ProcessingPool as Pool
#import pathos.multiprocessing as mp
class R_DDFG:
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
        self.clip_param = self.args.clip_param
        self.use_vfunction = self.args.use_vfunction
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = self.args.lr
        self.adj_lr = self.args.adj_lr
        self.tau = self.args.tau
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.highest_orders = self.args.highest_orders
        self.use_dyn_graph = self.args.use_dyn_graph
        self.num_factor = self.args.num_factor
        self.entropy_coef = self.args.entropy_coef
        self._use_valuenorm = self.args.use_valuenorm
        self.adj_max_grad_norm = self.args.adj_max_grad_norm
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}
        if self._use_valuenorm:
            self.value_normalizer = {policy_id: PopArt(1,self.device).to(self.device) for policy_id in self.policies.keys()}

        multidiscrete_list = None
        if any([isinstance(policy.act_dim, np.ndarray) for policy in self.policies.values()]):
            # multidiscrete
            multidiscrete_list = [len(self.policies[p_id].act_dim) *
                                  len(self.policy_agents[p_id]) for p_id in self.policy_ids]

        # target policies/networks
        self.adj_network = adj_network
        self.target_policies = {p_id: copy.deepcopy(self.policies[p_id]) for p_id in self.policy_ids}
        self.policy_parameters = []
        for policy in self.policies.values():
            self.policy_parameters += policy.parameters()
        self.policy_optimizer = torch.optim.Adam(params=self.policy_parameters, lr=self.lr, eps=self.opti_eps)
        self.adj_parameters = []
        self.adj_parameters += self.adj_network.parameters()
        self.adj_optimizer = torch.optim.Adam(params=self.adj_parameters, lr=self.adj_lr, eps=self.opti_eps)
        
        if args.use_double_q:
            print("double Q learning will be used")
    
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.adj_optimizer, episode, episodes, self.adj_lr)
        update_linear_schedule(self.policy_optimizer, episode, episodes, self.lr)
        
    def train_policy_on_batch(self, batch, use_same_share_obs=None):
        """See parent class."""

        obs_batch, cent_obs_batch, \
        act_batch, rew_batch, \
        dones_batch, dones_env_batch, \
        avail_act_batch, adj, \
        prob_adj, idxes = batch
        dones_env_batch = to_torch(dones_env_batch[self.policy_ids[0]]).transpose(0,1).to(**self.tpdv)
        
        # individual agent q values: each element is of shape (batch_size, 1)
        qs = []
        target_qs = []
        v = []
        target_v = []
        
        for p_id in self.policy_ids:
    
            policy = self.policies[p_id]
            target_policy = self.target_policies[p_id]
            # get data related to the policy id
            dones = to_torch(dones_batch[p_id]).permute(1,2,0,3).to(self.device)
            rewards = to_torch(rew_batch[p_id][0]).transpose(0,1).to(**self.tpdv) #[25,32,1]
            curr_obs_batch = to_torch(obs_batch[p_id]).transpose(0,2)#[3,26,32,18] 
            curr_act_batch = to_torch(act_batch[p_id]).transpose(0,2).to(**self.tpdv)  #[3,25,32,5] 
            adj = to_torch(adj[p_id])

            if avail_act_batch[p_id] is not None:
                curr_avail_act_batch = to_torch(avail_act_batch[p_id]).transpose(0,2).to(self.device)
            else:
                curr_avail_act_batch = None
                
            act_dim = curr_act_batch.shape[3]
            step = rewards.shape[1] + 1
            batch_size = curr_obs_batch.shape[0] 
            policy_qs = torch.zeros([batch_size,step-1,1],dtype=torch.float32).to(self.device)
            target_policy_qs = torch.zeros([batch_size,step-1,1],dtype=torch.float32).to(self.device)
            dones = torch.cat((torch.zeros(1,batch_size,self.num_agents, 1).to(**self.tpdv), dones),dim=0)
            dones_batch = torch.cat((torch.zeros(batch_size,1, 1).to(**self.tpdv), dones_env_batch),dim=1)
            bad_transitions_mask = dones_batch[:,:-1]
             
            stacked_act_batch = torch.cat(list(curr_act_batch), dim=-2) #[25,256,5]
            stacked_obs_batch = torch.cat(list(curr_obs_batch), dim=-2) #[26,256,48]
            pol_prev_act_buffer_seq = torch.cat((torch.zeros(1, batch_size*self.num_agents, act_dim).to(**self.tpdv),stacked_act_batch)) #[26,256,5]
            stacked_act_batch_ind = stacked_act_batch.max(dim=-1)[1] #[25,256]
            
            if self.use_dyn_graph:
                adj_input = torch.cat([adj,torch.eye(self.num_agents,dtype=torch.int64).repeat(step,batch_size,1,1)],dim=3).to(self.device)
            else:
                adj_input = torch.eye(self.num_agents,dtype=torch.int64).repeat(step,batch_size,1,1).to(self.device)
            
            rnn_states_1 = policy.init_hidden(self.num_agents,batch_size)
            target_rnn_states = target_policy.init_hidden(self.num_agents,batch_size)

            rnn_obs_batch_1, _, no_sequence = policy.get_hidden_states(stacked_obs_batch,pol_prev_act_buffer_seq,rnn_states_1)
            target_rnn_obs_batch, _, _ = target_policy.get_hidden_states(stacked_obs_batch,pol_prev_act_buffer_seq,target_rnn_states)
            curr_act_batch_ind = stacked_act_batch_ind.reshape((step-1)*batch_size,self.num_agents,-1)
            obs_q = rnn_obs_batch_1[:-1].reshape((step-1)*batch_size,self.num_agents,-1)
            adj_input_q = adj_input[:-1].reshape((step-1)*batch_size,self.num_agents,-1)
            dones_q = dones[:-1].reshape((step-1)*batch_size,self.num_agents,-1)
            policy_qs = policy.get_q_values(obs_q,curr_act_batch_ind,adj_input_q,no_sequence,dones_q)
                              
            obs_qtot = rnn_obs_batch_1[1:].reshape((step-1)*batch_size,self.num_agents,-1)
            adj_input_qtot = adj_input[1:].reshape((step-1)*batch_size,self.num_agents,-1)
            dones_qtot = dones[1:].reshape((step-1)*batch_size,self.num_agents,-1)
            target_obs = target_rnn_obs_batch[1:].reshape((step-1)*batch_size,self.num_agents,-1)
            curr_avail_act = curr_avail_act_batch.transpose(0,1)[1:].reshape((step-1)*batch_size,self.num_agents,-1)
            with torch.no_grad():
                greedy,_,_,_ = policy.get_actions(obs_qtot, curr_avail_act, None, False, adj_input_qtot, no_sequence,dones_qtot) 
                curr_nact_batch_ind = torch.from_numpy(greedy).max(dim=-1)[1].to(self.device)
                target_policy_qs = target_policy.get_q_values(target_obs, curr_nact_batch_ind.unsqueeze(dim=-1), adj_input_qtot,no_sequence,dones_qtot)
             
            qs.append(policy_qs.reshape(step-1,batch_size).transpose(0,1))
            target_qs.append(target_policy_qs.reshape(step-1,batch_size).transpose(0,1))
            
            if self.use_vfunction:
                policy_v = policy.get_v_values(obs_q,adj_input_q,no_sequence,dones_q)
                with torch.no_grad():
                    target_policy_v =target_policy.get_v_values(target_obs,adj_input_qtot,no_sequence,dones_qtot)
                v.append(policy_v.reshape(step-1,batch_size).transpose(0,1))
                target_v.append(target_policy_v.reshape(step-1,batch_size).transpose(0,1))
        # combine the agent q value sequences to feed into mixer networks
        curr_Q_tot = torch.cat(qs, dim=-1).unsqueeze(-1)
        next_step_Q_tot = torch.cat(target_qs, dim=-1).unsqueeze(-1)
        
        # all agents must share reward, so get the reward sequence for an agent
        if self._use_valuenorm:
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
        if self.use_vfunction:   
            curr_v_tot = torch.cat(v, dim=-1).unsqueeze(-1)
            next_step_v_tot = torch.cat(target_v, dim=-1).unsqueeze(-1)
            v_tot_targets = rewards + (1 - dones_env_batch) * self.args.gamma * next_step_v_tot
            error_v = (curr_v_tot - v_tot_targets.detach()) * (1 - bad_transitions_mask)
            loss_v = mse_loss(error_v).sum() / ((1 - bad_transitions_mask).sum())
            newloss = loss+loss_v
        self.policy_optimizer.zero_grad()
        if self.use_vfunction:  
            newloss.backward()
        else:
            loss.backward()
        #import pdb;pdb.set_trace()
        #for param in self.policy_parameters:
            #if param.grad is not None:
                #print("param=",param.shape)
                #print("grad=",param.grad)
                #print("grad_sum=",param.grad.data.norm(2))
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_parameters, self.args.max_grad_norm)
        self.policy_optimizer.step()
        train_info = {}
        train_info['loss'] = loss.cpu().item() 
        train_info['Q_tot'] = curr_Q_tot.mean().cpu().item() 
        train_info['grad_norm'] = grad_norm.cpu().item() 
        return train_info, new_priorities, idxes
      
    def train_adj_on_batch(self, batch, use_adj_init, use_same_share_obs=None):
        """See parent class."""

        obs_batch, share_obs_batch, dones_batch, \
        dones_env_batch, adj_batch, prob_adj_batch, \
        advantages_batch, f_advts_batch, rnn_obs_batch = batch
        # individual agent q values: each element is of shape (batch_size, 1)
        qs = []
        target_qs = []
        tarprob_adj = []
        tarprob_extra = []
        adj_entropy = []
        #import pdb;pdb.set_trace()
        adj = to_torch(adj_batch)#[batch_size,step,num_agent,-1]
        batch_size = adj.shape[0] * adj.shape[1] 
        adj = adj.reshape(batch_size,self.num_agents,-1).to(**self.tpdv) 
        state_batch = to_torch(share_obs_batch).reshape(batch_size,-1).to(**self.tpdv)  
        dones = to_torch(dones_batch).reshape(batch_size,self.num_agents,-1).to(self.device)
        dones_env = to_torch(dones_env_batch).reshape(batch_size,-1).to(**self.tpdv)
        prob_adj = to_torch(prob_adj_batch).reshape(batch_size,self.num_agents,-1).to(**self.tpdv)
        f_advts = to_torch(f_advts_batch).reshape(batch_size,self.num_factor,-1).to(**self.tpdv)
        rnn_obs= to_torch(rnn_obs_batch).reshape(batch_size,self.num_agents,-1).to(**self.tpdv) #[batch_size,step-1,1]
        for p_id in self.policy_ids:

            target_prob_adj, _ , entropy  =  self.adj_network.sample(rnn_obs,state_batch,use_adj_init)
            target_prob = torch.where(adj==1,target_prob_adj,torch.ones_like(target_prob_adj,dtype=torch.float32))
            adj_entropy.append(entropy)
            tarprob_adj.append(torch.log(target_prob))
            
        tarlog_prob_adj = torch.cat(tarprob_adj, dim=-1)
        adj_entropy_batch = torch.cat(adj_entropy, dim=-1).unsqueeze(-1)
        log_prob_adj = torch.log(torch.where(adj==1,prob_adj,torch.ones_like(prob_adj,dtype=torch.float32)))

        if self.highest_orders == 3:
            sort_tar_proadj = torch.topk(tarlog_prob_adj, k=self.highest_orders, dim=1, largest=False)[0]
            sort_proadj = torch.topk(log_prob_adj, k=self.highest_orders, dim=1, largest=False)[0]
            idx1 = torch.tensor([[[2],[1],[0]]])
            idx2 = torch.tensor([[[1],[2],[0]]])
            idx_order2 = (adj.sum(-2)==1).unsqueeze(1)
            log_tar_1 = torch.where(idx_order2,sort_tar_proadj * idx1,sort_tar_proadj).sum(-2)
            log_tar_2 = torch.where(idx_order2,sort_tar_proadj * idx2,sort_tar_proadj).sum(-2)
            log_1 = torch.clamp(torch.where(idx_order2,sort_proadj * idx1,sort_proadj).sum(-2),min=-40)
            log_2 = torch.clamp(torch.where(idx_order2,sort_proadj * idx2,sort_proadj).sum(-2),min=-40)
            imp_weights = (torch.exp(log_tar_1)+torch.exp(log_tar_2))/(torch.exp(log_1)+torch.exp(log_2))
            imp_weights_multinomial = torch.where(adj.sum(-2)==1,imp_weights*imp_weights*imp_weights,imp_weights).unsqueeze(-1)       
        else:
            diff_log = torch.clamp(tarlog_prob_adj.sum(-2)-log_prob_adj.sum(-2),max=80) 
            imp_weights = torch.exp(diff_log)
            imp_weights_multinomial = torch.where(adj.sum(-2)==1,imp_weights*imp_weights,imp_weights).unsqueeze(-1)       
     
        bad_transitions_mask = dones_env
        clamp_imp_weights = torch.clamp(imp_weights_multinomial, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surr1 = imp_weights_multinomial * f_advts
        surr2 = clamp_imp_weights * f_advts
        rl_loss = -(torch.sum(torch.min(surr1, surr2),dim=-2) * (1 - bad_transitions_mask)).sum()  / ((1 - bad_transitions_mask).sum() +1e-3)       
        entropy_loss = (adj_entropy_batch * (1 - dones_env)).sum()  / ((1 - dones_env).sum() + 1e-3)
        loss = rl_loss - self.entropy_coef*entropy_loss

        self.adj_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.adj_parameters, self.adj_max_grad_norm)
        self.adj_optimizer.step()
        
        train_info = {}
        train_info['advantage'] = 0
        #mask_imp_weights = clamp_imp_weights * (1 - bad_transitions_mask)
        train_info['clamp_ratio'] = 0
        train_info['rl_loss'] = rl_loss.cpu().item()
        #rl_loss.item()
        train_info['entropy_loss'] = entropy_loss.cpu().item() 
        #agent_loss.item()
        train_info['grad_norm'] = grad_norm.cpu().item() 
        #loss.item()

        return train_info, None, None
      
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
            self.target_policies[p_id].rnn_network.train()
            for num_orders in range(1,self.highest_orders+1):
                self.policies[p_id].q_network[num_orders].train()
                self.target_policies[p_id].q_network[num_orders].train()
                if self.use_vfunction:
                    self.policies[p_id].v_network[num_orders].train()
                    self.target_policies[p_id].v_network[num_orders].train()

    def prep_rollout(self):
        """See parent class."""

        self.adj_network.eval()
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.eval()
            self.target_policies[p_id].rnn_network.eval()
            for num_orders in range(1,self.highest_orders+1):
                self.policies[p_id].q_network[num_orders].eval()
                self.target_policies[p_id].q_network[num_orders].eval()
                if self.use_vfunction:
                    self.policies[p_id].v_network[num_orders].eval()
                    self.target_policies[p_id].v_network[num_orders].eval()
        #self.mixer.eval()
        #self.target_mixer.eval()
