import torch
import copy
from offpolicy.utils.util import soft_update, huber_loss, mse_loss, to_torch
from offpolicy.algorithms.qplex.algorithm.dmaq_general import DMAQer
from offpolicy.algorithms.qplex.algorithm.dmaq_qatten import DMAQ_QattenMixer
from offpolicy.algorithms.base.trainer import Trainer
from offpolicy.utils.popart import PopArt
import numpy as np

class QPlex(Trainer):
    def __init__(self, args, num_agents, policies, policy_mapping_fn, device=torch.device("cuda:0"), episode_length=None,qatten=True):
        """
        Trainer class for recurrent QMix/VDN. See parent class for more information.
        :param episode_length: (int) maximum length of an episode.
        :param vdnl: (bool) whether the algorithm being used is VDN.
        """
        self.args = args
        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = self.args.lr
        self.tau = self.args.tau
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay

        if episode_length is None:
            self.episode_length = self.args.episode_length
        else:
            self.episode_length = episode_length
        
        self.num_agents = num_agents
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id])
            for policy_id in self.policies.keys()}
        self.qatten = qatten
        if self.use_popart:
            self.value_normalizer = {policy_id: PopArt(1) for policy_id in self.policies.keys()}

        self.use_same_share_obs = self.args.use_same_share_obs

        multidiscrete_list = None
        if any([isinstance(policy.act_dim, np.ndarray) for policy in self.policies.values()]):
            # multidiscrete
            multidiscrete_list = [len(self.policies[p_id].act_dim) *
                                  len(self.policy_agents[p_id]) for p_id in self.policy_ids]

        # mixer network
        if self.qatten:
            self.mixer = DMAQ_QattenMixer(args,self.num_agents, self.policies['policy_0'].central_obs_dim, self.policies['policy_0'].act_dim,self.policies['policy_0'].unit_dim, self.device)
        else:
            self.mixer = DMAQer(args,self.num_agents, self.policies['policy_0'].central_obs_dim, self.policies['policy_0'].act_dim,self.device)
            

        # target policies/networks
        self.target_policies = {p_id: copy.deepcopy(self.policies[p_id]) for p_id in self.policy_ids}
        self.target_mixer = copy.deepcopy(self.mixer)

        # collect all trainable parameters: each policy parameters, and the mixer parameters
        self.parameters = []
        for policy in self.policies.values():
            self.parameters += policy.parameters()
        self.parameters += self.mixer.parameters()
        #self.optimizer = torch.optim.Adam(params=self.parameters, lr=self.lr, eps=self.opti_eps)
        self.optimizer = torch.optim.RMSprop(params=self.parameters, lr=args.lr, alpha=0.99, eps=0.00001)

        if self.args.use_double_q:
            print("double Q learning will be used")

    def train_policy_on_batch(self, batch, update_policy_id=None):
        """See parent class."""
        # unpack the batch
        obs_batch, cent_obs_batch, \
        act_batch, rew_batch, \
        dones_batch, dones_env_batch, \
        avail_act_batch, adj, \
        importance_weights, idxes = batch
        
        if self.use_same_share_obs:
            cent_obs_batch = to_torch(cent_obs_batch[self.policy_ids[0]]).to(**self.tpdv)
        else:
            choose_agent_id = 0
            cent_obs_batch = to_torch(cent_obs_batch[self.policy_ids[0]][choose_agent_id]).to(**self.tpdv)

        dones_env_batch = to_torch(dones_env_batch[self.policy_ids[0]]).to(**self.tpdv)
        # individual agent q value sequences: each element is of shape (ep_len, batch_size, 1)
        q_evals_all = []
        q_targets_all = []
        q_attend_regs_all = []

        for p_id in self.policy_ids:
            policy = self.policies[p_id]
            target_policy = self.target_policies[p_id]
            # get data related to the policy id
            pol_obs_batch = to_torch(obs_batch[p_id])
            curr_act_batch = to_torch(act_batch[p_id]).to(**self.tpdv)

            # stack over policy's agents to process them at once
            stacked_act_batch = torch.cat(list(curr_act_batch), dim=-2).to(**self.tpdv)
            stacked_obs_batch = torch.cat(list(pol_obs_batch), dim=-2).to(**self.tpdv)

            if avail_act_batch[p_id] is not None:
                curr_avail_act_batch = to_torch(avail_act_batch[p_id])
                stacked_avail_act_batch = torch.cat(list(curr_avail_act_batch), dim=-2).to(**self.tpdv)
            else:
                stacked_avail_act_batch = None

            # [num_agents, episode_length, episodes, dim]
            step = pol_obs_batch.shape[1]
            batch_size = pol_obs_batch.shape[2]
            total_batch_size = batch_size * len(self.policy_agents[p_id])

            sum_act_dim = int(sum(policy.act_dim)) if policy.multidiscrete else policy.act_dim

            pol_prev_act_buffer_seq = torch.cat((torch.zeros(1, total_batch_size, sum_act_dim).to(**self.tpdv),
                                                 stacked_act_batch))

            # sequence of q values for all possible actions
            individual_q_evals, _ = policy.get_q_values(stacked_obs_batch, pol_prev_act_buffer_seq,policy.init_hidden(-1, total_batch_size))
            individual_q_targets, _ = target_policy.get_q_values(stacked_obs_batch, 
                                                                              pol_prev_act_buffer_seq,
                                                                              target_policy.init_hidden(-1, total_batch_size))
            
            mac_out = individual_q_evals.reshape(step,self.num_agents,batch_size,-1).transpose(1,2)
            target_mac_out = individual_q_targets.reshape(step,self.num_agents,batch_size,-1).transpose(1,2)
            avail_actions = stacked_avail_act_batch.reshape(step,self.num_agents,batch_size,-1).transpose(1,2)
            actions_onehot  = stacked_act_batch.reshape(step-1,self.num_agents,batch_size,-1).transpose(1,2)
            _, max_action_qvals = policy.actions_from_q(mac_out[:-1], available_actions=avail_actions[:-1])
            cur_max_actions, _ = policy.actions_from_q(mac_out[1:], available_actions=avail_actions[1:])      
            cur_max_actions_onehot = torch.tensor(cur_max_actions,dtype=torch.float32).to(**self.tpdv)
            _, target_max_qvals = target_policy.actions_from_q(target_mac_out[1:], available_actions=avail_actions[1:])
            chosen_action_qvals = policy.q_values_from_actions(mac_out[:-1], actions_onehot)
            target_chosen_qvals = target_policy.q_values_from_actions(target_mac_out[1:], cur_max_actions_onehot)
            
            if self.qatten:
                ans_chosen, q_attend_regs, head_entropies = \
                    self.mixer(chosen_action_qvals, cent_obs_batch[:-1], is_v=True)
                ans_adv, _, _ = self.mixer(chosen_action_qvals, cent_obs_batch[:-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                qvals = ans_chosen + ans_adv
                target_chosen, _, _ = self.target_mixer(target_chosen_qvals, cent_obs_batch[1:], is_v=True)
                target_adv, _, _ = self.target_mixer(target_chosen_qvals, cent_obs_batch[1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                target_qvals = target_chosen + target_adv
                q_attend_regs_all.append(q_attend_regs)
            else:
                ans_chosen = self.mixer(chosen_action_qvals, cent_obs_batch[:-1], is_v=True)
                ans_adv = self.mixer(chosen_action_qvals, cent_obs_batch[:-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                qvals = ans_chosen + ans_adv
                target_chosen = self.target_mixer(target_chosen_qvals, cent_obs_batch[1:], is_v=True)
                target_adv = self.target_mixer(target_chosen_qvals, cent_obs_batch[1:],actions=cur_max_actions_onehot,max_q_i=target_max_qvals, is_v=False)
                target_qvals = target_chosen + target_adv
                
            q_evals_all.append(qvals)
            q_targets_all.append(target_qvals)
            
            # don't need the first Q values for next step
        # combine agent q value sequences to feed into mixer networks
        #import pdb;pdb.set_trace()
        q_evals_batch = torch.cat(q_evals_all, dim=-1)
        q_targets_batch = torch.cat(q_targets_all, dim=-1)
        
        # agents share reward
        rewards = to_torch(rew_batch[self.policy_ids[0]][0]).to(**self.tpdv)
        # form bad transition mask
        bad_transitions_mask = torch.cat((torch.zeros(1, batch_size, 1).to(**self.tpdv), dones_env_batch[:self.episode_length - 1, :, :]))
        #import pdb;pdb.set_trace()
        # L_td
        y_dqn = rewards + (1 - dones_env_batch) * self.args.gamma * q_targets_batch
        error = (q_evals_batch - y_dqn.detach()) * (1 - bad_transitions_mask)
        #import pdb;pdb.set_trace()
        if self.qatten:
            q_attend_regs_batch = q_attend_regs_all[0]
            loss = mse_loss(error).sum() / (1 - bad_transitions_mask).sum() + q_attend_regs_batch
        else:
            loss = mse_loss(error).sum() / (1 - bad_transitions_mask).sum()
        #import pdb;pdb.set_trace()
        # backward pass and gradient step
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.args.max_grad_norm)
        self.optimizer.step()
        # log
        train_info = {}
        train_info['loss'] = loss.cpu().item()
        train_info['grad_norm'] = grad_norm.cpu().item()
        train_info['Q_tot'] = (q_evals_batch * (1 - bad_transitions_mask)).mean().cpu().item()

        return train_info, None, None


    def hard_target_updates(self):
        """Hard update the target networks."""
        print("hard update targets")
        for policy_id in self.policy_ids:
            self.target_policies[policy_id].load_state(self.policies[policy_id])
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_target_updates(self):
        """Soft update the target networks."""
        for policy_id in self.policy_ids:
            soft_update(self.target_policies[policy_id], self.policies[policy_id], self.tau)
        if self.mixer is not None:
            soft_update(self.target_mixer, self.mixer, self.tau)

    def prep_training(self):
        """See parent class."""
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.train()
            self.target_policies[p_id].q_network.train()
        self.mixer.train()
        self.target_mixer.train()

    def prep_rollout(self):
        """See parent class."""
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.eval()
            self.target_policies[p_id].q_network.eval()
        self.mixer.eval()
        self.target_mixer.eval()
