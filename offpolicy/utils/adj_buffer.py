import numpy as np
from offpolicy.utils.util import get_dim_from_space
from offpolicy.utils.segment_tree import SumSegmentTree, MinSegmentTree
import torch

def _cast(x):
    return x.transpose(2, 0, 1, 3)


class AdjBuffer(object):
    def __init__(self, policy_info, policy_agents, num_factor, buffer_size, episode_length, use_same_share_obs, use_avail_acts,use_reward_normalization=False,gamma=0.97,gae_lambda=0.95,hidden_size=64):
        """
        Replay buffer class for training RNN policies. Stores entire episodes rather than single transitions.

        :param policy_info: (dict) maps policy id to a dict containing information about corresponding policy.
        :param policy_agents: (dict) maps policy id to list of agents controled by corresponding policy.
        :param buffer_size: (int) max number of transitions to store in the buffer.
        :param use_same_share_obs: (bool) whether all agents share the same centralized observation.
        :param use_avail_acts: (bool) whether to store what actions are available.
        :param use_reward_normalization: (bool) whether to use reward normalization.
        """
        self.policy_info = policy_info

        self.policy_buffers = {p_id: AdjPolicyBuffer(buffer_size,
                                                     episode_length,
                                                     len(policy_agents[p_id]),
                                                     num_factor,
                                                     self.policy_info[p_id]['obs_space'],
                                                     self.policy_info[p_id]['share_obs_space'],
                                                     self.policy_info[p_id]['act_space'],
                                                     use_same_share_obs,
                                                     use_avail_acts,
                                                     use_reward_normalization,
                                                     gamma,
                                                     gae_lambda,
                                                     hidden_size)
                               for p_id in self.policy_info.keys()}

    def __len__(self):
        return self.policy_buffers['policy_0'].filled_i
      
    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts, adj=None,prob_adj=None,q_tot=None,f_v=None,f_q=None,rnn_states=None):
        """
        Insert a set of episodes into buffer. If the buffer size overflows, old episodes are dropped.

        :param num_insert_episodes: (int) number of episodes to be added to buffer
        :param obs: (dict) maps policy id to numpy array of observations of agents corresponding to that policy
        :param share_obs: (dict) maps policy id to numpy array of centralized observation corresponding to that policy
        :param acts: (dict) maps policy id to numpy array of actions of agents corresponding to that policy
        :param rewards: (dict) maps policy id to numpy array of rewards of agents corresponding to that policy
        :param dones: (dict) maps policy id to numpy array of terminal status of agents corresponding to that policy
        :param dones_env: (dict) maps policy id to numpy array of terminal status of env
        :param valid_transition: (dict) maps policy id to numpy array of whether the corresponding transition is valid of agents corresponding to that policy
        :param avail_acts: (dict) maps policy id to numpy array of available actions of agents corresponding to that policy

        :return: (np.ndarray) indexes in which the new transitions were placed.
        """
        for p_id in self.policy_info.keys():
            idx_range = self.policy_buffers[p_id].insert(num_insert_episodes, np.array(obs[p_id]),
                                                         np.array(share_obs[p_id]), np.array(acts[p_id]),
                                                         np.array(rewards[p_id]), np.array(dones[p_id]),
                                                         np.array(dones_env[p_id]), np.array(avail_acts[p_id]),
                                                         np.array(adj[p_id]),np.array(prob_adj[p_id]),np.array(q_tot[p_id]),
                                                         np.array(f_v[p_id]), np.array(f_q[p_id]), np.array(rnn_states[p_id]))
        return idx_range

    '''def sample(self, batch_size,data_chunk_length,num_mini_batch):
        """
        Sample a set of episodes from buffer, uniformly at random.
        :param batch_size: (int) number of episodes to sample from buffer.

        :return: obs: (dict) maps policy id to sampled observations corresponding to that policy
        :return: share_obs: (dict) maps policy id to sampled observations corresponding to that policy
        :return: acts: (dict) maps policy id to sampled actions corresponding to that policy
        :return: rewards: (dict) maps policy id to sampled rewards corresponding to that policy
        :return: dones: (dict) maps policy id to sampled terminal status of agents corresponding to that policy
        :return: dones_env: (dict) maps policy id to sampled environment terminal status corresponding to that policy
        :return: valid_transition: (dict) maps policy_id to whether each sampled transition is valid or not (invalid if corresponding agent is dead)
        :return: avail_acts: (dict) maps policy_id to available actions corresponding to that policy
        """
        inds = np.random.choice(self.__len__(), batch_size)
        #import pdb;pdb.set_trace()
        obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, prob_adj_batch, advantages_batch, f_advts_batch, rnn_obs_batch = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p_id in self.policy_info.keys():
            obs_batch[p_id], share_obs_batch[p_id], dones_batch[p_id], dones_env_batch[p_id], adj_batch[p_id], prob_adj_batch[p_id], advantages_batch[p_id], f_advts_batch[p_id], rnn_obs_batch[p_id] = self.policy_buffers[p_id].sample_inds(inds,data_chunk_length,num_mini_batch)

        return obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, prob_adj_batch, advantages_batch, f_advts_batch, rnn_obs_batch'''
        
    def compute_advantage(self, idx, value_normalizer=None):

        for p_id in self.policy_info.keys():
            self.policy_buffers[p_id].compute_advantage(idx)
        return idx

class AdjPolicyBuffer(object):
    def __init__(self, buffer_size, episode_length, num_agents, num_factor, obs_space, share_obs_space, act_space,
                 use_same_share_obs, use_avail_acts, use_reward_normalization=False,gamma=0.97,gae_lambda=0.95,hidden_size=64):
        """
        Buffer class containing buffer data corresponding to a single policy.

        :param buffer_size: (int) max number of episodes to store in buffer.
        :param episode_length: (int) max length of an episode.
        :param num_agents: (int) number of agents controlled by the policy.
        :param obs_space: (gym.Space) observation space of the environment.
        :param share_obs_space: (gym.Space) centralized observation space of the environment.
        :param act_space: (gym.Space) action space of the environment.
        :use_same_share_obs: (bool) whether all agents share the same centralized observation.
        :use_avail_acts: (bool) whether to store what actions are available.
        :param use_reward_normalization: (bool) whether to use reward normalization.
        """
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.num_factor = num_factor
        self.use_same_share_obs = use_same_share_obs
        self.use_avail_acts = use_avail_acts
        self.use_reward_normalization = use_reward_normalization
        self.filled_i = 0
        self.current_i = 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.hidden_size = hidden_size
        # obs
        if obs_space.__class__.__name__ == 'Box':
            obs_shape = obs_space.shape
            share_obs_shape = share_obs_space.shape
        elif obs_space.__class__.__name__ == 'list':
            obs_shape = obs_space
            share_obs_shape = share_obs_space
        else:
            raise NotImplementedError

        self.obs = np.zeros((self.episode_length + 1, self.buffer_size,
                             self.num_agents, obs_shape[0]), dtype=np.float32)

        if self.use_same_share_obs:
            self.share_obs = np.zeros((self.episode_length + 1, self.buffer_size, share_obs_shape[0]), dtype=np.float32)
        else:
            self.share_obs = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, share_obs_shape[0]),
                                      dtype=np.float32)

        # action
        act_dim = np.sum(get_dim_from_space(act_space))
        self.acts = np.zeros((self.episode_length, self.buffer_size, self.num_agents, act_dim), dtype=np.float32)
        if self.use_avail_acts:
            self.avail_acts = np.ones((self.episode_length + 1, self.buffer_size, self.num_agents, act_dim),
                                      dtype=np.float32)

        # rewards
        self.rewards = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)

        # default to done being True
        self.dones = np.ones_like(self.rewards, dtype=np.float32)
        self.dones_env = np.ones((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        self.adj = np.zeros((self.episode_length+1,self.buffer_size, self.num_agents, self.num_factor), dtype=np.int64)
        self.prob_adj = np.zeros((self.episode_length+1,self.buffer_size, self.num_agents, self.num_factor), dtype=np.float32)
        self.qtot = np.zeros((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        self.margin_q = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)
        self.f_q = np.zeros((self.episode_length, self.buffer_size, self.num_factor+self.num_agents, 1), dtype=np.float32)
        self.advantage = np.zeros((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        self.f_v = np.zeros((self.episode_length, self.buffer_size, self.num_factor+self.num_agents, 1), dtype=np.float32)
        self.f_advt = np.zeros((self.episode_length, self.buffer_size, self.num_factor, 1), dtype=np.float32)
        self.rnn_obs = np.zeros((self.episode_length + 1, self.buffer_size,self.num_agents, self.hidden_size), dtype=np.float32)


    def __len__(self):
        return self.filled_i      
    def compute_advantage(self, idx, value_normalizer=None):
        
        f_gae = np.zeros((1,self.num_factor, 1), dtype=np.float32)
        for step in reversed(range(0,self.rewards.shape[0])):
            #value_normalizer.denormalize(self.qtot[step,idx]
            f_delta = self.f_q[step,idx,:self.num_factor] -  self.f_v[step,idx,:self.num_factor]
            f_gae = f_delta + self.gamma * self.gae_lambda * (1-self.dones_env[step,idx]) * f_gae
            self.f_advt[step,idx] = f_gae
           
        return idx
    
    
    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts, adj=None, prob_adj=None,qtot=None,f_v=None,f_q=None,rnn_obs=None):
        """
        Insert a set of episodes corresponding to this policy into buffer. If the buffer size overflows, old transitions are dropped.

        :param num_insert_steps: (int) number of transitions to be added to buffer
        :param obs: (np.ndarray) observations of agents corresponding to this policy.
        :param share_obs: (np.ndarray) centralized observations of agents corresponding to this policy.
        :param acts: (np.ndarray) actions of agents corresponding to this policy.
        :param rewards: (np.ndarray) rewards of agents corresponding to this policy.
        :param dones: (np.ndarray) terminal status of agents corresponding to this policy.
        :param dones_env: (np.ndarray) environment terminal status.
        :param valid_transition: (np.ndarray) whether each transition is valid or not (invalid if agent was dead during transition)
        :param avail_acts: (np.ndarray) available actions of agents corresponding to this policy.

        :return: (np.ndarray) indexes of the buffer the new transitions were placed in.
        """

        # obs: [step, episode, agent, dim]

        episode_length = acts.shape[0]
        assert episode_length == self.episode_length, ("different dimension!")

        if self.current_i + num_insert_episodes <= self.buffer_size:
            idx_range = np.arange(self.current_i, self.current_i + num_insert_episodes)
        else:
            num_left_episodes = self.current_i + num_insert_episodes - self.buffer_size
            idx_range = np.concatenate((np.arange(self.current_i, self.buffer_size), np.arange(num_left_episodes)))

        if self.use_same_share_obs:
            # remove agent dimension since all agents share centralized observation
            share_obs = share_obs[:, :, 0]
        
        self.obs[:, idx_range] = obs.copy()
        self.share_obs[:, idx_range] = share_obs.copy()
        self.acts[:, idx_range] = acts.copy()
        self.rewards[:, idx_range] = rewards.copy()
        self.dones[:, idx_range] = dones.copy()
        self.dones_env[:, idx_range] = dones_env.copy()
        self.adj[:,idx_range] =adj.copy()
        self.prob_adj[:,idx_range] =prob_adj.copy()
        self.qtot[:,idx_range] =qtot.copy()
        self.f_v[:,idx_range] = f_v.copy()
        self.f_q[:,idx_range] = f_q.copy()
        self.rnn_obs[:,idx_range] =rnn_obs.copy()
        

        if self.use_avail_acts:
            self.avail_acts[:, idx_range] = avail_acts.copy()

        self.current_i = idx_range[-1] + 1
        self.filled_i = min(self.filled_i + len(idx_range), self.buffer_size)

        return idx_range

    def sample_inds(self, data_chunk_length, num_mini_batch):
        """
        Sample a set of transitions from buffer from the specified indices.
        :param sample_inds: (np.ndarray) indices of samples to return from buffer.

        :return: obs: (np.ndarray) sampled observations corresponding to that policy
        :return: share_obs: (np.ndarray) sampled observations corresponding to that policy
        :return: acts: (np.ndarray) sampled actions corresponding to that policy
        :return: rewards: (np.ndarray) sampled rewards corresponding to that policy
        :return: dones: (np.ndarray) sampled terminal status of agents corresponding to that policy
        :return: dones_env: (np.ndarray) sampled environment terminal status corresponding to that policy
        :return: valid_transition: (np.ndarray) whether each sampled transition in episodes are valid or not (invalid if corresponding agent is dead)
        :return: avail_acts: (np.ndarray) sampled available actions corresponding to that policy
        """
        #import pdb;pdb.set_trace()
        batch_size = self.episode_length * self.buffer_size
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        
        obs = self.obs[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
        dones = np.concatenate((np.zeros((1, self.buffer_size, self.num_agents,1), dtype=np.float32),self.dones[:-1]))
        dones_env = np.concatenate((np.zeros((1, self.buffer_size,1), dtype=np.float32),self.dones_env[:-1]))
        adj = self.adj[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
        prob_adj = self.prob_adj[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
        advantage = self.advantage
        
        rnn_obs = self.rnn_obs[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
        advantage_copy = advantage.copy()
        advantage_copy[dones_env == 1.0] = np.nan
        mean_advantage = np.nanmean(advantage_copy[1:])
        std_advantage = np.nanstd(advantage_copy[1:])
        advantage[1:] = (advantage[1:] - mean_advantage) / (std_advantage + 1e-10)
        advantages = advantage.transpose(1,0,2).reshape(batch_size,-1)
        
        f_advt = self.f_advt
        f_advt_copy = f_advt.copy()
        f_advt_copy[np.tile(dones_env,self.num_factor) == 1.0] = np.nan
        mean_advt_f = np.nanmean(f_advt_copy)
        std_advt_f = np.nanstd(f_advt_copy)
        f_advt = (f_advt - mean_advt_f) / (std_advt_f + 1e-5)
        f_advts = f_advt.transpose(1,0,2,3).reshape(batch_size,self.num_factor,-1)
        dones = dones.transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
        dones_env = dones_env.transpose(1,0,2).reshape(batch_size,-1)
        if self.use_same_share_obs:
            share_obs = self.share_obs[:-1].transpose(1,0,2).reshape(batch_size,-1)
        else:
            share_obs = self.share_obs[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)

        for indices in sampler:
            obs_batch = []
            share_obs_batch = []
            dones_batch = []
            dones_env_batch = []
            adj_batch = []
            prob_adj_batch = []
            advantages_batch = []
            f_advts_batch = []
            rnn_obs_batch = []

            for i in indices:
                ind = i * data_chunk_length
                obs_batch.append(obs[ind:ind+data_chunk_length])
                share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                dones_batch.append(dones[ind:ind+data_chunk_length])
                dones_env_batch.append(dones_env[ind:ind+data_chunk_length])
                adj_batch.append(adj[ind:ind+data_chunk_length])
                prob_adj_batch.append(prob_adj[ind:ind+data_chunk_length])
                advantages_batch.append(advantages[ind:ind+data_chunk_length])
                f_advts_batch.append(f_advts[ind:ind+data_chunk_length])
                rnn_obs_batch.append(rnn_obs[ind:ind+data_chunk_length])
            obs_batch = np.stack(obs_batch,axis=0)
            share_obs_batch = np.stack(share_obs_batch,axis=0)
            dones_batch = np.stack(dones_batch,axis=0)
            dones_env_batch = np.stack(dones_env_batch,axis=0)
            adj_batch = np.stack(adj_batch,axis=0)
            prob_adj_batch = np.stack(prob_adj_batch,axis=0)
            advantages_batch = np.stack(advantages_batch,axis=0)
            f_advts_batch = np.stack(f_advts_batch,axis=0)
            rnn_obs_batch = np.stack(rnn_obs_batch,axis=0)
            
            yield obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, prob_adj_batch, advantages_batch, f_advts_batch, rnn_obs_batch
        
 



