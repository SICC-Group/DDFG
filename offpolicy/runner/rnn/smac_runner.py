import numpy as np
import torch
import time
from offpolicy.runner.rnn.base_runner import RecRunner
import pandas as pd
class SMACRunner(RecRunner):
    def __init__(self, config):
        """Runner class for the StarcraftII environment (SMAC). See parent class for more information."""
        super(SMACRunner, self).__init__(config)
        # fill replay buffer with random actions
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.start = time.time()
        self.warmup(num_warmup_episodes)
        end = time.time()
        print("\n Env {} Map {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}. \n"
              .format(self.env_name,
                      self.args.map_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / (end - self.start))))
        self.log_clear()
    
    def eval(self):
        """Collect episodes to evaluate the policy."""
        self.trainer.prep_rollout()

        eval_infos = {}
        eval_infos['win_rate'] = []
        eval_infos['average_episode_rewards'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter(explore=False, training_episode=False, warmup=False)
            
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")
    
    @torch.no_grad()
    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env

        obs, share_obs, avail_acts = env.reset()

        self.act_dim = policy.output_dim

        last_acts_batch = np.zeros((self.num_envs * len(self.policy_agents[p_id]), self.act_dim), dtype=np.float32)
        rnn_states_batch = np.zeros((self.num_envs * len(self.policy_agents[p_id]), self.hidden_size), dtype=np.float32)
        # init
        episode_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, self.act_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id : np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id : np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.act_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_adj = {p_id : np.zeros((self.episode_length+1,self.num_envs, self.num_agents, self.num_factor), dtype=np.int64) for p_id in self.policy_ids}
        episode_prob_adj = {p_id : np.zeros((self.episode_length+1,self.num_envs, self.num_agents, self.num_factor), dtype=np.float32) for p_id in self.policy_ids}
        episode_qtot = {p_id : np.zeros((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_f_v = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_factor+self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_f_q = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_factor+self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_rnn_states = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.hidden_size), dtype=np.float32) for p_id in self.policy_ids}
        
        dones = np.zeros((self.num_envs, self.num_agents, 1), dtype=np.bool_)
        t = 0
        while t < self.episode_length:
            obs_batch = np.concatenate(obs)
            states_batch = np.concatenate(share_obs)
            avail_acts_batch = np.concatenate(avail_acts)
            # get actions for all agents to step the env
            if self.algorithm_name in self.adj_correlation:
                _, rnn_states_batch ,_ = policy.get_hidden_states(obs_batch,last_acts_batch,rnn_states_batch)
                if self.use_dyn_graph:   
                    prob_adj, adj, _ =  self.adj_network.sample(rnn_states_batch.unsqueeze(0),states_batch,self.use_adj_init,explore,self.total_env_steps)
                    adj_all = torch.cat([adj.cpu().detach(),torch.eye(self.num_agents,dtype=torch.int64).unsqueeze(0)],dim=2)
                else:
                    prob_adj = torch.zeros((1, self.num_factor),dtype=torch.float32)
                    adj = self.adj
                    adj_all = adj
                if warmup:
                    acts_batch = policy.get_random_actions(obs_batch,avail_acts_batch)
                else:
                    if self.algorithm_name == 'casec':
                        acts_batch,  qtot, _, f_q = policy.get_actions(rnn_states_batch.unsqueeze(0) ,
                                                                    torch.tensor(avail_acts_batch),
                                                                    t_env=self.total_env_steps,
                                                                    explore=explore)
                    else:
                        acts_batch,  qtot, _, f_q = policy.get_actions(rnn_states_batch.unsqueeze(0),
                                                                    torch.tensor(avail_acts_batch).to(self.device),
                                                                    t_env=self.total_env_steps,
                                                                    explore=explore,
                                                                    adj_input = adj_all.to(self.device),
                                                                    no_sequence = False,
                                                                    dones = torch.tensor(dones).to(self.device))
                    if self.use_vfunction:
                        f_v = policy.get_v_values(rnn_states_batch.unsqueeze(0) ,adj_all.to(self.device),no_sequence = False,
                                                                    dones = torch.tensor(dones).to(self.device))
                  
            else:
                if warmup:
                    # completely random actions in pre-training warmup phase
                    acts_batch = policy.get_random_actions(obs_batch, avail_acts_batch)
                    # get new rnn hidden state
                    _, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                last_acts_batch,
                                                                rnn_states_batch,
                                                                avail_acts_batch)

                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                        last_acts_batch,
                                                                        rnn_states_batch,
                                                                        avail_acts_batch,
                                                                        t_env=self.total_env_steps,
                                                                        explore=explore)
                prob_adj = torch.zeros((self.num_agents, self.num_factor),dtype=torch.float32)
                adj = torch.zeros((self.num_agents, self.num_factor),dtype=torch.int64)    
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
          
            last_acts_batch = acts_batch
            env_acts = np.split(acts_batch, self.num_envs)
            # env step and store the relevant episode information
            next_obs, next_share_obs, rewards, dones, infos, next_avail_acts = env.step(env_acts)
            if training_episode or warmup:
                self.total_env_steps += self.num_envs

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1

            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = env_acts
            episode_rewards[p_id][t] = rewards
            episode_rnn_states[p_id][t] = rnn_states_batch
            
            # here dones store agent done flag of the next step
            if self.algorithm_name in self.adj_correlation:
                env_adj = adj.cpu().detach().numpy()[0] if self.algorithm_name == "rddfg_cent_rw" else adj.cpu().detach().numpy()
                env_prob_adj = prob_adj.cpu().detach().numpy()[0] if self.algorithm_name == "rddfg_cent_rw" else prob_adj.cpu().detach().numpy()
                episode_adj[p_id][t] = env_adj
                episode_prob_adj[p_id][t] = env_prob_adj
                if self.algorithm_name == "rddfg_cent_rw" and not warmup:
                    episode_qtot[p_id][t] = qtot.cpu().detach().numpy() 
                    episode_f_q[p_id][t] = f_q.cpu().detach().numpy()
                    if self.use_vfunction:
                        episode_f_v[p_id][t] = f_v.cpu().detach().numpy()     
            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env
            episode_avail_acts[p_id][t] = avail_acts
            t += 1

            obs = next_obs
            share_obs = next_share_obs
            avail_acts = next_avail_acts

            assert self.num_envs == 1, ("only one env is support here.")
            if terminate_episodes:
                for i in range(self.num_envs):
                    if 'won' in infos[i][0].keys():
                        # take one agent
                        env_info['win_rate'] = 1 if infos[i][0]['won'] else 0
                break

        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = share_obs
        episode_avail_acts[p_id][t] = avail_acts
        if self.algorithm_name in self.adj_correlation:
            _, rnn_states_batch ,_ = policy.get_hidden_states(np.concatenate(obs),last_acts_batch,rnn_states_batch) 
            if self.use_dyn_graph:   
                prob_adj, adj,_ =  self.adj_network.sample(rnn_states_batch.unsqueeze(0),np.concatenate(share_obs),self.use_adj_init,explore,self.total_env_steps)
            else:
                prob_adj = torch.zeros((self.num_agents, self.num_factor),dtype=torch.float32)
                adj = self.adj
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            env_adj = adj.cpu().detach().numpy()[0] if self.algorithm_name == "rddfg_cent_rw" else adj.cpu().detach().numpy()
            env_prob_adj = prob_adj.cpu().detach().numpy()[0] if self.algorithm_name == "rddfg_cent_rw" else prob_adj.cpu().detach().numpy()
            episode_adj[p_id][t] = env_adj
            episode_prob_adj[p_id][t] = env_prob_adj
            episode_rnn_states[p_id][t] = rnn_states_batch
        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer
            ind = self.buffer.insert(self.num_envs,
                               episode_obs,
                               episode_share_obs,
                               episode_acts,
                               episode_rewards,
                               episode_dones,
                               episode_dones_env,
                               episode_avail_acts,
                               episode_adj,
                               episode_prob_adj)
            if self.algorithm_name == "rddfg_cent_rw" and not warmup:
                rewards = self.buffer.norm_reward(ind)
                idx = self.adj_buffer.insert(self.num_envs,
                               episode_obs,
                               episode_share_obs,
                               episode_acts,
                               rewards,
                               episode_dones,
                               episode_dones_env,
                               episode_avail_acts,
                               episode_adj,
                               episode_prob_adj,
                               episode_qtot,
                               episode_f_v,
                               episode_f_q,
                               episode_rnn_states)
                self.adj_buffer.compute_advantage(idx)

        env_info['average_episode_rewards'] = np.sum(episode_rewards[p_id][:, 0, 0, 0])

        return env_info

    def log(self):
        """See parent class."""
        end = time.time()
        print("\n Env {} Map {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}. \n"
              .format(self.env_name,
                      self.args.map_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / (end - self.start))))
        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}
        self.env_infos['win_rate'] = []
        self.env_infos['average_episode_rewards'] = []
