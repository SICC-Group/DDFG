import os
import numpy as np
import wandb
import torch
from tensorboardX import SummaryWriter
import pandas as pd
import time
from offpolicy.utils.rec_buffer import RecReplayBuffer, PrioritizedRecReplayBuffer
from offpolicy.utils.adj_buffer import AdjBuffer
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from offpolicy.utils.util import DecayThenFlatSchedule

class RecRunner(object):
    """Base class for training recurrent policies."""

    def __init__(self, config):
        """
        Base class for training recurrent policies.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        self.args = config["args"]
        self.device = config["device"]
        self.adj = config["adj"]
        self.q_learning = ["qplex","qtran","wqmix","qmix","vdn","rddfg_cent_rw","rmfg_cent","sopcg","casec"]
        self.adj_correlation = ["rddfg_cent_rw","rmfg_cent","sopcg","casec"]

        self.share_policy = self.args.share_policy
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps
        self.use_wandb = self.args.use_wandb
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_popart = self.args.use_popart
        self.use_per = self.args.use_per
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.adj_buffer_size = self.args.adj_buffer_size
        self.hidden_size = self.args.hidden_size
        self.highest_orders = self.args.highest_orders
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval_episode = self.args.hard_update_interval_episode
        self.popart_update_interval_step = self.args.popart_update_interval_step
        self.actor_train_interval_step = self.args.actor_train_interval_step
        self.train_interval_episode = self.args.train_interval_episode
        self.train_adj_episode = self.args.train_adj_episode
        self.drop_temperature_episode = self.args.drop_temperature_episode
        self.train_interval = self.args.train_interval
        self.use_dyn_graph = self.args.use_dyn_graph
        self.equal_vdn = self.args.equal_vdn
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval
        self.gae_lambda = self.args.gae_lambda
        self.gamma = self.args.gamma
        self.use_linear_lr_decay = self.args.use_linear_lr_decay
        self.independent_p_q = self.args.independent_p_q
        self.pair_rnn_hidden_dim = self.args.pair_rnn_hidden_dim
        self.epsilon_anneal_time = self.args.epsilon_anneal_time
        self.total_env_steps = 0  # total environment interactions collected during training
        self.num_episodes_collected = 0  # total episodes collected during training
        self.total_train_steps = 0  # number of gradient updates performed
        self.last_train_episode = 0  # last episode after which a gradient update was performed
        self.last_train_adj_episode = 0
        self.last_drop_t_episode = 0
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0 # last timestep after which information was logged
        self.last_hard_update_episode = 0 # last episode after which target policy was updated to equal live policy
        self.use_vfunction = self.args.use_vfunction
        self.use_save = self.args.use_save
        self.pretrain_adj = self.args.pretrain_adj
        self.num_mini_batch = self.args.num_mini_batch
        self.adj_begin_step = self.args.adj_begin_step
        self.use_adj_init = self.args.use_adj_init
        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        if config.__contains__("buffer_length"):
            self.episode_length = config["buffer_length"]
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = config["buffer_length"]
            else:
                self.data_chunk_length = self.args.data_chunk_length
        else:
            self.episode_length = self.args.episode_length
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = self.args.episode_length
            else:
                self.data_chunk_length = self.args.data_chunk_length
        #import pdb;pdb.set_trace()
        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
        #import pdb;pdb.set_trace()
        self.policy_mapping_fn = config["policy_mapping_fn"]

        self.num_agents = config["num_agents"]
        self.num_factor = self.args.num_factor
        self.agent_ids = [i for i in range(self.num_agents)]

        self.env = config["env"]
        self.eval_env = config["eval_env"]
        # no parallel envs
        self.num_envs = 1
        self.action_repr_updating = True
        # dir
        #import pdb;pdb.set_trace()
        self.model_dir = self.args.model_dir
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            #import pdb;pdb.set_trace()
            self.run_dir = config["run_dir"]
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # initialize all the policies and organize the agents corresponding to each policy
        if self.algorithm_name == "rmatd3":
            from offpolicy.algorithms.r_matd3.algorithm.rMATD3Policy import R_MATD3Policy as Policy
            from offpolicy.algorithms.r_matd3.r_matd3 import R_MATD3 as TrainAlgo
        elif self.algorithm_name == "rmaddpg":
            assert self.actor_train_interval_step == 1, (
                "rmaddpg only supports actor_train_interval_step=1.")
            from offpolicy.algorithms.r_maddpg.algorithm.rMADDPGPolicy import R_MADDPGPolicy as Policy
            from offpolicy.algorithms.r_maddpg.r_maddpg import R_MADDPG as TrainAlgo
        elif self.algorithm_name == "rmasac":
            assert self.actor_train_interval_step == 1, (
                "rmasac only support actor_train_interval_step=1.")
            from offpolicy.algorithms.r_masac.algorithm.rMASACPolicy import R_MASACPolicy as Policy
            from offpolicy.algorithms.r_masac.r_masac import R_MASAC as TrainAlgo
        elif self.algorithm_name == "qtran":
            from offpolicy.algorithms.qtran.algorithm.QTranPolicy import QTranPolicy as Policy
            from offpolicy.algorithms.qtran.qtran import QTran as TrainAlgo
        elif self.algorithm_name == "qplex":
            from offpolicy.algorithms.qplex.algorithm.QPlexPolicy import QPlexPolicy as Policy
            from offpolicy.algorithms.qplex.qplex import QPlex as TrainAlgo
        elif self.algorithm_name == "wqmix":
            from offpolicy.algorithms.wqmix.algorithm.QMixPolicy import QMixPolicy as Policy
            from offpolicy.algorithms.wqmix.algorithm.QMixPolicy import QMixPolicy as Central_Policy
            from offpolicy.algorithms.wqmix.qmix import WQMix as TrainAlgo
        elif self.algorithm_name == "qmix":
            from offpolicy.algorithms.qmix.algorithm.QMixPolicy import QMixPolicy as Policy
            from offpolicy.algorithms.qmix.qmix import QMix as TrainAlgo
        elif self.algorithm_name == "vdn":
            from offpolicy.algorithms.vdn.algorithm.VDNPolicy import VDNPolicy as Policy
            from offpolicy.algorithms.vdn.vdn import VDN as TrainAlgo
        elif self.algorithm_name == "rmfg_cent":
            from offpolicy.algorithms.r_mfg_cent.algorithm.adj_generator import Adj_Generator as Adj
            from offpolicy.algorithms.r_mfg_cent.algorithm.rMFGPolicy_parallel import R_MFGPolicy_Parallel as Policy
            from offpolicy.algorithms.r_mfg_cent.r_mfg import R_MFG as TrainAlgo
        elif self.algorithm_name == "rddfg_cent_rw":
            from offpolicy.algorithms.r_ddfg_cent_rw.algorithm.adj_generator_new import Adj_Generator as Adj
            from offpolicy.algorithms.r_ddfg_cent_rw.algorithm.rDDFGPolicy import R_DDFGPolicy as Policy
            from offpolicy.algorithms.r_ddfg_cent_rw.r_ddfg import R_DDFG as TrainAlgo
        elif self.algorithm_name == "sopcg":
            from offpolicy.algorithms.sopcg.algorithm.adj_generator import Adj_Generator as Adj
            from offpolicy.algorithms.sopcg.algorithm.rSOPCGPolicy_parallel import R_SOPCGPolicy_Parallel as Policy
            from offpolicy.algorithms.sopcg.r_sopcg import R_SOPCG as TrainAlgo
        elif self.algorithm_name == "casec":
            from offpolicy.algorithms.casec.algorithm.adj_generator import Adj_Generator as Adj
            from offpolicy.algorithms.casec.algorithm.rCASECPolicy import R_CASECPolicy as Policy
            from offpolicy.algorithms.casec.r_casec import R_CASEC as TrainAlgo
        else:
            raise NotImplementedError
        
        self.collecter = self.collect_rollout
        if self.algorithm_name in self.adj_correlation:
            self.saver = self.save_q_mdfg_cent
            self.restorer = self.restore_mdfg_cent
        elif self.algorithm_name in self.q_learning:
            self.saver = self.save_q
            self.restorer = self.restore_q
        else:
            self.saver = self.save
            self.restorer = self.restore
        self.train = self.batch_train_q if self.algorithm_name in self.q_learning else self.batch_train
        self.train_adj = self.batch_train_adj

        self.policies = {p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}
        
        
        # initialize trainer class for updating policies
        if self.algorithm_name == "rmdfg":
            self.adj_network = Adj(self.num_agents,self.num_factor,self.args.highest_orders,self.device)
            self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.adj_network, self.policy_mapping_fn,device=self.device, episode_length=self.episode_length)
        elif self.algorithm_name in self.adj_correlation:
            self.obs_dim = get_dim_from_space(self.policy_info[self.policy_ids[0]]["obs_space"])
            self.state_dim = get_dim_from_space(self.policy_info[self.policy_ids[0]]["share_obs_space"])
            self.adj_network = Adj(self.args,self.obs_dim,self.state_dim,self.device)
            self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.adj_network, self.policy_mapping_fn,device=self.device, episode_length=self.episode_length)
        elif self.algorithm_name == "wqmix":
            self.central_policies = {p_id: Central_Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}
            self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.central_policies, self.policy_mapping_fn,
                                 device=self.device, episode_length=self.episode_length)
        else:
            self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.policy_mapping_fn,
                                 device=self.device, episode_length=self.episode_length)
        
        if self.model_dir is not None:
            if self.pretrain_adj:
                self.load_adj()
            else:
                self.restorer()
        # map policy id to agent ids controlled by that policy
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}

        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}

        num_train_episodes = (self.num_env_steps / self.episode_length) / (self.train_interval_episode)
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_episodes, decay="linear")

        if self.use_per:
            self.buffer = PrioritizedRecReplayBuffer(self.per_alpha,
                                                     self.policy_info,
                                                     self.policy_agents,
                                                     self.buffer_size,
                                                     self.episode_length,
                                                     self.use_same_share_obs,
                                                     self.use_avail_acts,
                                                     self.use_reward_normalization)
        else:
            self.buffer = RecReplayBuffer(self.policy_info,
                                          self.policy_agents,
                                          self.num_factor,
                                          self.buffer_size,
                                          self.episode_length,
                                          self.use_same_share_obs,
                                          self.use_avail_acts,
                                          self.use_reward_normalization)
            
        self.adj_buffer = AdjBuffer(self.policy_info,
                                          self.policy_agents,
                                          self.num_factor,
                                          self.adj_buffer_size,
                                          self.episode_length,
                                          self.use_same_share_obs,
                                          self.use_avail_acts,
                                          self.use_reward_normalization,
                                          self.gamma,
                                          self.gae_lambda,
                                          self.hidden_size)
    def run(self):
        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""
        # collect data
        self.trainer.prep_rollout()
        env_info = self.collecter(explore=True, training_episode=True, warmup=False)
        for k, v in env_info.items():
            self.env_infos[k].append(v)
            
        # train
        if ((self.num_episodes_collected - self.last_train_episode - self.batch_size) / self.train_interval_episode) >= 1:
            self.train()
            self.total_train_steps += 1
            self.last_train_episode = self.num_episodes_collected - self.batch_size
        if self.use_dyn_graph and self.total_env_steps >= self.adj_begin_step and ((self.num_episodes_collected - self.last_train_adj_episode - self.batch_size) / self.train_adj_episode) >= 1 :    
            if self.use_linear_lr_decay:
                self.trainer.lr_decay(self.total_env_steps-self.adj_begin_step,self.num_env_steps-self.adj_begin_step)
            if not self.pretrain_adj:
                self.train_adj()
                self.log_train_adj(self.train_adj_infos)   
            self.last_train_adj_episode = self.num_episodes_collected - self.batch_size
        # save
        if self.use_save and (self.total_env_steps - self.last_save_T) / self.save_interval >= 1:
            self.saver()
            self.last_save_T = self.total_env_steps
        # log
        if self.total_env_steps > 8500 and ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
            
            self.log()
            self.last_log_T = self.total_env_steps

        # eval
        if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1:
            self.eval()
            self.last_eval_T = self.total_env_steps

        return self.total_env_steps
    
    def warmup(self, num_warmup_episodes):
        """
        Fill replay buffer with enough episodes to begin training.

        :param: num_warmup_episodes (int): number of warmup episodes to collect.
        """
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("warm up...")
        for _ in range((num_warmup_episodes // self.num_envs)):
            #env_info = self.collecter(explore=True, training_episode=False, warmup=False)
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info['average_episode_rewards'])
        warmup_reward = np.mean(warmup_rewards)
        print("warmup average episode rewards: {}".format(warmup_reward))

    def batch_train(self):
        """Do a gradient update for all policies."""
        self.trainer.prep_training()

        # gradient updates
        self.train_infos = []
        update_actor = False
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)
            
            update_method = self.trainer.shared_train_policy_on_batch if self.use_same_share_obs else self.trainer.cent_train_policy_on_batch
            
            train_info, new_priorities, idxes = update_method(p_id, sample)
            update_actor = True#train_info['update_actor']

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if ((self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def batch_train_q(self):
        """Do a q-learning update to policy (used for QMix and VDN)."""
        self.trainer.prep_training()
        # gradient updates
        self.train_infos = []
        
        '''for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)
            if self.algorithm_name == 'casec':
                train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample,self.action_repr_updating)
            else:
                train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample)
            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)'''

        if self.algorithm_name == 'casec':
            sample = self.buffer.sample(self.batch_size)
            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample,self.action_repr_updating)
            self.train_infos.append(train_info)
        else:  
            for _ in range(4):
                for p_id in self.policy_ids:
                    if self.use_per:
                        beta = self.beta_anneal.eval(self.total_train_steps)
                        sample = self.buffer.sample(self.batch_size, beta, p_id)
                    else:
                        sample = self.buffer.sample(self.batch_size//4)

                    train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample)

                    if self.use_per:
                        self.buffer.update_priorities(idxes, new_priorities, p_id)

                    self.train_infos.append(train_info)
                
        if self.algorithm_name == 'casec' and self.action_repr_updating:
            self.trainer.update_action_repr()
            self.action_repr_updating = False
            self.trainer.hard_target_updates()
            self.last_hard_update_episode = self.num_episodes_collected
        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected
                
    def batch_train_adj(self):
        """Do a q-learning update to policy (used for QMix and VDN)."""
        self.trainer.prep_training()
        # gradient updates
        self.train_adj_infos = {}
        self.train_adj_infos['advantage'] = []
        self.train_adj_infos['clamp_ratio'] = []
        self.train_adj_infos['rl_loss'] = []
        self.train_adj_infos['entropy_loss'] = []
        self.train_adj_infos['grad_norm'] = []
        data_chunk_length = 10
        
        for _ in range(10):
            for p_id in self.policy_info.keys():
                data_generator = self.adj_buffer.policy_buffers[p_id].sample_inds(data_chunk_length,self.num_mini_batch)
                for sample in data_generator:
                    train_adj_info, new_priorities, idxes = self.trainer.train_adj_on_batch(sample,self.use_adj_init)
                    for k, v in train_adj_info.items():
                        self.train_adj_infos[k].append(v)
            self.use_adj_init = False

        
    def save(self):
        """Save all policies to the path specified by the config."""
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(),
                       critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(),
                       actor_save_path + '/actor.pt')

    def save_q(self):
        """Save all policies to the path specified by the config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + '/q_network.pt')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.algorithm_name == "qtran":
            torch.save(self.trainer.eval_joint_q.state_dict(),
                   self.save_dir + '/eval_joint_q.pt')
            torch.save(self.trainer.v.state_dict(),
                   self.save_dir + '/v.pt')
        elif self.algorithm_name == "wqmix":
            for pid in self.policy_ids:
                policy_Q = self.central_policies[pid].q_network
                p_save_path = self.save_dir + '/' + str(pid)
                if not os.path.exists(p_save_path):
                    os.makedirs(p_save_path)
                torch.save(policy_Q.state_dict(), p_save_path + '/central_q_network.pt')
            torch.save(self.trainer.mixer.state_dict(),
                   self.save_dir + '/mixer.pt')
            torch.save(self.trainer.central_mixer.state_dict(),
                   self.save_dir + '/central_mixer.pt') 
        else:
            torch.save(self.trainer.mixer.state_dict(),
                   self.save_dir + '/mixer.pt')

    def save_q_mdfg_cent(self):
        """Save all policies to the path specified by the config. Used for QMix and VDN."""
        save_path = self.save_dir
        if not os.path.exists(save_path):
                os.makedirs(save_path)

        adj = self.adj_network
        torch.save(adj.state_dict(), save_path + '/adj_network.pt')
        for pid in self.policy_ids:
            p_save_path = save_path + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            rnn_Q = self.policies[pid].rnn_network
            torch.save(rnn_Q.state_dict(), p_save_path + '/rnn_network.pt')
            for num_orders in range(1,self.highest_orders+1):
                policy_Q = self.policies[pid].q_network[num_orders]
                torch.save(policy_Q.state_dict(), p_save_path + '/q_network_{}.pt'.format(num_orders))
                if self.use_vfunction:
                    policy_V = self.policies[pid].v_network[num_orders]
                    torch.save(policy_V.state_dict(), p_save_path + '/v_network_{}.pt'.format(num_orders))

    def restore(self):
        """Load policies policies from pretrained models specified by path in config."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    def restore_q(self):
        """Load policies policies from pretrained models specified by path in config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_q_state_dict = torch.load(path + '/q_network.pt')           
            self.policies[pid].q_network.load_state_dict(policy_q_state_dict)
            
        policy_mixer_state_dict = torch.load(str(self.model_dir) + '/mixer.pt')
        self.trainer.mixer.load_state_dict(policy_mixer_state_dict)

    def load_adj(self):
        """Load policies policies from pretrained models specified by path in config. Used for QMix and VDN."""
        path_adj = str(self.model_dir)
        adj_state_dict = torch.load(path_adj + 'adj_network.pt')   
        self.adj_network.load_state_dict(adj_state_dict)
                    
    def restore_mdfg_cent(self):
        """Load policies policies from pretrained models specified by path in config. Used for QMix and VDN."""
        path_adj = str(self.model_dir)
        adj_state_dict = torch.load(path_adj + 'adj_network.pt')   
        self.adj_network.load_state_dict(adj_state_dict)
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            rnn_state_dict = torch.load(path + '/rnn_network.pt')
            self.policies[pid].rnn_network.load_state_dict(rnn_state_dict)
            for num_orders in range(1,self.highest_orders+1):
                policy_q_state_dict = torch.load(path + '/q_network_{}.pt'.format(num_orders))     
                self.policies[pid].q_network[num_orders].load_state_dict(policy_q_state_dict)
                if self.use_vfunction:
                    policy_v_state_dict = torch.load(path + '/v_network_{}.pt'.format(num_orders))     
                    self.policies[pid].v_network[num_orders].load_state_dict(policy_v_state_dict)

    def log(self):
        """Log relevent training and rollout colleciton information.."""
        raise NotImplementedError

    def log_clear(self):
        """Clear logging variables so they do not contain stale information."""
        raise NotImplementedError

    def log_env(self, env_info, suffix=None):
        """
        Log information related to the environment.
        :param env_info: (dict) contains logging information related to the environment.
        :param suffix: (str) optional string to add to end of keys in env_info when logging. 
        """
        data_env = []
        data_env.append(self.total_env_steps)
        for k, v in env_info.items():
            if len(v) > 0:
                v = np.mean(v)
                data_env.append(v)
                suffix_k = k if suffix is None else suffix + k 
                print(suffix_k + " is " + str(v))
                if self.use_wandb:
                    wandb.log({suffix_k: v}, step=self.total_env_steps)
                else:
                    self.writter.add_scalars(suffix_k, {suffix_k: v}, self.total_env_steps)
        if suffix=="eval_":
            progress_filename = os.path.join(self.run_dir,'progress_eval.csv')
        else:
            progress_filename = os.path.join(self.run_dir,'progress.csv')
        df = pd.DataFrame([data_env])
        df.to_csv(progress_filename,mode='a',header=False,index=False) 
        
    def log_train(self, policy_id, train_info):
        """
        Log information related to training.
        :param policy_id: (str) policy id corresponding to the information contained in train_info.
        :param train_info: (dict) contains logging information related to training.
        """
        data_env = []
        data_env.append(self.total_env_steps)
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k
            data_env.append(v)
            if self.use_wandb:
                wandb.log({policy_k: v}, step=self.total_env_steps)
            else:
                self.writter.add_scalars(policy_k, {policy_k: v}, self.total_env_steps)
                
        progress_filename = os.path.join(self.run_dir,'progress_train.csv')
        df = pd.DataFrame([data_env])
        df.to_csv(progress_filename,mode='a',header=False,index=False) 

    def log_train_adj(self, train_adj_info):
        """
        Log information related to training.
        :param policy_id: (str) policy id corresponding to the information contained in train_info.
        :param train_info: (dict) contains logging information related to training.
        """
        data_env = []
        data_env.append(self.total_env_steps)             
        for k, v in train_adj_info.items():
            if len(v) > 0:
                #import pdb;pdb.set_trace()
                v = np.mean(v)
                data_env.append(v)
            if self.use_wandb:
                wandb.log({k: v}, step=self.total_env_steps)
            else:
                self.writter.add_scalars(k, {k: v}, self.total_env_steps)
        progress_filename = os.path.join(self.run_dir,'progress_train_adj.csv')
        df = pd.DataFrame([data_env])
        df.to_csv(progress_filename,mode='a',header=False,index=False)         
        
    def collect_rollout(self):
        """Collect a rollout and store it in the buffer."""
        raise NotImplementedError