from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import numpy as np
import random
from gym.spaces import Discrete

class DisperseEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(self,kwargs):
        # Map arguments
        self.args = kwargs
        self._seed = self.args.seed
        np.random.seed(self._seed)
        self.num_agents = self.args.num_agents
        self.n_agents = self.args.num_agents
        # Observations and state
        self.n_actions = self.args.n_hospitals
        self.initial_need = [0, 0, 0, 0]

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.episode_limit = self.args.episode_length
        self.needs = self.initial_need
        self.actions = np.random.randint(0, self.n_actions, self.n_agents)
        self._match = 0
        self.unit_dim = 2
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.num_agents):
            self.action_space.append(Discrete(self.n_actions))
            self.observation_space.append(self.get_obs_size())
            self.share_observation_space.append(self.get_state_size())

    def _split_x(self, x, n):
        result = np.zeros(n)
        p = np.random.randint(low=0, high=n)
        low = x // 2
        result[p] = np.random.randint(low=low, high=x+1)
        return result

    def step(self, actions):
        """Returns reward, terminated, info."""
        # print(self.needs)
        # print(actions)
        self._total_steps += 1
        self._episode_steps += 1
        
        info = [{} for i in range(self.num_agents)]
        actions = np.where(actions==1)[1]
        reward = np.zeros((1,self.num_agents,1), dtype=np.float32)
        terminated = np.zeros((self.num_agents, 1), dtype=bool)
        #info['battle_won'] = False
        # actions_numpy = actions.detach().cpu().numpy()

        delta = []
        for action_i in range(self.n_actions):
            supply = float((actions == action_i).sum())
            need = float(self.needs[action_i])

            if supply >= need:
                self._match += 1

            delta.append(min(supply - need, 0))
        reward += float(np.array(delta).sum()) / self.n_agents

        # print('step', self._episode_steps, ':')
        # print(self.needs)
        # print(self.actions)
        # print(reward)

        self.needs = self._split_x(self.n_agents, self.n_actions)
        #info['match'] = self._match

        if self._episode_steps >= self.episode_limit:
            # print(self._match)
            # print(reward)
            terminated = np.ones((self.num_agents, 1), dtype=bool)
            self._episode_count += 1
            self.battles_game += 1

            if self._match == self.n_actions * self.episode_limit:
                #info['battle_won'] = True
                self.battles_won += 1

        return self.get_obs(), self.get_state(),reward, terminated, info, self.get_avail_actions()

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        agent_action = self.actions[agent_id]
        # print([agent_action, self.needs[agent_action]])
        # print([float(x) for x in (self.actions == agent_action)])
        action_one_hot = np.zeros(self.n_actions)
        action_one_hot[agent_action] = 1.
        return np.concatenate((action_one_hot, [self.needs[agent_action]], [float(x) for x in (self.actions == agent_action)]))
        # return np.array([agent_action, self.needs[agent_action], (self.actions == agent_action).sum()])

    def get_obs_size(self):
        """Returns the size of the observation."""
        return [self.n_actions + 1 + self.n_agents]

    def get_state(self):
        """Returns the global state."""
        return np.concatenate(self.get_obs())

    def get_state_size(self):
        """Returns the size of the global state."""
        return [self.n_agents * self.get_obs_size()[0]]

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.needs = self.initial_need
        self.actions = np.random.randint(0, self.n_actions, self.n_agents)
        self._match = 0

        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game
        }
        return stats