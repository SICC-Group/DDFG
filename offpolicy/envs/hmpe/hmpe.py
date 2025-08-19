from __future__ import annotations
from typing import Any, Dict, Generic, Iterable, Iterator, TypeVar
from copy import copy

from numpy import ndarray

import numpy as np

import itertools
import random

class Agent:
    def __init__(self, name=None, x=0, y=0) -> None:
        self.pos_x = x
        self.pos_y = y
        self.name = None
        self.cap = 0

    def reset(self):
        self.cap = 0

class Doo:
    def __init__(self, w = 1, name=None) -> None:
        self.pos_x = 0
        self.pos_y = 0
        self.w = w
        self.name = None
        self.split = False
    
    def reset(self):
        self.split = False

ACTION_TO_STR = {
    0: 'no_action',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right',
    5: 'pickup',
    6: 'putdown',
    7: 'split',
}

FindTrash,PickTrash,SplitBig,PutTrash = list(range(4))

Up = 0
Down = 1
Left = 2
Right = 3
Pickup = 4
Putdown = 5
Split = 6

moveagent = 0
sTrash = 1
bTrash = 2
Bin = 3
# bin: 4 回收站 

# class hmpe(AECEnv):
class hmpe():
    metadata = {
        'name': 'hmpe_v0',
    }

    def __init__(self, num_agents = 3, n_k1 = 3, n_k2 = 3, max_cycles = 50, is_goaltrain = False) -> None:
        """
        action_space: no_action, up, down, left, right, pickup, putdown, split
        obs_space:
        
        state_space:
        """
        self.action_space = 7
        self.obs_view = (5,5)
        self.obs_dim = self.obs_view[0] * self.obs_view[1] * 4 + 4
        

        self.height = 10
        self.weight = 10
        self.num_agents = num_agents

        self.state_dim = self.height * self.weight * 4

        self.max_cycles = max_cycles
        self.num_envs = 1
        # State space
        # curAgent: 1
        # otherAgent: 2
        # empty: 0
        # trash: 3
        # bin: 4 回收站 
        # x 是纵轴 y 是横轴

        # self.agnts = [Agent('agent_' + str(i)) for i in range(self.num_agents)]

        self.n_agents = {}
        #self._startpos = [(0, 0), (0, 2), (0, 4)]
        for i in range(self.num_agents):
            self.n_agents[i] = Agent('agent_' + str(i), 0, i * 2)

        # for a in self.n_agents:
        #     self.observations[a] = np.zeros((1, self.height, self.weight))

        # K1个小的物体，K2个大的物体
        self.max_w = 3
        self.find_reward = 0
        self.split_reward = 0
        self.pick_reward = 0
        self.put_reward = 1
        self.collision_reward = 0
        self.penalty = 0
        self.n_k1, self.n_k2 = n_k1, n_k2
        self.s_w = 1
        self.k1s = [Doo(1) for _ in range(self.n_k1)]
        self.k2s = [Doo(3) for _ in range(self.n_k2)]
        self.remain = n_k1 + n_k2
        self.move_base = np.array([[-1,0],[1,0],[0,-1],[0,1]])

        self._goal_space = [0,1,2,3]
        #['FindTrash','PickTrash','SplitBig','PutTrash']

        self.is_goaltrain = is_goaltrain
        self.goals = np.zeros((self.num_agents))

    @property
    def goal_space(self):
        return len(self._goal_space)

    def setgoals(self, agent, goal):

        self.goals[agent] = goal

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict[Any, dict]]:
        # self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.remain = self.n_k1 + self.n_k2
        self.states = np.zeros((4, self.height, self.weight))

        _pos = list(itertools.product(range(1, self.height-2), range(0, self.weight-2)))
        pos_list = random.sample(_pos, self.num_agents + self.n_k1 + self.n_k2)

        # init 智能体和物体的位置
        _it = 0
        for a in self.n_agents:
            pos_x, pos_y = pos_list[_it]
            self.n_agents[a].pos_x = pos_x
            self.n_agents[a].pos_y = pos_y
            self.n_agents[a].reset()
            _it += 1 
            self.states[moveagent, pos_x, pos_y] = 1
        for k1 in self.k1s:
            k1.pos_x, k1.pos_y = pos_list[_it]
            k1.reset()
            _it += 1
            self.states[sTrash, k1.pos_x, k1.pos_y] = 1
        for k2 in self.k2s:
            k2.pos_x, k2.pos_y = pos_list[_it]
            k2.reset()
            _it += 1
            self.states[bTrash, k2.pos_x, k2.pos_y] = 1

        self.states[Bin, 8:, 6:] = 1
        
        return self.get_obs()[None], self.get_states(), np.ones((1,self.num_agents,self.action_space))
    
    def get_obs(self):
        observations = []
        for a in self.n_agents:
            pos_x, pos_y = self.n_agents[a].pos_x, self.n_agents[a].pos_y
            agent_obs = [pos_x / (self.height-1), pos_y / (self.weight-1)]
            other_pos = np.zeros((4,self.obs_view[0],self.obs_view[1]))
            for i in range(4):
                other_pos[i] = self.get_other_map(pos_x, pos_y, i)
            agent_obs += other_pos.flatten().tolist()
            agent_obs += [self.n_agents[a].cap/self.max_w]
            agent_obs += [self.timestep/self.max_cycles]
            observations.append(agent_obs)

        return np.array(observations)
    
    def get_states(self):

        return self.states.reshape(1,-1)

    
    def get_other_map(self, pos_x, pos_y, idx):
        dis_r = int(self.obs_view[0]-1)//2
        dis_c = int(self.obs_view[1]-1)//2
        row_min = max(0,pos_x-dis_r) #行
        row_max = min(self.height-1,pos_x+dis_c)+1
        col_min = max(0,pos_y-dis_r) #列
        col_max = min(self.weight-1,pos_y+dis_c)+1
        idx_state = self.states[idx].copy()
        if idx == moveagent:
            idx_state[pos_x,pos_y] = 0

        other_map = np.zeros((self.obs_view[0],self.obs_view[1])) 
        map_x = pos_x-dis_r
        map_y = pos_y-dis_r
        other_map[row_min-map_x:row_max-map_x,col_min-map_y:col_max-map_y] = idx_state[row_min:row_max,col_min:col_max]    
        return other_map
    
    def low_reward(self):
        return 1 - 0.5 * (self.timestep/self.max_cycles)
    
    def high_reward(self):
        return 1 - 0.2 * (self.timestep/self.max_cycles)
    
    def updateobs(self):
        return self.observations
    
    def get_high_action_mask(self):
        avail_highactions = np.ones((self.num_agents,self.goal_space),dtype=np.float32)
        if self.timestep == 0:
            avail_highactions[:,1:] = 0
            return avail_highactions
         
        for a in self.n_agents:
            if self.n_agents[a].cap == 0:
                avail_highactions[a,-1] = 0
            elif self.n_agents[a].cap == self.max_w:
                avail_highactions[a,:-1] = 0 
        
        return avail_highactions


    def step(self, action) -> tuple[dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]]:
        """
        ## TODO
        - clid: 智能体间的碰撞
        - reward: 全局的奖励
        Takes in an action for the current agent (specified by agent_selection).

        And any internal state used by observe() or render()
        """
        # 低层奖励
        # if low_level:
        reward = np.zeros((self.num_agents,1))
        low_reward = np.zeros((self.num_agents,1))
        getgoal = np.zeros((self.num_agents,1),dtype=np.bool_)
        
        # action = ACTION_TO_STR(_action)
        # X 行 Y 列
        for i in np.random.permutation(self.num_agents):
            cur_x = self.n_agents[i].pos_x
            cur_y = self.n_agents[i].pos_y
            if action[i] in [Up, Down, Left, Right]:
                next_x = max(min(cur_x+self.move_base[action[i]][0],self.height-1),0)
                next_y = max(min(cur_y+self.move_base[action[i]][1],self.weight-1),0)
                if self.states[moveagent, next_x, next_y] == 1:
                    reward[i] -= self.collision_reward
                else:
                    self.states[moveagent, cur_x, cur_y] = 0
                    self.states[moveagent, next_x,next_y] = 1
                    self.n_agents[i].pos_x = next_x
                    self.n_agents[i].pos_y = next_y
                    if self.states[bTrash, cur_x, cur_y] == 1 or self.states[sTrash, cur_x, cur_y] == 1:
                        reward[i] += self.find_reward * self.high_reward()
                    if self.goals[i] == self._goal_space[FindTrash] and (self.states[sTrash, next_x,next_y] == 1 or self.states[bTrash, next_x,next_y] == 1):
                        low_reward[i] += self.low_reward()
                        getgoal[i] = True
            elif action[i] == Pickup:
                if self.states[bTrash, cur_x, cur_y] == 1:
                    reward[i] -= self.penalty
                elif self.states[sTrash, cur_x, cur_y] == 1:
                    if self.n_agents[i].cap < self.max_w:
                        self.n_agents[i].cap += self.s_w
                        self.states[sTrash, cur_x, cur_y] = 0
                        reward[i] += self.pick_reward * self.high_reward()
                        if self.goals[i] == self._goal_space[PickTrash]:
                            low_reward[i] += self.low_reward()
                            getgoal[i] = True
                    else:
                        reward[i] -= self.penalty
                else:
                    reward[i] -= self.penalty
            elif action[i] == Putdown:
                if self.states[Bin, cur_x, cur_y] == 1 and self.n_agents[i].cap >= 1:
                    num_trash = self.n_agents[i].cap
                    self.remain -= num_trash
                    self.n_agents[i].cap = 0
                    reward[i] += (num_trash * self.high_reward() * self.put_reward)
                    # if self.remain == 0:
                    #     reward[i] += 5
                    #self.put_reward
                    if self.goals[i] == self._goal_space[PutTrash]:
                        low_reward[i] += self.low_reward()*num_trash
                        getgoal[i] = True
                else:
                    reward[i] -= self.penalty
            elif action[i] == Split:
                if self.states[bTrash, cur_x, cur_y] == 1:
                    self.states[bTrash, cur_x, cur_y] = 0
                    self.states[sTrash, cur_x, cur_y] = 1
                    reward[i] += self.split_reward * self.high_reward()
                    if self.goals[i] == self._goal_space[SplitBig]:
                        low_reward[i] += self.low_reward()
                        getgoal[i] = True
                elif self.states[sTrash, cur_x, cur_y] == 1:
                    reward[i] -= self.penalty
                else:
                    reward[i] -= self.penalty
        
        dones = np.zeros((self.num_agents,1),dtype=np.bool_)
        if (self.timestep >= self.max_cycles) | (self.remain == 0):
            # rewards = {}
            dones[:] = True
        
        # print("remian:",self.remain)
        self.timestep += 1

        infos = {a: {} for a in self.n_agents}
        # if any(terminations.values()) or all (truncations.values()):
        #     self.agents = []
        
        if self.is_goaltrain:
            return self.get_obs(), low_reward, getgoal, dones, infos
        else:
            return self.get_obs()[None], self.get_states(), reward[None], dones[None], infos, np.ones((1,self.num_agents,self.action_space))
    
    # def render(self) -> ndarray | str | list | None:
    #     return super().render()
    
    # def observation_space(self, agent: Any) -> Space:
    #     return super().observation_space(agent)
    
    # def action_space(self, agent: Any) -> Space:
    #     return Discrete(8)
        
if __name__ == '__main__':
    env = hmpe()
    obs, _ = env.reset()