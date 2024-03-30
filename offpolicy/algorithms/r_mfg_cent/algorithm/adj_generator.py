#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:03:55 2018

@author: xinruyue
"""
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from offpolicy.utils.util import gumbel_softmax_mdfg



# Network Generator
# 此类为一个利用Gumbel softmax生成离散网络的类
class Adj_Generator(nn.Module):
    def __init__(self, args, obs_dim ,state_dim,device, temp = 10, temp_drop_frac = 0.999):
        super(Adj_Generator, self).__init__()
        #self.distribution = torch.distributions.bernoulli.Bernoulli(0.1)
        self.cluster_rep = Parameter(torch.empty((1, 2),dtype=torch.float32).uniform_(-1,1))
        self.device = device
        self.to(device) 

    def get_random_adj(self):
        index_factor = np.arange(self.num_variable)
        adj = np.zeros((self.num_variable,self.num_factor))
        for i in range(self.num_factor):
            num_orders = np.random.choice([1,2,3],1)
            adj[np.random.choice(index_factor,num_orders,False),i] = 1
        return adj

    def drop_temperature(self):
        # 降温过程
        self.temperature = self.temperature * self.temp_drop_frac

    def sample(self, hard=False):
        # 采样——得到一个临近矩阵
        self.logp = self.gen_matrix.view(-1, 2)
        #self.logp[:,1] -= 0.4
        
        out = gumbel_softmax_mdfg(self.logp, self.temperature, hard, self.num_factor,self.highest_orders,self.device,True)

        return out
    def get_temperature(self):
        return self.temperature
    def get_cross_entropy(self, obj_matrix):
        # 计算与目标矩阵的距离
        logps = F.softmax(self.gen_matrix, 2)
        logps = torch.log(logps[:,:,0] + 1e-10) * obj_matrix + torch.log(logps[:,:,1] + 1e-10) * (1 - obj_matrix)
        result = - torch.sum(logps)
        result = result.cpu() if use_cuda else result
        return result.data.numpy()
    def get_entropy(self):
        logps = F.softmax(self.gen_matrix, 2)
        result = torch.mean(torch.sum(logps * torch.log(logps + 1e-10), 1))
        result = result.cpu() if use_cuda else result
        return(- result.data.numpy())
    def randomization(self, fraction):
        # 将gen_matrix重新随机初始化，fraction为重置比特的比例
        sz = self.gen_matrix.size()[0]
        numbers = int(fraction * sz * sz)
        original = self.gen_matrix.cpu().data.numpy()
        
        for i in range(numbers):
            ii = np.random.choice(range(sz), (2, 1))
            z = torch.rand(2).cuda() if use_cuda else torch.rand(2)
            self.gen_matrix.data[ii[0], ii[1], :] = z


