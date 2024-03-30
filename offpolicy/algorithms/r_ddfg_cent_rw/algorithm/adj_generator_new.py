#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:03:55 2018

@author: xinruyue
"""
import sys
sys.path.append('..')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from offpolicy.utils.util import gumbel_softmax_mdfg, to_torch, update_linear_schedule, DecayThenFlatSchedule
from offpolicy.algorithms.utils.autoencoder import Autoencoder
from offpolicy.algorithms.utils.adj_policy import AdjPolicy
# Network Generator
# 此类为一个利用Gumbel softmax生成离散网络的类
class Adj_Generator(nn.Module):
    def __init__(self, args, obs_dim ,state_dim, device):
        super(Adj_Generator, self).__init__()
        #self.distribution = torch.distributions.bernoulli.Bernoulli(0.1)
        self.adj_hidden_dim = args.adj_hidden_dim
        self.adj_output_dim = args.adj_output_dim
        self.use_epsilon_greedy = args.use_epsilon_greedy
        self.num_variable = args.num_agents
        self.num_factor = args.num_factor
        self.alpha = args.adj_alpha
        self.num = 1
        self.device = device
        self.adj_policy = AdjPolicy(args,args.hidden_size ,state_dim,device,args.use_ReLU)
        self.exploration = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.adj_anneal_time,decay="linear")
        #gen_matrix 为邻接矩阵的概率
        self.device = device
        self.highest_orders = args.highest_orders
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.to(device) 

    def sample(self, obs, state, use_adj_init, explore=False, t_env=None):
        # 采样——得到一个临近矩阵   
        batch_size = obs.shape[0]
        input_batch = to_torch(obs).to(**self.tpdv)
        if len(state.shape) == 1:
            state_batch = to_torch(state).to(**self.tpdv).unsqueeze(0)
        else:
            state_batch = to_torch(state).to(**self.tpdv)
        softmax, log_probs = self.adj_policy(input_batch,state_batch,use_adj_init)
        softmax_pre =  softmax.transpose(1,2)
        
        if explore:
            if self.use_epsilon_greedy:
                eps = torch.tensor(self.exploration.eval(t_env))
                rand_numbers = torch.rand(batch_size,self.num_factor,1)
                take_random = torch.where(rand_numbers < eps,torch.ones_like(rand_numbers,dtype=torch.int64),torch.zeros_like(rand_numbers,dtype=torch.int64)).to(self.device)

                x = softmax_pre.reshape(-1,self.num_variable).shape[0]
                random_probability = softmax_pre.new_ones((x,self.num_variable))
                random_indices = torch.multinomial(random_probability, self.highest_orders,replacement=False).reshape(batch_size,-1,self.highest_orders)
                greedy_indices = torch.topk(softmax_pre.reshape(-1,self.num_variable), k=self.highest_orders, dim=1, largest=True)[1].reshape(batch_size,-1,self.highest_orders)
                indices =  (1 - take_random) * greedy_indices + take_random * random_indices
            else:
                indices = torch.multinomial(softmax_pre.reshape(-1,self.num_variable), self.highest_orders,replacement=True).reshape(batch_size,-1,self.highest_orders)

        else:
            value, indices = torch.topk(softmax_pre.reshape(-1,self.num_variable), k=self.highest_orders, dim=1, largest=True)
            value = value.reshape(batch_size,-1,self.highest_orders)
            indices = indices.reshape(batch_size,-1,self.highest_orders)
            if self.highest_orders == 3:
                p_order3 = value[...,0]**3
                p_order2 = 3*value[...,1]*value[...,2]*(value[...,1]+value[...,2])
                p_order1 = 6*value[...,0]*value[...,1]*value[...,2]
                chosen_order3 = (p_order3 > p_order2) & (p_order3 > p_order1)
                tmp_order3 = indices[chosen_order3]
                tmp_order3[:,1] = tmp_order3[:,0]
                tmp_order3[:,2] = tmp_order3[:,0]
                indices[chosen_order3] = tmp_order3
                chosen_order2 = (p_order2 >= p_order3) & (p_order2 > p_order1)
                tmp_order2 = indices[chosen_order2]
                tmp_order2[:,2] = tmp_order2[:,0]
                indices[chosen_order2] = tmp_order2
            else:
                chosen = (value[...,0]**2) > (2*value[...,1])
                tmp_idx = indices[chosen]
                tmp_idx[:,1] = tmp_idx[:,0]
                indices[chosen] = tmp_idx
        #softmax[0][0] = 0
        entropy = -softmax * log_probs
        
        x = torch.ones_like(softmax,dtype=torch.int64)
        y = torch.zeros_like(softmax,dtype=torch.int64)
        if explore:
            cond_adj_1 = torch.where(softmax>1e-2,x,y)
        else:
            cond_adj_1 = torch.where(softmax>1e-2,x,y)
        cond_adj_2 = torch.zeros_like(softmax,dtype=torch.int64)
        cond_adj_2= cond_adj_2.transpose(1,2).scatter(2,indices,1).transpose(1,2)
        cond_adj = cond_adj_1 & cond_adj_2
        
        return softmax, cond_adj, entropy.sum(-2).mean(-1)

    def parameters(self):
        parameters_sum = []
        #parameters_sum += self.autoencoder.parameters()
        parameters_sum += self.adj_policy.parameters()

        return parameters_sum

    def load_state(self, source_adjnetwork):
        #self.autoencoder.load_state_dict(source_adjnetwork.autoencoder.state_dict())
        self.adj_policy.load_state_dict(source_adjnetwork.adj_policy.state_dict())
