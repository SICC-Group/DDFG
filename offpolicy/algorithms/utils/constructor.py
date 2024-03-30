import torch as th
import numpy as np
from offpolicy.algorithms.utils.c_utils import load_c_lib, c_ptr, c_int, c_longlong, c_float, c_double
import copy
import ctypes
import torch_scatter


def preprocess_values(f, g, avail_actions):
    n_agents, n_actions = f.shape[1], f.shape[2]
    if not th.is_tensor(avail_actions):
        avail_actions = th.tensor(avail_actions)
    f[avail_actions == 0] = -9999999
    g[avail_actions.unsqueeze(1).unsqueeze(-2).repeat(1, n_agents, 1, n_actions, 1) == 0] = -9999999
    g[avail_actions.unsqueeze(2).unsqueeze(-1).repeat(1, 1, n_agents, 1, n_actions) == 0] = -9999999
    return f, g


def preprocess_values_given_actions(f, g, actions):
    n_agents, n_actions = f.shape[1], f.shape[2]
    f = th.gather(f, dim=-1, index=actions).squeeze(-1)
    g = th.gather(g, dim=-1, index=actions.unsqueeze(1).unsqueeze(-2).repeat(1, n_agents, 1, n_actions, 1)).squeeze(-1)
    g = th.gather(g, dim=-1, index=actions.unsqueeze(2).repeat(1, 1, n_agents, 1)).squeeze(-1)
    return f, g


class MatchingSolver:
    def __init__(self, args):
        self.blossom_lib = load_c_lib('./src/utils/blossom_lib.cpp')
        self.individual_q = args.individual_q

    def brute_force(self, f, g, avail_actions, device):  # bs,n,A; bs,n,n,A,A; bs,n,A
        f, g = f.detach(), g.detach()
        f, g = preprocess_values(f, g, avail_actions)
        f = f.max(dim=-1)[0]
        g = g.max(dim=-1)[0].max(dim=-1)[0]

        bs, n = f.shape[0], f.shape[1]
        best_scores = th.zeros(bs, ).to(device)
        best_graphs = th.zeros(bs, n, n).to(device)
        graph = th.zeros(n, n)
        matched = [False] * n

        def dfs(b, k, score, rest_isolated):
            if k == n:
                if score > best_scores[b]:
                    best_scores[b] = score
                    best_graphs[b] = graph.clone()
                return
            if matched[k] == True:
                dfs(b, k + 1, score, rest_isolated)
            else:
                for i in range(k + 1, n):
                    if matched[i] == False:
                        matched[k], matched[i] = True, True
                        graph[k, i], graph[i, k] = 1, 1
                        dfs(b, k + 1, score + g[b, k, i], rest_isolated)
                        matched[k], matched[i] = False, False
                        graph[k, i], graph[i, k] = 0, 0
                if rest_isolated > 0:
                    dfs(b, k + 1, score + f[b, k], rest_isolated - 1)

        for b in range(bs):
            dfs(b, 0, 0.0, n % 2)
        return best_graphs

    def solve(self, f, g, avail_actions, device):
        f, g = f.detach(), g.detach()
        f, g = preprocess_values(f, g, avail_actions)
        bs, n, a = f.shape[0], f.shape[1], f.shape[2]

        if self.individual_q:
            g = g + f.unsqueeze(1).unsqueeze(-2).repeat(1, n, 1, a, 1) + f.unsqueeze(2).unsqueeze(-1).repeat(1, 1, n, 1, a)
            g = g.max(dim=-1)[0].max(dim=-1)[0]
            f = f.max(dim=-1)[0]
            g = g - f.unsqueeze(-1).repeat(1, 1, n) - f.unsqueeze(-2).repeat(1, n, 1)
        else:
            f = f.max(dim=-1)[0]
            g = g.max(dim=-1)[0].max(dim=-1)[0]
            g = g - f.unsqueeze(-1).repeat(1, 1, n) - f.unsqueeze(-2).repeat(1, n, 1)
        g = g + 10000  # Restrict to maximal matchings

        _f = np.array(copy.deepcopy(g).cpu()).astype(ctypes.c_double)
        _best_graphs = np.zeros((bs, n, n)).astype(ctypes.c_double)

        self.blossom_lib.blossom_solve_para(c_ptr(_f), c_ptr(_best_graphs), bs, n)

        best_graphs = th.tensor(copy.deepcopy(_best_graphs)).to(device)

        best_graphs = best_graphs.float()

        return best_graphs

    def solve_given_actions(self, f, g, actions, device):
        f, g = f.detach(), g.detach()
        f, g = preprocess_values_given_actions(f, g, actions)
        bs, n = f.shape[0], f.shape[1]

        if self.individual_q:
            pass
        else:
            g = g - f.unsqueeze(-1).repeat(1, 1, n) - f.unsqueeze(-2).repeat(1, n, 1)
        g = g + 10000  # Restrict to maximal matchings

        _f = np.array(copy.deepcopy(g).cpu()).astype(ctypes.c_double)
        _best_graphs = np.zeros((bs, n, n)).astype(ctypes.c_double)

        self.blossom_lib.blossom_solve_para(c_ptr(_f), c_ptr(_best_graphs), bs, n)

        best_graphs = th.tensor(copy.deepcopy(_best_graphs)).to(device)

        best_graphs = best_graphs.float()

        return best_graphs

    # def graph_epsilon_greedy(self, graphs, eps):
    #     _graphs = np.array(copy.deepcopy(graphs.detach()).cpu()).astype(ctypes.c_double)
    #     self.blossom_lib.graph_epsilon_greedy(c_ptr(_graphs), graphs.shape[0], graphs.shape[1], c_double(eps))
    #     new_graphs = th.tensor(copy.deepcopy(_graphs)).to(dtype=graphs.dtype, device=graphs.device)
    #     return new_graphs


class TreeSolver:
    def __init__(self, args):
        self.tree_lib = load_c_lib('../algorithms/utils/tree.cpp')
        self.individual_q = True

    def solve(self, f, g, avail_actions, device):
        f, g = f.detach(), g.detach()
        f, g = preprocess_values(f, g, avail_actions)
        bs, n, m = f.shape

        if self.individual_q:
            _f = np.array(copy.deepcopy(f).cpu()).astype(ctypes.c_double)
        else:
            _f = np.zeros(f.shape).astype(ctypes.c_double)
        _g = np.array(copy.deepcopy(g).cpu()).astype(ctypes.c_double)
        _best_graphs = np.zeros((bs, n, n)).astype(ctypes.c_double)

        self.tree_lib.greedy_spanning_tree(c_ptr(_f), c_ptr(_g), c_ptr(_best_graphs), bs, n, m)

        best_graphs = th.tensor(copy.deepcopy(_best_graphs)).to(device)

        best_graphs = best_graphs.float()

        return best_graphs

    def solve_given_actions(self, f, g, actions, device):
        f, g = f.detach(), g.detach()
        f, g = preprocess_values_given_actions(f, g, actions)

        bs, n = f.shape[0], f.shape[1]

        _g = np.array(copy.deepcopy(g).cpu()).astype(ctypes.c_double)
        _best_graphs = np.zeros((bs, n, n)).astype(ctypes.c_double)

        self.tree_lib.maximum_spanning_tree(c_ptr(_g), c_ptr(_best_graphs), bs, n)

        best_graphs = th.tensor(copy.deepcopy(_best_graphs)).to(device)

        best_graphs = best_graphs.float()

        return best_graphs

    # def graph_epsilon_greedy(self, graphs, eps):
    #     _graphs = np.array(copy.deepcopy(graphs.detach()).cpu()).astype(ctypes.c_double)
    #     self.tree_lib.graph_epsilon_greedy(c_ptr(_graphs), graphs.shape[0], graphs.shape[1], c_double(eps))
    #     new_graphs = th.tensor(copy.deepcopy(_graphs)).to(dtype=graphs.dtype, device=graphs.device)
    #     return new_graphs

class LineSolver:
    def __init__(self, args):
        self.tree_lib = load_c_lib('./src/utils/tree.cpp')
        self.individual_q = args.individual_q

        n = args.n_agents
        self.line_graph = th.zeros((n, n))
        for i in range(n - 1):
            self.line_graph[i, i + 1] = 1.
            self.line_graph[i + 1, i] = 1.
    
    def solve(self, f, g, avail_actions, device):
        bs, n, m = f.shape

        best_graphs = self.line_graph.clone().detach().unsqueeze(0).repeat(bs, 1, 1).to(device)
  
        return best_graphs

    def solve_given_actions(self, f, g, actions, device):
        bs, n, m = f.shape

        best_graphs = self.line_graph.clone().detach().unsqueeze(0).repeat(bs, 1, 1).to(device)

        return best_graphs

    def graph_epsilon_greedy(self, graphs, eps):
        _graphs = np.array(copy.deepcopy(graphs.detach()).cpu()).astype(ctypes.c_double)
        self.tree_lib.graph_epsilon_greedy(c_ptr(_graphs), graphs.shape[0], graphs.shape[1], c_double(eps))
        new_graphs = th.tensor(copy.deepcopy(_graphs)).to(dtype=graphs.dtype, device=graphs.device)
        return new_graphs


class StarSolver:
    def __init__(self, args):
        self.tree_lib = load_c_lib('./src/utils/tree.cpp')
        self.individual_q = args.individual_q

        n = args.n_agents
        self.star_graph = th.zeros((n, n))
        for i in range(n - 1):
            self.star_graph[0, i + 1] = 1.
            self.star_graph[i + 1, 0] = 1.
    
    def solve(self, f, g, avail_actions, device):
        bs, n, m = f.shape

        best_graphs = self.star_graph.clone().detach().unsqueeze(0).repeat(bs, 1, 1).to(device)

        return best_graphs

    def solve_given_actions(self, f, g, actions, device):
        bs, n, m = f.shape

        best_graphs = self.star_graph.clone().detach().unsqueeze(0).repeat(bs, 1, 1).to(device)

        return best_graphs

    def graph_epsilon_greedy(self, graphs, eps):
        _graphs = np.array(copy.deepcopy(graphs.detach()).cpu()).astype(ctypes.c_double)
        self.tree_lib.graph_epsilon_greedy(c_ptr(_graphs), graphs.shape[0], graphs.shape[1], c_double(eps))
        new_graphs = th.tensor(copy.deepcopy(_graphs)).to(dtype=graphs.dtype, device=graphs.device)
        return new_graphs


class Constructor:
    def __init__(self, args, n_agents, act_dim):
        self.args = args
        self.n_agents = n_agents
        self.n_actions = act_dim
        
        self.tree_lib = load_c_lib('../algorithms/utils/tree.cpp')

    # graphs is bs,n,n tensor indicating the edges
    # g must be symmetric!!!

    def compute_values_given_actions(self, f, g, actions, graphs):
        f, g = preprocess_values_given_actions(f, g, actions)
        if self.args.individual_q:
            values = f.sum(dim=-1) + (g * graphs).sum(dim=-1).sum(dim=-1) / 2  # /2 since each edge is computed twice
        else:
            isolated_nodes = th.max(1 - graphs.sum(dim=-1), th.zeros_like(graphs.sum(dim=-1)))
            values = (f * isolated_nodes).sum(dim=-1) + (g * graphs).sum(dim=-1).sum(dim=-1) / 2  # /2 since each edge is computed twice
        return values

    def compute_outputs(self, f, g, avail_actions, graphs):
        f, g = preprocess_values(f, g, avail_actions)
        if self.args.construction == 'matching':
            if self.args.individual_q:
                n, a = f.shape[1], f.shape[2]
                g = g + f.unsqueeze(1).unsqueeze(-2).repeat(1, n, 1, a, 1) + f.unsqueeze(2).unsqueeze(-1).repeat(1, 1, n, 1, a)
                g = g.max(dim=-1)[0]
                isolated_nodes = th.max(1 - graphs.sum(dim=-1), th.zeros_like(graphs.sum(dim=-1)))
                agent_outputs = f * isolated_nodes.unsqueeze(-1) + (g * graphs.unsqueeze(-1)).sum(dim=-2) / 2  # /2 since each edge is computed twice
            else:
                g = g.max(dim=-1)[0]
                isolated_nodes = th.max(1 - graphs.sum(dim=-1), th.zeros_like(graphs.sum(dim=-1)))
                agent_outputs = f * isolated_nodes.unsqueeze(-1) + (g * graphs.unsqueeze(-1)).sum(dim=-2) / 2  # /2 since each edge is computed twice
        elif self.args.construction in ['tree', 'line', 'star'] :
            bs, n, m = f.shape[0], f.shape[1], self.n_actions

            if self.args.individual_q:
                _f = np.array(copy.deepcopy(f.detach()).cpu()).astype(ctypes.c_double)
            else:
                _f = np.zeros(f.shape).astype(ctypes.c_double)
            _g = np.array(copy.deepcopy(g.detach()).cpu()).astype(ctypes.c_double)
            _graphs = np.array(copy.deepcopy(graphs.detach()).cpu()).astype(ctypes.c_double)
            _best_actions = np.zeros((bs, n)).astype(ctypes.c_double)

            self.tree_lib.solve_tree_DCOP(c_ptr(_f), c_ptr(_g), c_ptr(_graphs), c_ptr(_best_actions), bs, n, m)

            best_actions = th.tensor(copy.deepcopy(_best_actions), dtype=th.int64, device=f.device).unsqueeze(-1)

            agent_outputs = f.new_zeros(f.shape[0], self.n_agents, self.n_actions)
            agent_outputs.scatter_(dim=-1, index=best_actions,
                                   src=agent_outputs.new_ones(1, 1, 1).expand_as(best_actions))
        else:
            raise Exception('unimplemented')
        return agent_outputs

