#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
using namespace std;

const int maxN = 50; // number of agents
const int maxM = 50; // number of edges
const int MAX_BATCH_SIZE = 35;

class Tree_Solver
{

    int n, m;
    vector<pair<double, pair<int, int> > > Edge;
    int fa[maxN];

    int find(int x)
    {
        return fa[x] == x ? x : fa[x] = find(fa[x]);
    }

public:
    void solve(double *py_g, double *best_graph, int py_n)
    {
        n = py_n;
        Edge.clear();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Edge.push_back(make_pair(py_g[i * py_n + j], make_pair(i, j)));
        m = Edge.size();
        sort(Edge.begin(), Edge.end());
        for (int i = 1; i <= n; i++)
            fa[i] = i;
        for (int i = m - 1; i >= 0; i--)
        {
            int u = Edge[i].second.first + 1, v = Edge[i].second.second + 1;
            if (find(u) != find(v))
            {
                best_graph[(u - 1) * n + v - 1] = best_graph[(v - 1) * n + u - 1] = 1;
                fa[find(u)] = find(v);
            }
        }
    }
};

Tree_Solver tree_solver[MAX_BATCH_SIZE];

extern "C" void
maximum_spanning_tree(double *py_g, double *best_graph, int py_bs, int py_n)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < py_bs; i++)
        tree_solver[i].solve(py_g + i * py_n * py_n, best_graph + i * py_n * py_n, py_n);
}

//-------------------------------------------------------

class DCOP_Solver
{
    int n, m, n_edge;
    vector<pair<int, int> > G[maxN];
    double value[maxN * 2][maxM][maxM];
    double value_f[maxN][maxM];
    double dp[maxN][maxM];
    int vi[maxN];

    void dfs_dp(int u)
    {
        vi[u] = 1;
        for (int j = 1; j <= m; j++)
            dp[u][j] = value_f[u][j];
        for (int i = 0; i < G[u].size(); i++)
        {
            int v = G[u][i].first, e = G[u][i].second;
            if (!vi[v])
            {
                dfs_dp(v);
                for (int j = 1; j <= m; j++)
                {
                    double max_value = -1e30;
                    for (int k = 1; k <= m; k++)
                        max_value = max(max_value, dp[v][k] + value[e][j][k]);
                    dp[u][j] += max_value;
                }
            }
        }
    }

    void dfs_construct(int u, int action, double *best_actions)
    {
        best_actions[u - 1] = action - 1;
        vi[u] = 1;
        for (int i = 0; i < G[u].size(); i++)
        {
            int v = G[u][i].first, e = G[u][i].second;
            if (!vi[v])
            {
                double max_value = -1e30;
                int action_v = 0;
                for (int j = 1; j <= m; j++)
                    if (dp[v][j] + value[e][action][j] > max_value)
                    {
                        max_value = dp[v][j] + value[e][action][j];
                        action_v = j;
                    }
                dfs_construct(v, action_v, best_actions);
            }
        }
    }

public:
    void solve(double *py_f, double *py_g, double *graphs, double *best_actions, int py_n, int py_m)
    {
        n = py_n;
        m = py_m;
        n_edge = 0;
        for (int i = 1; i <= n; i++)
            G[i].clear();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++)
                if (graphs[i * n + j] > 0.5)
                {
                    G[i + 1].push_back(make_pair(j + 1, n_edge));
                    for (int k = 0; k < m; k++)
                        for (int l = 0; l < m; l++)
                            value[n_edge][k + 1][l + 1] = py_g[i * n * m * m + j * m * m + k * m + l];
                    n_edge++;
                    G[j + 1].push_back(make_pair(i + 1, n_edge));
                    for (int k = 0; k < m; k++)
                        for (int l = 0; l < m; l++)
                            value[n_edge][l + 1][k + 1] = py_g[i * n * m * m + j * m * m + k * m + l];
                    n_edge++;
                }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                value_f[i + 1][j + 1] = py_f[i * m + j];
        for (int i = 1; i <= n; i++)
            vi[i] = 0;
        for (int i = 1; i <= n; i++)
            if (!vi[i])
                dfs_dp(i);
        for (int i = 1; i <= n; i++)
            vi[i] = 0;
        for (int i = 1; i <= n; i++)
            if (!vi[i])
            {
                double max_value = -1e30;
                int action = 0;
                for (int j = 1; j <= m; j++)
                    if (dp[i][j] > max_value)
                    {
                        max_value = dp[i][j];
                        action = j;
                    }
                dfs_construct(i, action, best_actions);
            }
    }
};

DCOP_Solver dcop_solver[MAX_BATCH_SIZE];

extern "C" void
solve_tree_DCOP(double *py_f, double *py_g, double *graph, double *best_actions, int py_bs, int py_n, int py_m)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < py_bs; i++)
        dcop_solver[i].solve(py_f + i * py_n * py_m, py_g + i * py_n * py_n * py_m * py_m, graph + i * py_n * py_n, best_actions + i * py_n, py_n, py_m);
}

//-------------------------------------------------------

class GreedySpanningTree_solver
{
    struct edge
    {
        int u, v, tag;
        double f[maxM], maxf;
        edge *nxt;
    } pool[maxN * 3], *tp, *fst[maxN], *e[maxN];
    int n, m, seq[maxN][maxN][maxM];
    double *f, *g, delta[maxN][maxN], maxg2[maxN][maxN], maxg3[maxN][maxN][maxM];
    bool stale[maxN][maxN];

    double value(int i, int j, int ai, int aj)
    {
        i -= 1, j -= 1, ai -= 1, aj -= 1;
        return g[i * n * m * m + j * m * m + ai * m + aj];
    }

    double value_f(int i,int ai)
    {
        i -= 1, ai -= 1;
        return f[i * m + ai];
    }

    void add_edge(int u, int v)
    {
        tp->u = u, tp->v = v;
        tp->nxt = fst[u], fst[u] = tp++;
    }

    void dp(edge *e, int tag)
    {
        if (e->tag == tag)
            return;
        e->tag = tag;
        e->maxf = -1e30;
        for (int i = 1; i <= m; ++i)
            e->f[i] = value_f(e->v, i);
        for (edge *son = fst[e->v]; son; son = son->nxt)
            if (son->v != e->u)
            {
                dp(son, tag);
                for (int i = 1; i <= m; ++i)
                {
                    double maxv = -1e30;
                    for (int j = 1; j <= m; ++j)
                        maxv = max(maxv, son->f[j] + value(e->v, son->v, i, j));
                    e->f[i] += maxv;
                }
            }
        for (int i = 1; i <= m; ++i)
            if (e->f[i] > e->maxf)
                e->maxf = e->f[i];
    }

    public:
    void solve(double *py_f, double *py_g, double *best_graph, int py_n, int py_m)
    {
        n = py_n, m = py_m, f = py_f, g = py_g;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                best_graph[i * n + j] = 0;
        memset(pool, 0, sizeof(pool)), tp = pool;
        memset(fst, 0, sizeof(fst));
        for (int i = 1; i <= n; ++i)
            e[i] = tp, add_edge(0, i), dp(e[i], i);
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= n; ++j)
            {
                maxg2[i][j] = -1e30;
                for (int ai = 1; ai <= m; ++ai)
                {
                    maxg3[i][j][ai] = -1e30;
                    for (int aj = 1; aj <=m; ++aj)
                        maxg3[i][j][ai] = max(maxg3[i][j][ai], value(i, j, ai, aj));
                    maxg2[i][j] = max(maxg2[i][j], maxg3[i][j][ai]);
                    for (int k = ai - 1; k >= 0; --k)
                        if (k == 0 || maxg3[i][j][seq[i][j][k]] > maxg3[i][j][ai])
                        {
                            seq[i][j][k + 1] = ai;
                            break;
                        }
                        else
                            seq[i][j][k + 1] = seq[i][j][k];
                }
                delta[i][j] = maxg2[i][j];
            }
        memset(stale, 0, sizeof(stale));
        for (int k = 1; k < n; ++k)
        {
            double max_delta = -1e30;
            int u, v;
            for (int i = 1; i <= n; ++i)
                for (int j = i + 1; j <= n; ++j)
                    if (e[i]->tag != e[j]->tag && !stale[i][j] && delta[i][j] > max_delta)
                        max_delta = delta[i][j], u = i, v = j;
            for (int i = 1; i <= n; ++i)
                for (int j = i + 1; j <= n; ++j)
                    if (e[i]->tag != e[j]->tag && stale[i][j] && maxg2[i][j] > max_delta)
                    {
                        stale[i][j] = false, delta[i][j] = -1e30;
                        for (int k = 1; k <= m; ++k)
                        {
                            int ai = seq[i][j][k];
                            if (maxg3[i][j][ai] + e[i]->f[ai] + e[j]->maxf > delta[i][j])
                                for (int aj = 1; aj <= m; ++aj)
                                    delta[i][j] = max(delta[i][j], value(i, j, ai, aj) + e[i]->f[ai] + e[j]->f[aj]);
                        }
                        delta[i][j] -= e[i]->maxf + e[j]->maxf;
                        if (delta[i][j] > max_delta)
                            max_delta = delta[i][j], u = i, v = j;
                    }
            add_edge(u, v), add_edge(v, u);
            best_graph[(u - 1) * n + v - 1] = best_graph[(v - 1) * n + u - 1] = 1;
            int tag_u = e[u]->tag, tag_v = e[v]->tag;
            for (int i = 1; i <= n; ++i)
                if (e[i]->tag == tag_u || e[i]->tag == tag_v)
                    dp(e[i], n + k);
            for (int i = 1; i <= n; ++i)
                for (int j = i + 1; j <= n; ++j)
                    if (e[i]->tag != e[j]->tag && (e[i]->tag == n + k || e[j]->tag == n + k))
                        stale[i][j] = true;
        }
    }
};

GreedySpanningTree_solver greedy[MAX_BATCH_SIZE];

extern "C" void
greedy_spanning_tree(double *py_f, double *py_g, double *best_graphs, int py_bs, int py_n, int py_m)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < py_bs; i++)
        greedy[i].solve(py_f + i * py_n * py_m, py_g + i * py_n * py_n * py_m * py_m, best_graphs + i * py_n * py_n, py_n, py_m);
}

//-------------------------------------------------------

class GraphEpsilonGreedy_solver
{
    int n, m, fa[maxN];
    bool flag;

    int find(int x)
    {
        return fa[x] == x ? x : fa[x] = find(fa[x]);
    }

    public:
    void solve(double *graph, int py_n, double eps)
    {
        n = py_n, m = n - 1;
        if (!flag)
            srand(time(0)), flag = true;
        for (int i = 0; i < n; ++i)
            fa[i] = i;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (graph[i * n + j] > 0.5)
                    if (double(rand()) / RAND_MAX < eps)
                        graph[i * n + j] = graph[j * n + i] = 0, --m;
                    else
                        fa[find(i)] = j;
        for (; m < n - 1; ++m)
        {
            int x = 0, y = 0;
            while (find(x) == find(y))
                x = rand() % n, y = rand() % n;
            fa[find(x)] = y;
            graph[x * n + y] = graph[y * n + x] = 1;
        }
    }
};

GraphEpsilonGreedy_solver epsgreedy[MAX_BATCH_SIZE];

extern "C" void
graph_epsilon_greedy(double *graphs, int py_bs, int py_n, double eps)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < py_bs; i++)
        epsgreedy[i].solve(graphs + i * py_n * py_n, py_n, eps);
}