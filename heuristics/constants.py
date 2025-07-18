import networkx as nx
import numpy as np

SEED : int = 42

N_SAMPLE_EVAL = 100_000
N_SAMPLE_LS = 10_000

LOCAL_SEARCH_ITER = 1
GRASP_RESTARTS = 3

K = 10
NODES = 100

# nodes 100, edges 200 (Sparse Graphs)
graph_models = {
  'ER': nx.erdos_renyi_graph(NODES, 0.0443, seed=SEED),
  'BA': nx.barabasi_albert_graph(NODES, 2,seed=SEED),
  'SW': nx.watts_strogatz_graph(NODES, 4, 0.3, seed=SEED)
}

# DENSE_NODES = 50 
# K = 5
# # nodes 50, p = 0.5 (Dense Graphs) 2500 edges

# graph_models_dense = {
#   'ER': nx.erdos_renyi_graph(DENSE_NODES, 0.5025, seed=SEED),
#   'BA': nx.barabasi_albert_graph(DENSE_NODES, 25,seed=SEED),
#   'SW': nx.watts_strogatz_graph(DENSE_NODES, 25, 0.3, seed=SEED)
# }

dist_funcs = {
  'uniform': lambda: np.random.uniform(0.0, 1.0),
  'normal': lambda: np.clip(np.random.normal(0.5, 0.2), 0, 1),
  'beta': lambda: np.random.beta(2, 5),
}