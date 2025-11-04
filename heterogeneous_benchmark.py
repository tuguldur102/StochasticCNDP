import time
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import random

from heuristics.greedy_mis_variants import greedy_with_mis, greedy_epc_mis_celf
from heuristics.rega import rega  
from heuristics.utils import local_search, epc_mc_deleted

SEED : int = 42

N_SAMPLE_EVAL = 100_000
N_SAMPLE_LS = 10_000

LOCAL_SEARCH_ITER = 1
NUM_MIS_TRAILS = 30

NODES = 100
K = int(NODES * 0.1)

np.random.seed(SEED)
random.seed(SEED)

# nodes 100, edges 200 (Sparse Graphs)
graph_models = {
  'ER': nx.erdos_renyi_graph(NODES, 0.0443, seed=SEED),
  'BA': nx.barabasi_albert_graph(NODES, 2,seed=SEED),
  'SW': nx.watts_strogatz_graph(NODES, 4, 0.3, seed=SEED)
}

dist_funcs = {
  'uniform': lambda: np.random.uniform(0.0, 1.0),
  'normal': lambda: np.clip(np.random.normal(0.5, 0.2), 0, 1),
  'beta': lambda: np.random.beta(2, 5),
}

for name_model, G in tqdm(
  graph_models.items(), 
  desc="Processing models", 
  total=len(graph_models)):
  records = []  

  for name_dist, dist_func in tqdm(
    dist_funcs.items(),
    desc="Processing",
    total=len(dist_funcs)):

    def fresh_graph():
      """Fresh graph with new edge weights."""
      H = G.copy()
      for u, v in H.edges():
        H[u][v]['p'] = dist_func()
      return H

    # heuristics 1: Greedy MIS CELF
    t0 = time.perf_counter()

    greedy_mis_S, _ = greedy_with_mis(
      fresh_graph(), K, num_trails=NUM_MIS_TRAILS, num_samples=N_SAMPLE_LS)
    epc_greedy_mis = epc_mc_deleted(fresh_graph(), greedy_mis_S, N_SAMPLE_EVAL)
    t_greedy_mis = time.perf_counter() - t0

    t0 = time.perf_counter()

    greedy_mis_S, _ = greedy_with_mis(
      fresh_graph(), K, num_trails=NUM_MIS_TRAILS, num_samples=N_SAMPLE_LS)
    greedy_mis_S_ls = local_search(fresh_graph(), greedy_mis_S, num_samples=N_SAMPLE_LS)
    epc_greedy_mis_ls = epc_mc_deleted(fresh_graph(), greedy_mis_S_ls, N_SAMPLE_EVAL)
    t_greedy_mis_ls = time.perf_counter() - t0

    # heuristics 2: REGA
    t0 = time.perf_counter()

    rega_S = rega(
      fresh_graph(),
      k=K,
      num_samples=N_SAMPLE_LS)
    epc_rega = epc_mc_deleted(fresh_graph(), rega_S, N_SAMPLE_EVAL)
    t_rega = time.perf_counter() - t0

    t0 = time.perf_counter()
    rega_S = rega(
      fresh_graph(),
      k=K,
      num_samples=N_SAMPLE_LS)
    rega_S_ls = local_search(fresh_graph(), rega_S, N_SAMPLE_LS)
    epc_rega_ls = epc_mc_deleted(fresh_graph(), rega_S_ls, N_SAMPLE_EVAL)
    t_rega_ls = time.perf_counter() - t0

    for algo, t, epc in [
      ('Greedy with MIS', t_greedy_mis, epc_greedy_mis),
      ('Greedy with MIS + Local Search', t_greedy_mis_ls, epc_greedy_mis_ls),
      
      ('REGA', t_rega, epc_rega),
      ('REGA + Local Search', t_rega_ls, epc_rega_ls),
    ]:
      
      records.append({
        'model': name_model,
        'name_dist': name_dist,
        'algo': algo,
        'time': t,
        'epc': epc
      })

    df = pd.DataFrame(records)
    df.to_csv(f"Results_{name_model}_{NODES}_{K}_all_DIST.csv", index=False)

