# === Standard Library ===
import time

# === Third-Party Libraries ===

# --- Scientific Computing ---
import numpy as np
import pandas as pd

# --- Plotting ---
import matplotlib.pyplot as plt

from tqdm import tqdm

# --- Graph Processing ---
import networkx as nx


from greedy_es_variants import greedy_empty_set_celf, greedy_empty_set_celf_local_search
from utils import local_search, epc_mc_deleted
from graph_centrality import remove_k_betweenness, remove_k_degree_centrality, remove_k_pagerank_nodes
from greedy_mis_variants import greedy_with_mis, greedy_with_mis_local_search
from grasp import grasp_cndp, grasp_with_local_search_outside, grasp_meta

SEED : int = 42

N_SAMPLE_EVAL = 100_000
N_SAMPLE_LS = 10_000

LOCAL_SEARCH_ITER = 1
GRASP_RESTARTS = 3

NODES = 100
K = int(NODES * 0.1)

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


for name_model, G in tqdm(
  graph_models.items(), 
  desc="Processing models", 
  total=len(graph_models)):

  records = []

  # for p in tqdm(np.arange(0.0, 1.1, 0.1), desc="Processing", total=int(1.1/0.1)):

  #   def fresh_graph():
  #     H = G.copy()
  #     for u, v in H.edges():
  #       H[u][v]['p'] = p
  #     return H
  
  # dist functions
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


    # Heuristics 1: Degree-Based Centrality
    t0 = time.perf_counter()
    degree_S  = remove_k_degree_centrality(fresh_graph(), K)

    # print(set(degree_S))
    epc_degree = epc_mc_deleted(fresh_graph(), set(degree_S), N_SAMPLE_EVAL)
    t_degree  = time.perf_counter() - t0


    # Heuristics 2: Betweenness
    t0 = time.perf_counter()
    between_S  = remove_k_betweenness(fresh_graph(), K)

    epc_between = epc_mc_deleted(fresh_graph(), set(between_S), N_SAMPLE_EVAL)

    t_between  = time.perf_counter() - t0

    # Heuristics 3: PageRank node
    t0 = time.perf_counter()
    pagerank_S  = remove_k_pagerank_nodes(fresh_graph(), K)

    epc_pagerank = epc_mc_deleted(fresh_graph(), set(pagerank_S), N_SAMPLE_EVAL)

    t_pagerank  = time.perf_counter() - t0

    # heuristics 4: Greedy ES optimized
    t0 = time.perf_counter()

    t_greedy_es, epc_greedy_es, greedy_es_S_ls = greedy_es_local_opt(
      fresh_graph(), K, num_samples=N_SAMPLE_LS,
      local_iter=LOCAL_SEARCH_ITER)
    
    epc_greedy_es_ls = epc_mc_deleted(fresh_graph(), greedy_es_S_ls, N_SAMPLE_EVAL)
    
    t_greedy_es_final = time.perf_counter() - t0

    # heuristics 5: Greedy MIS optimized
    t0 = time.perf_counter()

    t_greedy_mis_initial, mis_epc_initial, mis_epc_init_std, mis_epc_final, mis_epc_final_std = robust_greedy_mis_optimized(
      fresh_graph(), K, num_samples=N_SAMPLE_LS,
      trials=5, max_iter=LOCAL_SEARCH_ITER)
    
    t_greedy_mis_final = time.perf_counter() - t0

    # heuristics 6: REGA
    t0 = time.perf_counter()

    rega_D = rega(
      fresh_graph(),
      k=K,
      epc_func=epc_mc_deleted,
      num_samples=N_SAMPLE_LS,
      max_iter=LOCAL_SEARCH_ITER,
      use_tqdm=False)
    
    rega_epc = epc_mc_deleted(fresh_graph(), rega_D, N_SAMPLE_EVAL)
    t_rega = time.perf_counter() - t0

    # print(f"\n ---- REGA!!! ---- \n")
    # heuristics 7: Grasp + ls
    t0 = time.perf_counter()

    S_star_outside, epc_outside = grasp_with_local_search_outside(
          fresh_graph(), K=K,
          alpha=0.05,
          mc_samples_grasp=N_SAMPLE_LS,
          mc_samples_ls=N_SAMPLE_LS,
          restarts=GRASP_RESTARTS,
          max_ls_iter=LOCAL_SEARCH_ITER
          )
    
    epc_grasp_ls = epc_mc_deleted(fresh_graph(), S_star_outside, N_SAMPLE_EVAL)

    t_grasp_ls = time.perf_counter() - t0

    # print(f"\n ---- GRASP!!! ---- \n")

    # heuristics 8: Grasp + ls + path relinking + reactive alpha
    t0 = time.perf_counter()

    grasp_opt_S, grasp_opt_epc = grasp_meta(
      fresh_graph(), K, 
      restarts=GRASP_RESTARTS, mc_samples_grasp = N_SAMPLE_LS, 
      mc_samples_final=N_SAMPLE_LS, mc_samples_ls=N_SAMPLE_LS,
      max_ls_iter=LOCAL_SEARCH_ITER, elite_size=5)
    
    t_grasp_opt = time.perf_counter() - t0

    # print(f"\n ---- grasp ls!!! ---- \n")

    # print(f"\nGreedy ES init: {epc_greedy_es} vs {epc_greedy_es_ls}\n")
    # print(f"\nGreedy MIS init: {mis_epc_initial} vs {mis_epc_final}\n")

    for algo, t, epc, std in [
      ('Degree-based', t_degree, epc_degree, 0.0),
      ('Betweenness', t_between, epc_between, 0.0),
      ('PageRank', t_pagerank, epc_pagerank, 0.0),

      ('Greedy_ES_initial', t_greedy_es, epc_greedy_es, 0.0),
      ('Greedy_ES_final', t_greedy_es_final, epc_greedy_es_ls, 0.0),
      ('Greedy_MIS_initial', t_greedy_mis_initial, mis_epc_initial, mis_epc_init_std),
      ('Greedy_MIS_final', t_greedy_mis_final, mis_epc_final, mis_epc_final_std),

      ('REGA', t_rega, rega_epc, 0.0),
      ('grasp', t_grasp_ls, epc_grasp_ls, 0.0),
      ('grasp_path_relink', t_grasp_opt, grasp_opt_epc, 0.0),
    ]:
      
      records.append({
        'model': name_model,
        'dist': name_dist,
        'algo': algo,
        'time': t,
        'epc': epc,
        'epc_std': std,
      })

    # SAVE_PATH_ROOT = r"C:\Users\btugu\Documents\develop\research\SCNDP\src\extension\heuristics\results"

    # df = pd.DataFrame(records)
    # df.to_csv(f"{SAVE_PATH_ROOT}/csv/sparse/Result_heuristics_{name_model}_{NODES}_{K}_all_ls_.csv", index=False)

    SAVE_PATH_ROOT = "/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/extension/heuristics/results"

    df = pd.DataFrame(records)
    df.to_csv(f"{SAVE_PATH_ROOT}/csv/dist_job/Result_heuristics_{name_model}_{NODES}_{K}_dist.csv", index=False)

    # SAVE_PATH_ROOT = r"C:\Users\btugu\Documents\develop\research\SCNDP\src\extension\heuristics\results"

    # df = pd.DataFrame(records)
    # df.to_csv(f"{SAVE_PATH_ROOT}/csv/dist/Result_heuristics_{name_model}_{NODES}_{K}_all_DIST_FUNC.csv", index=False)