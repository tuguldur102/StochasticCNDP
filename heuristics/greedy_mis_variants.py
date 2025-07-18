import numpy as np
from typing import Tuple, Set, Union, List, Dict
import heapq
from numba import njit
from .utils import nx_to_csr, epc_mc, epc_mc_deleted, local_search
import networkx as nx

@njit
def greedy_with_mis_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    probs: np.ndarray,
    deleted: np.ndarray,
    n: int,
    k: int,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    target_survivors = n - k
    survivors = n - np.sum(deleted)

    flips_needed = target_survivors - survivors
    if flips_needed < 0:
        flips_needed = 0

    max_steps = 1 + flips_needed
    trace_np = np.empty(max_steps, dtype=np.float64)

    step = 0
    curr_epc = epc_mc(indptr, indices, probs, deleted, num_samples)
    trace_np[step] = curr_epc
    step += 1

    while survivors < target_survivors:
        best_sigma = 1e18
        best_j = -1

        for j in range(n):
            if deleted[j]:
                deleted[j] = False
                sigma = epc_mc(indptr, indices, probs, deleted, num_samples)
                deleted[j] = True
                if sigma < best_sigma:
                    best_sigma = sigma
                    best_j = j

        deleted[best_j] = False
        survivors += 1
        curr_epc = best_sigma

        trace_np[step] = curr_epc
        step += 1
    return deleted, trace_np, step

def greedy_mis_optimized(
  G: nx.Graph,
  k: int,
  num_trails: int = 20,
  num_samples: int = 100_000,
  return_trace: bool = False,
) -> Union[Set[int], Tuple[Set[int], list]]:
    
    # CSR conversion
    nodes, idx_of, indptr, indices, probs = nx_to_csr(G)
    n = len(nodes)

    best_deleted = None
    best_sigma = float("inf")

    for _ in range(num_trails):
        MIS = nx.maximal_independent_set(G)

        # deleted[i]==True means node i is removed
        deleted = np.ones(n, dtype=np.bool_)
        
        for u in MIS:
            deleted[idx_of[u]] = False
        
        curr_sigma = epc_mc(indptr, indices, probs, deleted, num_samples)


        if curr_sigma < best_sigma:
            best_sigma = curr_sigma
            best_deleted = deleted.copy()

    # MIS = nx.algorithms.approximation.maximum_independent_set(G)

    # deleted = np.ones(n, dtype=np.bool_)
    # for u in MIS:
    #     deleted[idx_of[u]] = False

    # best_deleted = deleted.copy()

    # Call the fast Numba core
    final_deleted, trace_np, cnt = greedy_with_mis_numba(
        indptr, indices, probs, 
        best_deleted.copy(), 
        n, k, num_samples
    )

    trace = trace_np[:cnt].tolist()

    D = {nodes[i] for i in range(n) if final_deleted[i]}

    return (D, trace) if return_trace else D

def robust_greedy_mis_optimized(
  G, k, 
  num_samples=10_000,
  ):  

  S = greedy_mis_optimized(
    G, k,
    num_samples=num_samples,
    return_trace=False)
  
  S_opt = local_search(G.copy(), S, num_samples=num_samples)

  return S_opt

def greedy_with_mis(
  G: nx.Graph,
  k: int,
  num_trails: int = 20,
  num_samples: int = 10_000,
) -> Tuple[Set, List]:
  """ Greedy algorithm for finding a set of nodes with minimal 
  expected pairwise connectivity (EPC) using 
  a maximal independent set (MIS) as a starting point."""

  best_sigma = float("inf")

  for _ in range(num_trails):
      MIS = nx.maximal_independent_set(G)
      
      curr_sigma = epc_mc_deleted(G, S=MIS, num_samples=num_samples)

      # print(f"{i}-th rount sigma: {curr_sigma}")

      if curr_sigma < best_sigma:
          best_sigma = curr_sigma

  target = len(G) - k
  R = set(MIS)

  sigma_delta = []

  # print(f"#MIS: {len(R)}")

  # Greedy grow R set until |R| = |V| - k
  while len(R) < target:
    best_j, best_sigma = None, float('inf')
    for j in G.nodes():
      if j in R:
        continue

      # delete node
      S_j = set(G.nodes()) - (R | {j})
      sigma = epc_mc_deleted(G, S=S_j, num_samples=num_samples)

      if sigma < best_sigma:
        best_sigma, best_j = sigma, j

        sigma_delta.append(best_sigma)

    R.add(best_j)
  
  D = set(G.nodes()) - R

  return D, sigma_delta

def greedy_with_mis_local_search(
    G: nx.Graph,
    k: int,
    num_trails: int = 20,
    num_samples: int = 10_000,
):
  S = greedy_with_mis(G.copy(), k, num_trails, num_samples)

  S_opt = local_search(G.copy(), S, num_samples=num_samples)

  return S_opt

def greedy_epc_mis_celf(
    G: nx.Graph,
    k: int,
    num_trials: int = 20,
    num_samples: int = 10_000,
) -> Tuple[Set[int], List[float]]:
    """
    CELF‑accelerated version of greedy_epc_mis."""

    best_sigma = float("inf")
    best_MIS   = None

    for _ in range(num_trials):

        MIS = nx.maximal_independent_set(G)
        sigma = epc_mc_deleted(
           G.copy(), S=set(G.nodes()) - set(MIS),
            num_samples=num_samples)
        
        if sigma < best_sigma:
            best_sigma, best_MIS = sigma, MIS

    R          = set(best_MIS)                  
    sigma_curr = best_sigma                   
    target     = len(G) - k                   
    sigma_hist = [sigma_curr]

    # CELF priority queue
    pq: list[Tuple[float, int, int]] = []
    round_id = 0

    for v in G.nodes():
        if v in R:
            continue

        gain = sigma_curr - epc_mc_deleted(
           G.copy(), S=set(G.nodes()) - (R | {v}),
            num_samples=num_samples)
        
        heapq.heappush(pq, (-gain, v, round_id)) 

    while len(R) < target:

        neg_gain, v, last_eval = heapq.heappop(pq)

        if last_eval == round_id:
            R.add(v)
            sigma_curr += neg_gain              
            sigma_hist.append(sigma_curr)
            round_id += 1
        else:
            gain = sigma_curr - epc_mc_deleted(
                G.copy(), S=set(G.nodes()) - (R | {v}), 
                num_samples=num_samples
            )

            heapq.heappush(pq, (-gain, v, round_id))

    D = set(G.nodes()) - R
    
    return D, sigma_hist