import random
from typing import Tuple, Set, Dict, List
import networkx as nx
from tqdm import tqdm

from .utils import epc_mc_deleted, nx_to_csr, local_search

def grasp_cndp(
  G: nx.Graph,
  K: int,
  alpha: float = 0.1,
  num_samples: int = 10_000,
  restarts: int = 3,
  use_tqdm: bool = False
  ) -> Tuple[Set[int], float]:
  """
  GRASP for Stochastic CNDP:
  """
  best_S, best_score = None, float('inf')

  if use_tqdm:
    it = tqdm(range(restarts), desc="Processing GRASP", total=restarts)
  else:
    it = range(restarts)

  for _ in it:
    S = set()
    # precompute sigma(empty)
    sigma_S = epc_mc_deleted(G, S, num_samples)

    for k in range(K):
      # compute improvement d_j = sigma(S) – sigma(S ∪ {j})
      improvements = {}
      for j in G.nodes():
        if j in S: 
          continue
        sigma_Sj = epc_mc_deleted(G, S | {j}, num_samples)
        improvements[j] = sigma_S - sigma_Sj

      # find best and worst d
      max_imp = max(improvements.values())
      min_imp = min(improvements.values())

      # build RCL = { j : d_j >= max_imp – alpha*(max_imp – min_imp) }
      threshold = max_imp - alpha * (max_imp - min_imp)
      RCL = [j for j, d in improvements.items() if d >= threshold]

      # pick one at random from RCL
      v = random.choice(RCL)
      S.add(v)

      # update sigma(S)
      sigma_S = epc_mc_deleted(G, S, num_samples)

    if sigma_S < best_score:
      best_score = sigma_S
      best_S = S.copy()

  return best_S, best_score

def grasp_with_local_search_outside(
    G: nx.Graph,
    K: int,
    alpha: float = 0.2,
    mc_samples_grasp: int = 10000,
    mc_samples_ls: int = 10000,
    restarts: int = 30
) -> Tuple[Set[int], float]:
    """
    Combined GRASP + local_search_swap procedure.
    """

    # best_inner_S, best_inner_score = set(), float('inf')
    best_S, best_score = set(), float('inf')

    S_grasp, _ = grasp_cndp(
        G.copy(), K, num_samples=mc_samples_grasp, 
        alpha=alpha, restarts=restarts, use_tqdm=False)

    S_opt = local_search(G.copy(), S_grasp, mc_samples_ls)

    return S_opt

class ReactiveAlpha:
    def __init__(self, alpha_vals: List[float]):
        self.alpha_vals = alpha_vals
        self.weights = [1.0] * len(alpha_vals)

    def sample(self) -> Tuple[int, float]:
        total = sum(self.weights)
        r = random.random() * total
        cum = 0.0
        for i, w in enumerate(self.weights):
            cum += w
            if r <= cum:
                return i, self.alpha_vals[i]
        return len(self.weights)-1, self.alpha_vals[-1]

    def reward(self, idx: int, amount: float = 1.0):
        self.weights[idx] += amount

    def penalize(self, idx: int, factor: float = 0.99):
        self.weights[idx] *= factor

def grasp_construct(G: nx.Graph,
                    K: int,
                    alpha: float,
                    mc_samples: int) -> Tuple[Set[int], float]:
    """One GRASP construction (no restarts)."""

    S: Set[int] = set()
    cache: Dict[frozenset, float] = {}

    # def sigma(SetS: Set[int]) -> float:
        
    #     key = frozenset(SetS)
    #     if key not in cache:
    #         cache[key] = epc_mc_deleted(G, SetS, num_samples=mc_samples)
    #     return cache[key]

    sigma_S = epc_mc_deleted(G.copy(), S, num_samples=mc_samples)

    for _ in range(K):
        gains = {}

        for v in G.nodes():
            if v in S:
                continue
            
            gains[v] = sigma_S - epc_mc_deleted(G.copy(), S | {v}, num_samples=mc_samples)

        d_max, d_min = max(gains.values()), min(gains.values())
        thresh = d_max - alpha * (d_max - d_min)

        RCL = [v for v, d in gains.items() if d >= thresh]

        choice = random.choice(RCL)
        S.add(choice)

        sigma_S = epc_mc_deleted(G.copy(), S, num_samples=mc_samples)

    return S, sigma_S

def path_relink(S: Set[int], E: Set[int],
                G: nx.Graph,
                mc_samples: int) -> Tuple[Set[int], float]:
    """Greedy walk from S toward E, returning best intermediate."""

    T = S.copy()
    best_T, best_score = T.copy(), epc_mc_deleted(G.copy(), T, mc_samples)

    D_add = list(E - T)
    D_rm  = list(T - E)

    while D_add and D_rm:
        
        best_move = None
        best_delta = 0.0

        for i in D_rm:
            for j in D_add:
                
                T_candidate = T.copy()
                T_candidate.remove(i)
                T_candidate.add(j)

                score = epc_mc_deleted(G.copy(), T_candidate, mc_samples)
                delta = best_score - score

                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, j, score)
        if best_move is None:
            break
        
        i, j, new_score = best_move
        T.remove(i); T.add(j)
        D_rm.remove(i); D_add.remove(j)

        if new_score < best_score:
            best_score = new_score
            best_T = T.copy()

    return best_T, best_score

def insert_into_elite(
  elite: List[Tuple[Set[int], float]],
  candidate: Tuple[Set[int], float],
  max_size: int = 10
  ):
    
    elite.append(candidate)
    elite.sort(key=lambda x: x[1])

    if len(elite) > max_size:
        elite.pop()

def grasp_meta(
    G: nx.Graph,
    K: int,
    restarts: int = 30,
    mc_samples: int = 10_000,
    elite_size: int = 5,
) -> Tuple[Set[int], float]:
    
    reactive = ReactiveAlpha([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9])
    elite: List[Tuple[Set[int], float]] = []

    for _ in range(restarts):
        idx_alpha, alpha = reactive.sample()

        # build a solution with greedy random adaptive construction
        S_raw, _ = grasp_construct(G.copy(), K, alpha, mc_samples)

        # evaluate the raw solution with a medium sample budget
        score_raw = epc_mc_deleted(G.copy(), S_raw, num_samples=mc_samples)

        # maintain the elite list
        insert_into_elite(elite, (S_raw, score_raw), max_size=elite_size)

        # reward or penalise that alpha using the raw score
        best_elite_score = elite[0][1]
        
        if score_raw <= best_elite_score:
            reactive.reward(idx_alpha)
        else:
            reactive.penalize(idx_alpha)

    best_raw_S, _ = elite[0]

    return best_raw_S
