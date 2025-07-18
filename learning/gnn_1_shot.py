from torch_geometric.loader import DataLoader
import torch
from .model import SAGEEdgeProbModel
from .dataloader import GraphEPCDataset
import matplotlib.pylab as pd
from tqdm import tqdm
import networkx as nx
from .utils import load_single_graph_as_data
from typing import Any

def predict(
  model: Any,
  G: nx.Graph,
  K: int,
  device: Any,
  ):  

  data = load_single_graph_as_data(G.copy()).to(device)

  with torch.no_grad():
    scores = model(data.x, data.edge_index, data.edge_prob)

  topk_nodes = scores.topk(K, largest=True).indices.tolist()
  
  return set(topk_nodes)

  
