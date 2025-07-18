from torch_geometric.loader import DataLoader
import torch
from model import SAGEEdgeProbModel
from dataloader import GraphEPCDataset
import matplotlib.pylab as pd
from ..heuristics.utils import epc_mc_deleted
import time, pickle
from tqdm import tqdm

ROOT = "/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/extension/learning/notebooks/dataset/gnn/final_test"

NODES_SIZE = 500
K = int(NODES_SIZE / 10)
AGGR = 'mean'

graphs_dir   = f"{ROOT}/dist"        
# labels_dir   = f"{ROOT}/regression_labels/test/test100"          # not needed for inference but Dataset expects it
CHKPNT_ROOT = "/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/extension/learning/notebooks/gnn/checkpoints"

ckpt_path    = f"{CHKPNT_ROOT}/best_model_cla_30_diff.pt" # saved in train_model()
mc_samples   = 100_000                 # per-graph Monte-Carlo samples for EPC
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_ds      = GraphEPCDataset(graphs_dir, None, split=f"{NODES_SIZE}")
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

in_dim       = 11                    
model = SAGEEdgeProbModel(
        in_dim=11, hidden_dim=256, heads=8, dropout=0.4, aggr=AGGR).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

all_epc = []
i = 0

data_lst = {
    'model': [],
    'dist_func': [],
    'algo': [],
    'time': [],
    'epc': [],
}

for data in tqdm(test_loader, desc="Inference"):

    data = data.to(device)
    fname = data.file_name[0]
    
    t0 = time.perf_counter()

    with torch.no_grad():
        scores = model(data.x, data.edge_index, data.edge_prob) 

    topk = scores.topk(K, largest=True).indices.tolist()        
    idx  = data.idx.item()     

    G_nx = pickle.load(open(test_ds.graph_paths[idx], 'rb'))['graph']
    epc_del = epc_mc_deleted(G_nx.copy(), set(topk), num_samples=mc_samples)

    t_gnn = time.perf_counter() - t0

    epc_0   = epc_mc_deleted(G_nx.copy(), set(), num_samples=mc_samples)  
    all_epc.append(epc_del)
    delta   = epc_del - epc_0    

    parts = fname.split("_")        

    data_lst['model'].append(fname[0:2])

    # for part in parts:
    #     if part.startswith("rp"):
    #         p_value = float(part[2:])
    #         data_lst['dist_func'].append(p_value)
    #         break
    
    data_lst['dist_func'].append(parts[3])

    data_lst['algo'].append("greedy_gnn")
    data_lst['time'].append(t_gnn)
    data_lst['epc'].append(epc_del)

    assert all(v in G_nx for v in topk)

SAVE_ROOT = f"/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/extension/learning/notebooks/dataset/gnn/final_test"
df = pd.DataFrame(data_lst)
# model_order = ['ER', 'BA', 'SW']

# df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
# df_sorted = df.sort_values(by=['model', 'p'])

model_order = ['ER', 'BA', 'SW']
dist_order = ['uniform', 'normal', 'beta']

# Convert to ordered categorical columns
df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
df['dist_func'] = pd.Categorical(df['dist_func'], categories=dist_order, ordered=True)

# Sort by both columns
df_sorted = df.sort_values(['model', 'dist_func'])

df_sorted.to_csv(f"{SAVE_ROOT}/csv/{NODES_SIZE}_gnn_DIST.csv", index=False)

import numpy as np
print(f"\nAverage EPC over {len(all_epc)} test graphs: {np.mean(all_epc):.4f}")