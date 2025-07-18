import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

class EdgeProbGATConv(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 heads: int = 2,
                 negative_slope: float = 0.2,
                 dropout: float = 0.2,
                 concat: bool = True,
                 bias: bool = True):
        super().__init__(aggr='add', node_dim=0)  # standard GAT aggregation
        
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.heads         = heads
        self.negative_slope= negative_slope
        self.dropout       = dropout
        self.concat        = concat

        # Linear projection for query/key/value
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        # Attention weights aᵀ [Wh_i || Wh_j]
        self.att = nn.Parameter(torch.Tensor(1, heads, 2*out_channels + 1))
        # self.att = nn.Parameter(torch.Tensor(1, heads, 2*out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_prob: torch.Tensor):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        edge_prob: [E]   (the p_ij for each edge in edge_index order)
        """
        N = x.size(0)
        # 1. Linearly project node features to multi-head space
        x = self.lin(x)                              # [N, heads*out]
        x = x.view(N, self.heads, self.out_channels) # [N, heads, out]

        # 2. Start propagation
        out = self.propagate(edge_index, x=x, edge_prob=edge_prob, size=(N, N))
        # out: [N, heads, out]

        # 3. Concat or average heads
        if self.concat:
            out = out.view(N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)  # [N, out]

        if self.bias is not None:
            out = out + self.bias

        return out

    # def message(self,
    #             x_j: torch.Tensor,
    #             x_i: torch.Tensor,
    #             edge_prob: torch.Tensor,
    #             index: torch.LongTensor,
    #             ptr,
    #             size_i):
    #     """
    #     x_j, x_i: [E, heads, out_channels] (sender and receiver node reps)
    #     edge_prob: [E]           (scalar reliability)
    #     index:   [E]             (destination node indices)
    #     """
    #     # 1. compute standard attention logits: aᵀ [Wh_i || Wh_j]
    #     cat = torch.cat([x_i, x_j], dim=-1)           # [E, heads, 2*out]
    #     alpha = (cat * self.att).sum(dim=-1)          # [E, heads]

    #     # 2. add log(edge_prob)
    #     log_p = edge_prob.log().unsqueeze(-1)        # [E, 1]
    #     alpha = alpha + log_p                        # broadcasting to [E, heads]

    #     # 3. leaky‐relu + softmax over all incoming edges
    #     alpha = F.leaky_relu(alpha, self.negative_slope)
    #     alpha = softmax(alpha, index, ptr, size_i)    # [E, heads]

    #     # 4. dropout on attention weights
    #     alpha = F.dropout(alpha, p=self.dropout, training=self.training)

    #     # 5. scale messages
    #     return x_j * alpha.unsqueeze(-1)             # [E, heads, out]

    def message(self, x_j, x_i, edge_prob, index, ptr, size_i):
        # concat node reps and edge scalar
        edge_prob = edge_prob.view(-1, 1, 1)               # [E,1,1]
        cat = torch.cat([x_i, x_j, edge_prob.expand(-1, self.heads, 1)], dim=-1)
        # shape: [E, heads, 2*out+1]

        alpha = (cat * self.att).sum(dim=-1)               # [E, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        # aggr_out: [N, heads, out] if concat else [N, out]
        return aggr_out