"""
gnn_encoder.py
--------------
Heterogeneous Graph Transformer (HGT) encoder for the UTCP problem.

Architecture (per blueprint):
  - Model: HGTConv (Hu et al., 2020)
  - Layers: 3
  - Hidden dim: 128
  - Output dim: 64 per node type
  - Supports 4 node types: course, faculty, room, timeslot
  - Supports all 5 edge types

Output: per-node embeddings (dict[str, Tensor]) for downstream RL policy.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear


# Node feature dimensions must match graph_builder.py
NODE_FEAT_DIMS = {
    "course":   6,   # [is_lab, enrol_norm, hours_norm, medium_enc×3]
    "faculty":  4,   # [medium_enc×3, max_hrs_norm]
    "room":     4,   # [cap_norm, type_enc×3]
    "timeslot": 9,   # [day_enc×6, period_norm, is_morning, is_last_period]
}

HIDDEN_DIM  = 128
OUT_DIM     = 64
NUM_LAYERS  = 3
NUM_HEADS   = 4   # attention heads for HGTConv


class HGTEncoder(nn.Module):
    """
    Multi-layer Heterogeneous Graph Transformer encoder.

    Produces node-level embeddings of dimension OUT_DIM for every node type.
    Used as the shared backbone by both actor and critic in the PPO policy.
    """

    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int = HIDDEN_DIM,
        out_dim: int = OUT_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.num_layers = num_layers

        # Input projections: raw features → hidden_dim for each node type
        self.input_proj = nn.ModuleDict({
            ntype: Linear(NODE_FEAT_DIMS[ntype], hidden_dim, bias=True)
            for ntype in node_types
        })

        # Stack of HGTConv layers
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(node_types, edge_types),
                heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        # Output projections: hidden_dim → out_dim
        self.output_proj = nn.ModuleDict({
            ntype: Linear(hidden_dim, out_dim, bias=True)
            for ntype in node_types
        })

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout    = nn.Dropout(p=0.1)

    def forward(self, data: HeteroData) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        data : HeteroData  (from graph_builder.build_hetero_graph)

        Returns
        -------
        embeddings : dict[str, Tensor]
            {node_type: Tensor of shape (N_nodes, out_dim)}
        """
        # Step 1: project raw features to hidden_dim
        x_dict: dict[str, Tensor] = {}
        for ntype in self.node_types:
            if hasattr(data[ntype], "x") and data[ntype].x is not None:
                x_dict[ntype] = self.activation(self.input_proj[ntype](data[ntype].x))
            else:
                # Fallback: zero tensor (handles missing node types gracefully)
                n = data[ntype].num_nodes if hasattr(data[ntype], "num_nodes") else 1
                x_dict[ntype] = torch.zeros(n, self.hidden_dim)

        # Step 2: edge_index_dict
        edge_index_dict = {
            etype: data[etype].edge_index
            for etype in self.edge_types
            if etype in data.edge_types and data[etype].edge_index.shape[1] > 0
        }

        # Step 3: message passing layers
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            # Residual connection + dropout
            x_dict = {
                ntype: self.dropout(self.activation(x_dict_new[ntype]))
                        + x_dict[ntype]  # residual
                for ntype in self.node_types
                if ntype in x_dict_new
            }

        # Step 4: output projection
        embeddings = {
            ntype: self.output_proj[ntype](x_dict[ntype])
            for ntype in self.node_types
            if ntype in x_dict
        }
        return embeddings


def build_encoder_from_data(data: HeteroData) -> HGTEncoder:
    """Convenience constructor: builds encoder matching a given HeteroData."""
    return HGTEncoder(
        node_types=list(data.node_types),
        edge_types=list(data.edge_types),
    )


if __name__ == "__main__":
    from graph_builder import build_hetero_graph
    import json, pathlib

    sample = pathlib.Path("data/indian_synthetic/instance.json")
    if sample.exists():
        with open(sample) as f:
            inst = json.load(f)
        data = build_hetero_graph(inst)
        encoder = build_encoder_from_data(data)
        print(encoder)
        embs = encoder(data)
        for k, v in embs.items():
            print(f"  {k}: {v.shape}")
    else:
        print("Run generate_indian_data.py first.")
