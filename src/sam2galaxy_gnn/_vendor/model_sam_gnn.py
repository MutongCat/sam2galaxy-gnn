from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import LayerNorm, SAGEConv


class MoEPointGaussianHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_components: int,
        num_continuous: int,
        hidden_dim: int,
        hidden_layers: int = 1,
        gate_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_components = int(num_components)
        self.num_continuous = int(num_continuous)
        self.dropout = float(dropout)

        gate_h = int(gate_hidden_dim) if gate_hidden_dim is not None else int(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, gate_h),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(gate_h, self.num_components),
        )

        layers = []
        d = in_dim
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(d, int(hidden_dim)))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity())
            d = int(hidden_dim)
        layers.append(nn.Linear(d, 2 * self.num_continuous))
        self.param_head = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor):
        mix_logits = self.gate(h)
        params = self.param_head(h)
        mu, log_sigma = params.split(self.num_continuous, dim=-1)
        return mix_logits, mu, log_sigma


class SAMGalaxyGNN_MultiZ(nn.Module):
    def __init__(
        self,
        in_dim,
        u_dim,
        hidden_dim,
        out_dim_per_z=5,
        num_z=9,
        num_layers=4,
        dropout=0.2,
        use_moe=True,
        mix_hidden_dim=128,
        mix_hidden_layers=2,
        mix_dropout=0.2,
        ssfr_mix_components=3,
        ssfr_mix_continuous=2,
        gas_mix_components=4,
        gas_mix_continuous=3,
        reg_head_type="point",
        reg_sigma_min=-7.0,
        reg_sigma_max=7.0,
    ):
        super().__init__()
        self.num_z = int(num_z)
        self.out_dim_per_z = int(out_dim_per_z)
        self.dropout = float(dropout)
        self.use_moe = bool(use_moe)
        self.mix_hidden_dim = int(mix_hidden_dim)
        self.mix_hidden_layers = int(mix_hidden_layers)
        self.mix_dropout = float(mix_dropout)
        self.ssfr_mix_components = int(ssfr_mix_components)
        self.ssfr_mix_continuous = int(ssfr_mix_continuous)
        self.gas_mix_components = int(gas_mix_components)
        self.gas_mix_continuous = int(gas_mix_continuous)
        self.reg_head_type = str(reg_head_type).lower()
        self.reg_sigma_min = float(reg_sigma_min)
        self.reg_sigma_max = float(reg_sigma_max)

        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden_dim)])
        self.norms = nn.ModuleList([LayerNorm(hidden_dim)])
        for _ in range(int(num_layers) - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))

        self.lin_u = nn.Linear(u_dim, hidden_dim)
        self.lin_1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.redshift_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, out_dim_per_z),
                )
                for _ in range(self.num_z)
            ]
        )
        if self.reg_head_type == "hetero":
            self.redshift_sigma_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim // 2, out_dim_per_z),
                    )
                    for _ in range(self.num_z)
                ]
            )
        else:
            self.redshift_sigma_heads = None

        if self.use_moe:
            self.ssfr_mix_heads = nn.ModuleList(
                [
                    MoEPointGaussianHead(
                        in_dim=hidden_dim,
                        num_components=self.ssfr_mix_components,
                        num_continuous=self.ssfr_mix_continuous,
                        hidden_dim=self.mix_hidden_dim,
                        hidden_layers=self.mix_hidden_layers,
                        gate_hidden_dim=self.mix_hidden_dim,
                        dropout=self.mix_dropout,
                    )
                    for _ in range(self.num_z)
                ]
            )
            self.gas_mix_heads = nn.ModuleList(
                [
                    MoEPointGaussianHead(
                        in_dim=hidden_dim,
                        num_components=self.gas_mix_components,
                        num_continuous=self.gas_mix_continuous,
                        hidden_dim=self.mix_hidden_dim,
                        hidden_layers=self.mix_hidden_layers,
                        gate_hidden_dim=self.mix_hidden_dim,
                        dropout=self.mix_dropout,
                    )
                    for _ in range(self.num_z)
                ]
            )
        else:
            self.ssfr_mix_heads = None
            self.gas_mix_heads = None

    def forward(self, x, edge_index, u, batch, z_indices):
        device = x.device
        if hasattr(batch, "batch"):
            if u.dim() == 2 and u.size(0) == 1:
                u_expanded = u[0].unsqueeze(0).repeat(x.size(0), 1).to(device)
            elif u.dim() == 2:
                u_expanded = u[batch.batch].to(device)
            else:
                u_expanded = u.unsqueeze(0).repeat(x.size(0), 1).to(device)
        else:
            if u.dim() == 2 and u.size(0) == 1:
                u_expanded = u[0].unsqueeze(0).repeat(x.size(0), 1).to(device)
            else:
                u_expanded = u.to(device).repeat(x.size(0), 1)

        u_expanded = u_expanded.to(device=device, dtype=x.dtype)

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = torch.nn.functional.gelu(h)
            h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)

        u_emb = torch.nn.functional.leaky_relu(self.lin_u(u_expanded))
        hcat = torch.cat([h, u_emb], dim=1)
        hcat = torch.nn.functional.leaky_relu(self.lin_1(hcat))

        preds = {}
        y_idx = batch.y_indices
        z_idx = batch.z_indices

        for z_i in range(self.num_z):
            sel = z_idx == z_i
            if not sel.any():
                preds[f"z{z_i}"] = None
                continue

            node_sel = y_idx[sel]
            if node_sel.max() >= hcat.size(0):
                node_sel = node_sel[node_sel < hcat.size(0)]
                if len(node_sel) == 0:
                    preds[f"z{z_i}"] = None
                    continue

            h_sub = hcat[node_sel]
            reg_out = self.redshift_heads[z_i](h_sub)
            logsigma_out = None
            if self.redshift_sigma_heads is not None:
                logsigma_out = self.redshift_sigma_heads[z_i](h_sub).clamp(self.reg_sigma_min, self.reg_sigma_max)

            pred_entry = {"reg": reg_out, "log_sigma": logsigma_out}
            if self.use_moe:
                pred_entry["mix"] = {
                    "ssfr": self.ssfr_mix_heads[z_i](h_sub),
                    "gas": self.gas_mix_heads[z_i](h_sub),
                }
            preds[f"z{z_i}"] = pred_entry

        return preds
