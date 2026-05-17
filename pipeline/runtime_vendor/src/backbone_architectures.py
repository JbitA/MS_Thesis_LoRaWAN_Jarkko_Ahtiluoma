"""
FILE STORY — ``backbone_architectures.py``
==========================================

**Role.** PyTorch LSTM, GRU, TCN, Transformer heads (battery + halt outputs).

**Connects.** ``multitask_battery_halt_trainer_bundle.py``.

**Developed with Cursor AI.**
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Forget-gate LSTM  (thesis: net_phi, y_phi, cell s^c, inject y_inj * g(net^c))
# ---------------------------------------------------------------------------


class ForgetGateLSTMCell(nn.Module):  # define neural / helper class ForgetGateLSTMCell
    """
    One LSTM timestep with explicit forget, input, output gates.

    For gate φ ∈ {f, i, o}:
        net_φj(t) = Σ_m w_φjm · y_m(t−1)   (here: linear(x_t, h_{t−1}))
        y_φj(t) = f_φ(net_φj(t))
    Cell (j index suppressed in batch form):
        s^c(0) = 0
        s^c(t) = y_f(t) ⊙ s^c(t−1) + y_i(t) ⊙ g(net^c(t)),  t > 0
        h(t) = y_o(t) ⊙ tanh(s^c(t))
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()  # call parent constructor
        self.hidden_dim = int(hidden_dim)  # bind self.hidden_dim
        # Combined linear maps implement Σ_m w_φjm y_m(t−1) for x_t and h_{t−1}.
        self.linear_f = nn.Linear(input_dim + hidden_dim, hidden_dim)  # bind self.linear_f
        self.linear_i = nn.Linear(input_dim + hidden_dim, hidden_dim)  # bind self.linear_i
        self.linear_g = nn.Linear(input_dim + hidden_dim, hidden_dim)  # bind self.linear_g
        self.linear_o = nn.Linear(input_dim + hidden_dim, hidden_dim)  # bind self.linear_o

    def forward(  # def forward
        self,  # continued expression
        x_t: torch.Tensor,  # continued expression
        h_prev: torch.Tensor,  # continued expression
        c_prev: torch.Tensor,  # continued expression
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # start block
        # Concatenate current input and previous hidden state (previous y_m).
        xh = torch.cat([x_t, h_prev], dim=-1)  # bind xh
        net_f = self.linear_f(xh)  # bind net_f
        y_f = torch.sigmoid(net_f)  # bind y_f
        net_i = self.linear_i(xh)  # bind net_i
        y_i = torch.sigmoid(net_i)  # bind y_i
        net_g = self.linear_g(xh)  # bind net_g
        y_g = torch.tanh(net_g)  # bind y_g
        net_o = self.linear_o(xh)  # bind net_o
        y_o = torch.sigmoid(net_o)  # bind y_o
        c_t = y_f * c_prev + y_i * y_g  # bind c_t
        h_t = y_o * torch.tanh(c_t)  # bind h_t
        return h_t, c_t


class ForgetGateLSTMEncoder(nn.Module):
    """Stacked forget-gate LSTM over time; returns final hidden state."""  # statement

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None: # define callable __init__
        super().__init__()  # invoke parent nn.Module / object init
        self.num_layers = int(num_layers)  # assign / bind self.num_layers
        self.hidden_dim = int(hidden_dim)  # assign / bind self.hidden_dim
        self.cells = nn.ModuleList(  # stack of forget-gate LSTM cells (layer 0 reads features)
            [ForgetGateLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )  # close ModuleList of per-layer cells

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # define callable forward
        # x: (B, T, F)
        bsz, _, _ = x.shape  # assign / bind bsz, _, _
        device = x.device  # assign / bind device
        h = [torch.zeros(bsz, self.hidden_dim, device=device) for _ in range(self.num_layers)] # assign / bind h
        c = [torch.zeros(bsz, self.hidden_dim, device=device) for _ in range(self.num_layers)] # assign / bind c
        for t in range(x.size(1)):  # iterate sequence / index range
            inp = x[:, t, :]  # assign / bind inp
            for layer, cell in enumerate(self.cells):  # iterate sequence / index range
                h[layer], c[layer] = cell(inp, h[layer], c[layer]) # assign / bind h[layer], c[layer]
                inp = h[layer]  # assign / bind inp
        return h[-1]  # return value to caller


class LSTMHead(nn.Module):  # define neural / helper class LSTMHead
    """Multitask head on forget-gate LSTM encoder."""  # module/class docstring (single-line form)

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_battery_out: int = 1, n_halt_out: int = 1) -> None: # define callable __init__
        super().__init__()  # invoke parent nn.Module / object init
        self.n_battery_out = int(n_battery_out)  # assign / bind self.n_battery_out
        self.n_halt_out = int(n_halt_out)  # assign / bind self.n_halt_out
        self.encoder = ForgetGateLSTMEncoder(in_dim, hidden_dim=hidden_dim, num_layers=2) # assign / bind self.encoder
        self.reg = nn.Linear(hidden_dim, self.n_battery_out)  # assign / bind self.reg
        self.cls = nn.Linear(hidden_dim, self.n_halt_out)  # assign / bind self.cls

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # define callable forward
        z = self.encoder(x)  # assign / bind z
        r = self.reg(z)  # assign / bind r
        c = self.cls(z)  # assign / bind c
        if self.n_battery_out == 1:  # conditional branch on runtime predicate
            r = r.squeeze(-1)  # assign / bind r
        if self.n_halt_out == 1:  # conditional branch on runtime predicate
            c = c.squeeze(-1)  # assign / bind c
        return r, c


# ---------------------------------------------------------------------------
# GRU  (thesis: r_j, z_j, h̃_j, h_j)
# ---------------------------------------------------------------------------


class GatedRecurrentUnitCell(nn.Module):  # define neural / helper class GatedRecurrentUnitCell
    """
    r_j = σ([W_r x]_j + [U_r h(t−1)]_j)
    z_j = σ([W_z x]_j + [U_z h(t−1)]_j)
    h̃_j = φ([W x]_j + [U (r ⊙ h(t−1))]_j)
    h_j = z_j h_{j-1} + (1 − z_j) h̃_j
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()  # call parent constructor
        self.hidden_dim = int(hidden_dim)  # bind self.hidden_dim
        self.linear_r = nn.Linear(input_dim + hidden_dim, hidden_dim)  # bind self.linear_r
        self.linear_z = nn.Linear(input_dim + hidden_dim, hidden_dim)  # bind self.linear_z
        self.linear_h = nn.Linear(input_dim + hidden_dim, hidden_dim)  # bind self.linear_h

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x_t, h_prev], dim=-1)  # bind xh
        r = torch.sigmoid(self.linear_r(xh))  # bind r
        z = torch.sigmoid(self.linear_z(xh))  # bind z
        xh_tilde = torch.cat([x_t, r * h_prev], dim=-1)  # bind xh_tilde
        h_tilde = torch.tanh(self.linear_h(xh_tilde))  # bind h_tilde
        h_t = z * h_prev + (1.0 - z) * h_tilde  # bind h_t
        return h_t


class GatedRecurrentUnitEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()  # call parent constructor
        self.hidden_dim = int(hidden_dim)  # bind self.hidden_dim
        self.num_layers = int(num_layers)  # bind self.num_layers
        self.cells = nn.ModuleList(  # stack of GRU cells (layer 0 reads features)
            [GatedRecurrentUnitCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )  # close ModuleList of per-layer cells

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, _ = x.shape  # bind bsz, _, _
        device = x.device  # bind device
        h = [torch.zeros(bsz, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        for t in range(x.size(1)):  # for-loop over iterable
            inp = x[:, t, :]  # bind inp
            for layer, cell in enumerate(self.cells):  # for-loop over iterable
                h[layer] = cell(inp, h[layer])  # bind h[layer]
                inp = h[layer]  # bind inp
        return h[-1]


class GRUHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, n_battery_out: int = 1, n_halt_out: int = 1) -> None:
        super().__init__()  # call parent constructor
        self.n_battery_out = int(n_battery_out)  # bind self.n_battery_out
        self.n_halt_out = int(n_halt_out)  # bind self.n_halt_out
        self.encoder = GatedRecurrentUnitEncoder(in_dim, hidden_dim=hidden_dim, num_layers=2)  # bind self.encoder
        self.reg = nn.Linear(hidden_dim, self.n_battery_out)  # bind self.reg
        self.cls = nn.Linear(hidden_dim, self.n_halt_out)  # bind self.cls

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)  # bind z
        r = self.reg(z)  # bind r
        c = self.cls(z)  # bind c
        if self.n_battery_out == 1:  # conditional branch
            r = r.squeeze(-1)  # bind r
        if self.n_halt_out == 1:  # conditional branch
            c = c.squeeze(-1)  # bind c
        return r, c


# ---------------------------------------------------------------------------
# TCN: F(s) = (x *d f)(s), ReLU, batch norm
# ---------------------------------------------------------------------------


class BatchNorm1dChannels(nn.Module):
    """
    BN over channel dimension for conv features:
        μ_B = (1/m) Σ x_i,  σ²_B = (1/m) Σ (x_i − μ_B)²
        x̂_i = (x_i − μ_B) / sqrt(σ²_B + ε),  y_i = γ x̂_i + β
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()  # call parent constructor
        self.bn = nn.BatchNorm1d(num_channels, eps=eps)  # bind self.bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class DilatedTemporalBlock(nn.Module):
    """Causal dilated conv: F(s) = Σ_i f(i) · x_{s − d·i}."""  # bind """Causal dilated conv: F(s)

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None: # define callable __init__
        super().__init__()  # invoke parent nn.Module / object init
        self.dilation = int(dilation)  # assign / bind self.dilation
        pad = (kernel_size - 1) * self.dilation  # assign / bind pad
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=self.dilation) # assign / bind self.conv
        self.relu = nn.ReLU()  # assign / bind self.relu
        self.norm = BatchNorm1dChannels(out_ch)  # assign / bind self.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # define callable forward
        # x: (B, C, T)
        h = self.conv(x)  # assign / bind h
        h = h[:, :, : x.size(2)]  # assign / bind h
        h = self.relu(h)  # assign / bind h
        h = self.norm(h)  # assign / bind h
        return h  # return value to caller


class TCNHead(nn.Module):  # define neural / helper class TCNHead
    def __init__(  # define callable __init__
        self,  # execute statement in training / data pipeline
        in_dim: int,  # execute statement in training / data pipeline
        channels: Tuple[int, ...] = (64, 64),  # assign / bind channels: Tuple[int, ...]
        kernel_size: int = 3,  # assign / bind kernel_size: int
        n_battery_out: int = 1,  # assign / bind n_battery_out: int
        n_halt_out: int = 1,  # assign / bind n_halt_out: int
    ) -> None:  # begin nested block
        super().__init__()  # invoke parent nn.Module / object init
        self.n_battery_out = int(n_battery_out)  # assign / bind self.n_battery_out
        self.n_halt_out = int(n_halt_out)  # assign / bind self.n_halt_out
        blocks: List[nn.Module] = []  # assign / bind blocks: List[nn.Module]
        c_in = in_dim  # assign / bind c_in
        for i, c_out in enumerate(channels):  # iterate sequence / index range
            blocks.append(DilatedTemporalBlock(c_in, c_out, kernel_size, dilation=2**i)) # assign / bind blocks.append(DilatedTemporalBlock(c_in, c_out, 
            c_in = c_out  # assign / bind c_in
        self.tcn = nn.Sequential(*blocks)  # assign / bind self.tcn
        self.reg = nn.Linear(c_in, self.n_battery_out)  # assign / bind self.reg
        self.cls = nn.Linear(c_in, self.n_halt_out)  # assign / bind self.cls

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # define callable forward
        h = self.tcn(x.transpose(1, 2))  # assign / bind h
        z = h[:, :, -1]  # assign / bind z
        r = self.reg(z)  # assign / bind r
        c = self.cls(z)  # assign / bind c
        if self.n_battery_out == 1:  # conditional branch on runtime predicate
            r = r.squeeze(-1)  # assign / bind r
        if self.n_halt_out == 1:  # conditional branch on runtime predicate
            c = c.squeeze(-1)  # assign / bind c
        return r, c  # return value to caller


# ---------------------------------------------------------------------------
# Transformer: Attention, MultiHead, FFN = max(0, xW1+b1)W2+b2
# ---------------------------------------------------------------------------


class ScaledDotProductAttention(nn.Module): # define neural / helper class ScaledDotProductAttention
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V."""  # module/class docstring (single-line form)

    def __init__(self, dropout: float = 0.1) -> None:  # define callable __init__
        super().__init__()  # invoke parent nn.Module / object init
        self.dropout = nn.Dropout(dropout)  # assign / bind self.dropout

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: # define callable forward
        d_k = q.size(-1)  # assign / bind d_k
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(d_k)) # assign / bind scores
        weights = F.softmax(scores, dim=-1)  # assign / bind weights
        weights = self.dropout(weights)  # assign / bind weights
        return torch.matmul(weights, v)


class MultiHeadSelfAttention(nn.Module):  # define neural / helper class MultiHeadSelfAttention
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()  # call parent constructor
        assert d_model % nhead == 0  # assert invariant
        self.nhead = int(nhead)  # bind self.nhead
        self.d_head = d_model // nhead  # bind self.d_head
        self.W_q = nn.Linear(d_model, d_model)  # bind self.W_q
        self.W_k = nn.Linear(d_model, d_model)  # bind self.W_k
        self.W_v = nn.Linear(d_model, d_model)  # bind self.W_v
        self.W_o = nn.Linear(d_model, d_model)  # bind self.W_o
        self.attn = ScaledDotProductAttention(dropout=dropout)  # bind self.attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tlen, _ = x.shape  # bind bsz, tlen, _
        q = self._split_heads(self.W_q(x))  # bind q
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))  # bind v
        heads = []  # bind heads
        for h in range(self.nhead):  # for-loop over iterable
            heads.append(self.attn(q[:, h], k[:, h], v[:, h]))
        concat = torch.cat(heads, dim=-1)  # bind concat
        return self.W_o(concat)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tlen, _ = x.shape  # bind bsz, tlen, _
        return x.view(bsz, tlen, self.nhead, self.d_head).transpose(1, 2)


class PositionwiseFeedForward(nn.Module):
    """NN(x) = max(0, x W1 + b1) W2 + b2."""  # bind """NN(x)

    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.15) -> None: # define callable __init__
        super().__init__()  # invoke parent nn.Module / object init
        self.w1 = nn.Linear(d_model, dim_ff)  # assign / bind self.w1
        self.w2 = nn.Linear(dim_ff, d_model)  # assign / bind self.w2
        self.dropout = nn.Dropout(dropout)  # assign / bind self.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # define callable forward
        h = F.relu(self.w1(x))  # assign / bind h
        h = self.dropout(h)  # assign / bind h
        return self.w2(h)  # return value to caller


class TransformerEncoderBlock(nn.Module):  # define neural / helper class TransformerEncoderBlock
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.15) -> None: # define callable __init__
        super().__init__()  # invoke parent nn.Module / object init
        self.ln1 = nn.LayerNorm(d_model)  # assign / bind self.ln1
        self.mha = MultiHeadSelfAttention(d_model, nhead, dropout=dropout) # assign / bind self.mha
        self.ln2 = nn.LayerNorm(d_model)  # assign / bind self.ln2
        self.ffn = PositionwiseFeedForward(d_model, dim_ff, dropout=dropout) # assign / bind self.ffn
        self.dropout = nn.Dropout(dropout)  # assign / bind self.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # define callable forward
        x = x + self.dropout(self.mha(self.ln1(x)))  # assign / bind x
        x = x + self.dropout(self.ffn(self.ln2(x)))  # assign / bind x
        return x  # return value to caller


class TransformerHeadV5(nn.Module):  # define neural / helper class TransformerHeadV5
    def __init__(  # define callable __init__
        self,  # execute statement in training / data pipeline
        in_dim: int,  # execute statement in training / data pipeline
        d_model: int = 96,  # assign / bind d_model: int
        nhead: int = 4,  # assign / bind nhead: int
        layers: int = 1,  # assign / bind layers: int
        n_battery_out: int = 1,  # assign / bind n_battery_out: int
        n_halt_out: int = 1,  # assign / bind n_halt_out: int
    ) -> None:  # begin nested block
        super().__init__()  # invoke parent nn.Module / object init
        self.n_battery_out = int(n_battery_out)  # assign / bind self.n_battery_out
        self.n_halt_out = int(n_halt_out)  # assign / bind self.n_halt_out
        self.in_proj = nn.Linear(in_dim, d_model)  # assign / bind self.in_proj
        self.blocks = nn.ModuleList(  # repeated transformer encoder blocks
            [TransformerEncoderBlock(d_model, nhead, d_model * 2, dropout=0.15) for _ in range(layers)]
        )  # close ModuleList of encoder blocks
        self.out_ln = nn.LayerNorm(d_model)  # assign / bind self.out_ln
        self.reg = nn.Linear(d_model, self.n_battery_out)  # assign / bind self.reg
        self.cls = nn.Linear(d_model, self.n_halt_out)  # assign / bind self.cls

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # define callable forward
        z = self.in_proj(x)  # assign / bind z
        for blk in self.blocks:  # iterate sequence / index range
            z = blk(z)  # assign / bind z
        u = self.out_ln(z[:, -1, :])  # assign / bind u
        r = self.reg(u)  # assign / bind r
        c = self.cls(u)  # assign / bind c
        if self.n_battery_out == 1:  # conditional branch on runtime predicate
            r = r.squeeze(-1)  # assign / bind r
        if self.n_halt_out == 1:  # conditional branch on runtime predicate
            c = c.squeeze(-1)  # assign / bind c
        return r, c  # return value to caller
