import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utilities
# -------------------------
def _masked_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """
    scores: [..., N]
    mask: same shape as scores or broadcastable. mask True means **valid**.
    """
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask != 0
        scores = scores.masked_fill(~mask, float("-inf"))
    out = F.softmax(scores, dim=dim)
    return torch.nan_to_num(out, nan=0.0)


class SafeLayerNorm(nn.LayerNorm):
    """nan/inf safe LayerNorm wrapper."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        return super().forward(x)


# -------------------------
# Graph Message Passing
# -------------------------
class GraphMessagePassing(nn.Module):
    """
    Attention-based message passing.
    Input: x: [B, N, D], adj: [B, N, N] (mask: 1 for edge/valid)
    Output: [B, N, D]
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.W_att = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.W_att)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [B,N,D]
        adj: [B,N,N] (0/1). If a row has degree 0 we'll add self-loop.
        """
        B, N, D = x.shape
        # ensure at least self-loop for rows with zero degree
        deg = adj.sum(-1, keepdim=True)  # [B,N,1]
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = torch.where(deg > 0, adj, eye)

        Q = self.q(x)   # [B,N,D]
        K = self.k(x)   # [B,N,D]
        V = self.v(x)   # [B,N,D]

        # compute scores: (Q W_att) K^T
        scores = (Q @ self.W_att) @ K.transpose(-1, -2)  # [B,N,N]
        scores = scores / math.sqrt(D)

        attn = _masked_softmax(scores, adj, dim=-1)      # [B,N,N]
        out = attn @ V                                   # [B,N,D]
        out = self.ln(x + self.drop(out))
        return out


# -------------------------
# Recurrent Slot Encoder (shared)
# -------------------------
class RecurrentSlotEncoder(nn.Module):
    """
    Learn slot concepts recurrently across timesteps.
    Input: H: [B,K,N,D]
    Output: S: [B,K,L,D], Beta (attn): [B,K,L,N]
    """
    def __init__(self, hidden_dim: int, num_slots: int, iters_per_step: int = 1, dropout: float = 0.0):
        super().__init__()
        self.D = hidden_dim
        self.L = num_slots
        self.iters = iters_per_step
        self.mu = nn.Parameter(torch.zeros(1, num_slots, hidden_dim))
        self.log_sigma = nn.Parameter(torch.full((1, num_slots, hidden_dim), -2.0))

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.ln = nn.LayerNorm(hidden_dim)

    def step_slot_attention(self, slots: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One cross-attention step on a single timestep.
        slots: [B,L,D], h_t: [B,N,D]
        return: updated_slots [B,L,D], attn [B,L,N]
        """
        B, L, D = slots.shape
        Q = self.q(slots)               # [B,L,D]
        K = self.k(h_t)                 # [B,N,D]
        V = self.v(h_t)                 # [B,N,D]

        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(D)  # [B,L,N]
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        updates = attn @ V              # [B,L,D]
        slots_flat = slots.reshape(-1, D)         # [B*L, D]
        updates_flat = updates.reshape(-1, D)     # [B*L, D]
        slots2 = self.gru(updates_flat, slots_flat).reshape(B, L, D)
        slots2 = self.ln(slots2 + self.ffn(slots2))
        return slots2, attn

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H: [B,K,N,D] -> S: [B,K,L,D], Beta: [B,K,L,N]
        slots initialized from learned mu/log_sigma and a single Gaussian draw per batch.
        """
        B, K, N, D = H.shape
        eps = torch.randn(B, self.L, D, device=H.device, dtype=H.dtype)
        slots = self.mu + torch.exp(self.log_sigma) * eps  # [B,L,D]

        S_list: List[torch.Tensor] = []
        Beta_list: List[torch.Tensor] = []
        for k in range(K):
            h_t = H[:, k]  # [B,N,D]
            slots, beta_k = self.step_slot_attention(slots, h_t)
            S_list.append(slots)
            Beta_list.append(beta_k)

        S = torch.stack(S_list, dim=1)    # [B,K,L,D]
        Beta = torch.stack(Beta_list, dim=1)  # [B,K,L,N]
        return S, Beta


# -------------------------
# Slot Decoder
# -------------------------
class SlotDecoder(nn.Module):
    """
    Decode slots into node-level representations.
    Input: H: [B,K,N,D], S: [B,K,L,D] -> Z: [B,K,N,D]
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, H: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        B, K, N, D = H.shape
        Z_list: List[torch.Tensor] = []
        for k in range(K):
            h = H[:, k]   # [B,N,D]
            s = S[:, k]   # [B,L,D]
            Q = self.q(h)     # [B,N,D]
            K_ = self.k(s)    # [B,L,D]
            V_ = self.v(s)    # [B,L,D]

            scores = (Q @ K_.transpose(-1, -2)) / math.sqrt(D)  # [B,N,L]
            attn = F.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)

            z = attn @ V_   # [B,N,D]
            z = self.ln(h + self.ffn(z))
            Z_list.append(z)

        Z = torch.stack(Z_list, dim=1)  # [B,K,N,D]
        return Z

# ------------------------------
# Temporal encoders
# ------------------------------
class TemporalLSTM(nn.Module):
    """
    Input: x [B, K, N, D_in]
    Output: [B, K, N, D_out]  (D_out == hidden_dim)
    """
    def __init__(self, d_in: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        if bidirectional:
            assert hidden_dim % 2 == 0, "hidden_dim must be even for bidirectional LSTM"
            hid = hidden_dim // 2
        else:
            hid = hidden_dim
        self.lstm = nn.LSTM(input_size=d_in, hidden_size=hid,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)
        # optional projection if needed (keeps dim consistent)
        self.proj = nn.Linear(hidden_dim, hidden_dim) if hidden_dim != hid * (2 if bidirectional else 1) else nn.Identity()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, N, D_in]
        B, K, N, D = x.shape
        x2 = x.permute(0, 2, 1, 3).reshape(B * N, K, D)   # [B*N, K, D_in]
        out, _ = self.lstm(x2)                            # [B*N, K, H]
        out = self.proj(out)                              # [B*N, K, hidden_dim]
        out = out.reshape(B, N, K, self.hidden_dim).permute(0, 2, 1, 3)  # [B, K, N, H]
        out = self.ln(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, K, d_model]
        L = x.size(1)
        return x + self.pe[:, :L].to(x.device)


class TemporalTransformer(nn.Module):
    """
    Input: x [B, K, N, D_in]
    Output: [B, K, N, D_out]  (D_out == d_model)
    Note: src_key_padding_mask (optional) shape -> [B*N, K] with True at PAD positions.
    """
    def __init__(self, d_in: int, d_model: int, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.output_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, K, N, D_in]
        B, K, N, D = x.shape
        x2 = x.permute(0, 2, 1, 3).reshape(B * N, K, D)   # [B*N, K, D_in]
        h = self.input_proj(x2)                           # [B*N, K, d_model]
        h = self.pos_enc(h)
        # src_key_padding_mask shape should be [B*N, K] (True means masked)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        h = self.output_proj(h)                           # [B*N, K, d_model]
        h = h.reshape(B, N, K, -1).permute(0, 2, 1, 3)    # [B, K, N, d_model]
        h = self.ln(h)
        return h


# -------------------------
# DynScan (整體組合)
# -------------------------
class DynScan(nn.Module):
    def __init__(self,
                 price_dim: int,
                 finance_dim: int,
                 event_dim: int,
                 news_dim: Optional[int] = None,
                 hidden_dim: int = 64,
                 num_slots: int = 10,
                 dropout: float = 0.1,
                 out_dim: int = 1,
                 alpha: float = 0.25,
                 lambd: float = 10.0):
        """
        Keeps same architecture intent as your original code.
        """
        super().__init__()
        self.D = hidden_dim
        self.S = num_slots
        self.alpha = alpha
        self.lambd = lambd
        self.has_news = news_dim is not None

        # projections + safe norms
        self.price_norm = SafeLayerNorm(price_dim)
        self.fin_norm = SafeLayerNorm(finance_dim)
        self.event_norm = SafeLayerNorm(event_dim)  
        self.price_proj = TemporalLSTM(d_in=price_dim, hidden_dim=hidden_dim, num_layers=1, dropout=0.1)
        self.fin_proj = TemporalLSTM(d_in=finance_dim, hidden_dim=hidden_dim, num_layers=1, dropout=0.1)
        self.event_proj = TemporalTransformer(d_in=event_dim, d_model=hidden_dim,
                                      nhead=4, num_layers=2, dim_feedforward=hidden_dim*4, dropout=0.1)
        self.news_norm = SafeLayerNorm(news_dim)
        self.news_proj = TemporalTransformer(d_in=news_dim, d_model=hidden_dim,
                                         nhead=4, num_layers=2, dim_feedforward=hidden_dim*4, dropout=0.1)


        # temporal / graph modules
        self.grmp = GraphMessagePassing(hidden_dim, dropout)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # shared slot encoder + decoders
        self.slot_enc = RecurrentSlotEncoder(hidden_dim, num_slots)
        self.dec_price = SlotDecoder(hidden_dim, dropout)
        self.dec_fin = SlotDecoder(hidden_dim, dropout)
        self.dec_event = SlotDecoder(hidden_dim, dropout)
        if self.has_news:
            self.dec_news = SlotDecoder(hidden_dim, dropout)

        # head: concat last-step [Z||H] from each modality
        num_modal = 3 + int(self.has_news)
        num_concat = num_modal * 2  # each modality contributes Z and H (both D)
        self.head = nn.Sequential(
            nn.Linear(num_concat * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # encoding helper
    def _encode_one_modality(self, x: torch.Tensor, adj: torch.Tensor, proj: nn.Module, norm: nn.Module) -> torch.Tensor:
        """
        x: [B,K,N,Dm], adj: [B,K,N,N] -> H: [B,K,N,D] via per-step GRMP then temporal GRU.
        proj may be:
        - nn.Linear (applied to last dim) -> returns [B,K,N,D]
        - TemporalLSTM/TemporalTransformer -> returns [B,K,N,D]
        """
        B, K, N, _ = x.shape
        X = proj(torch.clamp(norm(x), -1e3, 1e3))  # works for Linear and our Temporal encoders
        # per-timestep GraphMessagePassing on projected features
        h_steps = []
        for k in range(K):
            h_k = self.grmp(X[:, k], adj[:, k])  # [B,N,D]
            h_steps.append(h_k)
        H0 = torch.stack(h_steps, dim=1)  # [B,K,N,D]

        # temporal GRU over K (node-wise) - unchanged
        H_in = H0.permute(0, 2, 1, 3).reshape(B * N, K, self.D)
        H_out, _ = self.temporal_gru(H_in)
        H = H_out.reshape(B, N, K, self.D).permute(0, 2, 1, 3)  # [B,K,N,D]
        return H
        
    def forward(self,
                price: torch.Tensor,
                finance: torch.Tensor,
                network: torch.Tensor,
                event: torch.Tensor,
                news: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        price/finance/event/news: [B,K,N,Dm]
        network (adj): [B,K,N,N]
        returns: pred [B,N,out_dim], slots_dict, attn_dict
        """
        # encode modalities
        H_price = self._encode_one_modality(price, network, self.price_proj, self.price_norm)
        H_fin = self._encode_one_modality(finance, network, self.fin_proj, self.fin_norm)
        H_event = self._encode_one_modality(event, network, self.event_proj, self.event_norm)
        H_news = None
        H_news = self._encode_one_modality(news, network, self.news_proj, self.news_norm)

        # slot concepts (shared encoder)
        S_price, B_price = self.slot_enc(H_price)
        S_fin, B_fin = self.slot_enc(H_fin)
        S_event, B_event = self.slot_enc(H_event)
        S_news = B_news = None
        if H_news is not None:
            S_news, B_news = self.slot_enc(H_news)

        # decode per modality
        Z_price = self.dec_price(H_price, S_price)
        Z_fin = self.dec_fin(H_fin, S_fin)
        Z_event = self.dec_event(H_event, S_event)
        Z_news = None
        if H_news is not None:
            Z_news = self.dec_news(H_news, S_news)

        # gather last-step pairs [Z_{t-1} || H_{t-1}] per modality
        def last_step_pair(Z: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
            # both Z and H are [B,K,N,D]
            return torch.cat([Z[:, -1], H[:, -1]], dim=-1)  # [B,N,2D]

        pairs = [
            last_step_pair(Z_price, H_price),
            last_step_pair(Z_fin, H_fin),
            last_step_pair(Z_event, H_event)
        ]
        if Z_news is not None:
            pairs.append(last_step_pair(Z_news, H_news))

        F_node = torch.cat(pairs, dim=-1)  # [B,N, num_modal*2D]
        pred = self.head(F_node)           # [B,N,out_dim]

        slots_dict = {'price': S_price, 'finance': S_fin, 'event': S_event}
        attn_dict = {'price': B_price, 'finance': B_fin, 'event': B_event}
        if S_news is not None:
            slots_dict['news'] = S_news
            attn_dict['news'] = B_news

        return pred, slots_dict, attn_dict

    # -------------------------
    # losses (CMD-like + intra)
    # -------------------------
    @staticmethod
    def _cmd_second_order(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B,L,D] -> returns mean [B,D] and centered second moment c2 [B,D]
        """
        mean = x.mean(dim=1)  # [B,D]
        c2 = ((x - mean.unsqueeze(1)) ** 2).mean(dim=1)  # [B,D]
        return mean, c2

    def compute_losses(self,
                       pred: torch.Tensor,
                       target: torch.Tensor,
                       slots: Dict[str, torch.Tensor],
                       alpha: Optional[float] = None,
                       lambd: Optional[float] = None,
                       mask: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        pred: [B,N,out_dim], target: [B,N,out_dim]
        slots: dict of modality -> [B,K,S,D]
        mask: [B,N] or [B,N,1] (1 valid)
        returns: total, mse, L_inter, L_intra
        """
        # --- DEBUG / SAFETY CHECKS: ensure pred/target shapes and no NaN/Inf ---
        # Ensure shapes match
        if pred.shape != target.shape:
            # try common fixes: squeeze/unsqueeze last or batch dim
            if pred.dim() == target.dim() + 1 and pred.size(0) == 1 and target.dim() >= 2:
                target = target.unsqueeze(0)  # (N,1)->(1,N,1)
            elif target.dim() == pred.dim() + 1 and target.size(0) == 1:
                pred = pred.unsqueeze(0)
        # final assert (will raise informative error if mismatch remains)
        if pred.shape != target.shape:
            raise RuntimeError(f"[compute_losses] pred/target shape mismatch: pred={pred.shape}, target={target.shape}")

        # convert to float and guard against NaN/Inf
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e5, neginf=-1e5)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e5, neginf=-1e5)

        # quick NaN check before computing losses
        if torch.isnan(pred).any() or torch.isnan(target).any():
            print("[compute_losses] WARNING: NaN found in pred or target!")
            print("pred stats:", torch.min(pred), torch.max(pred), torch.mean(pred))
            print("target stats:", torch.min(target), torch.max(target), torch.mean(target))

        if alpha is None:
            alpha = self.alpha
        if lambd is None:
            lambd = self.lambd

        # MSE (masked)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            if mask.size(-1) == 1:
                mask = mask.expand_as(pred)
            mask = mask.to(pred.dtype)
            mse = ((pred - target) ** 2 * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        else:
            mse = F.mse_loss(pred, target)

        # Inter-modality CMD-like (use last timestep slots)
        S_last: List[torch.Tensor] = []
        # --- CHECK slots for NaN / inf / extreme values ---
        for kname, S in slots.items():
            if S is None:
                continue
            S_last_step = S[:, -1]   # [B,S,D]
            if torch.isnan(S_last_step).any() or torch.isinf(S_last_step).any():
                print(f"[compute_losses] WARNING: NaN/Inf in slots for {kname} (last step).")
                print(f"  stats - min {torch.min(S_last_step)}, max {torch.max(S_last_step)}, mean {torch.mean(S_last_step)}")
                # sanitize to prevent NaNs propagating
                slots[kname] = torch.nan_to_num(S, nan=0.0, posinf=1e5, neginf=-1e5)

        for key in ['price', 'finance', 'event', 'news']:
            if key in slots:
                S_last.append(slots[key][:, -1])  # [B,S,D]

        L_inter = pred.new_tensor(0.0)
        pair_cnt = 0
        for i in range(len(S_last)):
            for j in range(i + 1, len(S_last)):
                mi, ci2 = self._cmd_second_order(S_last[i])
                mj, cj2 = self._cmd_second_order(S_last[j])
                L_inter = L_inter + F.mse_loss(mi, mj) + F.mse_loss(ci2, cj2)
                pair_cnt += 1
        if pair_cnt > 0:
            L_inter = L_inter / pair_cnt

        # Intra-modality disentanglement: off-diagonal cov of features (encourage diagonal dominance)
        def offdiag_cov_loss(S: torch.Tensor) -> torch.Tensor:
            """
            S: [B,S,D] -> compute mean squared off-diagonal elements of covariance across feature-dim
            """
            B, S_, D = S.shape
            mean = S.mean(dim=1, keepdim=True)  # [B,1,D]
            X = S - mean                         # [B,S,D]
            # covariance across slots in feature space -> [B,D,D]
            cov = torch.bmm(X.transpose(1, 2), X) / (S_ - 1 + 1e-6)
            off_diag = cov - torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
            return (off_diag ** 2).mean()

        L_intra = pred.new_tensor(0.0)
        cntm = 0
        for S in S_last:
            L_intra = L_intra + offdiag_cov_loss(S)
            cntm += 1
        if cntm > 0:
            L_intra = L_intra / cntm

        total = mse + alpha * L_inter + lambd * L_intra
        return total, mse, L_inter, L_intra
