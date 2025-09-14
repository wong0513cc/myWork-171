import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

# ---------- Utils ----------
class SafeLayerNorm(nn.LayerNorm):
    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        return super().forward(x)

# ---------- Slot Encoder ----------
class RecurrentSlotEncoder(nn.Module):
    """H: [B,K,N,D]  ->  S: [B,K,L,D], Beta: [B,K,L,N]"""
    def __init__(self, hidden_dim: int, num_slots: int, iters_per_step: int = 1):
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
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.ln = nn.LayerNorm(hidden_dim)

    def _step(self, slots: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = slots.shape
        Q = self.q(slots)                          # [B,L,D]
        K = self.k(h_t)                            # [B,N,D]
        V = self.v(h_t)                            # [B,N,D]
        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(D)  # [B,L,N]
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        updates = attn @ V                         # [B,L,D]
        slots2 = self.gru(updates.reshape(-1, D), slots.reshape(-1, D)).reshape(B, L, D)
        slots2 = self.ln(slots2 + self.ffn(slots2))
        return slots2, attn

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, N, D = H.shape
        eps = torch.randn(B, self.L, D, device=H.device)
        slots = self.mu + torch.exp(self.log_sigma) * eps  # [B,L,D]
        S_list, Beta_list = [], []
        for k in range(K):
            slots, beta = self._step(slots, H[:, k])       # h_t: [B,N,D]
            S_list.append(slots)
            Beta_list.append(beta)
        S = torch.stack(S_list, dim=1)                     # [B,K,L,D]
        Beta = torch.stack(Beta_list, dim=1)               # [B,K,L,N]
        return S, Beta

class SlotESGSingleTaskSimple(nn.Module):
    """
    簡化版：
    - 沒有 SlotDecoder
    - 沒有每模態 GRU（改成 Linear+ReLU）
    - Fusion 改成 mean pooling
    """
    def __init__(self,
                 price_dim: int,
                 finance_dim: int,
                 event_dim: int,
                 news_dim: Optional[int] = None,
                 hidden_dim: int = 64,
                 num_slots: int = 6,
                 dropout: float = 0.1,
                 task: str = "E",
                 bounded_output: bool = True):
        super().__init__()
        assert task in {"E", "S", "G"}
        self.task = task
        self.bounded_output = bounded_output
        self.has_news = news_dim is not None
        self.D = hidden_dim

        # 投影到共享維度
        self.price_norm = SafeLayerNorm(price_dim)
        self.fin_norm   = SafeLayerNorm(finance_dim)
        self.event_norm = SafeLayerNorm(event_dim)
        self.price_proj = nn.Sequential(nn.Linear(price_dim, hidden_dim), nn.ReLU())
        self.fin_proj   = nn.Sequential(nn.Linear(finance_dim, hidden_dim), nn.ReLU())
        self.event_proj = nn.Sequential(nn.Linear(event_dim, hidden_dim), nn.ReLU())
        if self.has_news:
            self.news_norm = SafeLayerNorm(news_dim)
            self.news_proj = nn.Sequential(nn.Linear(news_dim, hidden_dim), nn.ReLU())

        # 共享 Slot Encoder
        self.slot_enc   = RecurrentSlotEncoder(hidden_dim, num_slots)

        # 融合 + head
        self.num_modal  = 3 + int(self.has_news)
        self.fusion = nn.Sequential(
            nn.Linear(num_slots * hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _encode_modality(self, x, norm, proj):
        x = torch.clamp(norm(x), -1e3, 1e3)  # [B,K,d_m]
        x = proj(x)                          # [B,K,D]
        return x

    def forward(self,
                price: torch.Tensor,   # [B,K,d_p]
                finance: torch.Tensor, # [B,K,d_f]
                event: torch.Tensor,   # [B,K,d_e]
                news: Optional[torch.Tensor] = None  # [B,K,d_n] or None
                ) -> Dict[str, torch.Tensor]:

        # (1) 各模態 embedding
        H_price = self._encode_modality(price,   self.price_norm, self.price_proj)   # [B,K,D]
        H_fin   = self._encode_modality(finance, self.fin_norm,   self.fin_proj)     # [B,K,D]
        H_event = self._encode_modality(event,   self.event_norm, self.event_proj)   # [B,K,D]
        tokens = [H_price, H_fin, H_event]
        if self.has_news and news is not None:
            H_news = self._encode_modality(news, self.news_norm, self.news_proj)     # [B,K,D]
            tokens.append(H_news)

        # (2) H_tokens: [B,K,N,D]
        H_tokens = torch.stack(tokens, dim=2)

        # (3) Slot Encoder
        S, Beta = self.slot_enc(H_tokens)    # S: [B,K,L,D]

        # (4) 取最後一步的 slots → flatten
        S_last = S[:, -1]                    # [B,L,D]
        F = S_last.reshape(S_last.size(0), -1)   # [B,L*D]
        F = self.fusion(F)                   # [B,D]

        # (5) 輸出
        pred = self.head(F)                  # [B,1]
        if self.bounded_output:
            pred = torch.sigmoid(pred)

        return {"pred": pred, "slots": S, "assign": Beta}
