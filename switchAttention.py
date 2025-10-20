import torch
import torch.nn as nn
import math
from typing import List, Optional

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        # x: [B, S, H]
        return self.fn(self.norm(x))

class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*hidden_mult, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class SwitchMultiModalBlock(nn.Module):
    """
    一層 Switch-Attention：
      - 輸入:  modal_list = [X1, X2, ..., XM]，每個 Xi 形狀 [B, K, N, H]
      - 作法:  第 l 層選擇某個模態 i 當 Q；其餘模態 concat 當 K/V
      - 輸出:  更新後的 modal_list，只有被選為 Q 的那個模態會被覆寫(殘差更新)
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = MLP(hidden_dim, hidden_mult=4, dropout=dropout)

        self.pre_attn = PreNorm(hidden_dim, nn.Identity())
        self.pre_ffn  = PreNorm(hidden_dim, nn.Identity())

    def _mhsa(self, Q, K, V):
        """
        Q: [B, SQ, H]
        K/V: [B, SK, H]
        回傳: [B, SQ, H]
        """
        B, SQ, H = Q.shape
        _, SK, _ = K.shape
        assert H % self.num_heads == 0, "H 必須能被 num_heads 整除"
        d = H // self.num_heads

        # 1) 線性投影
        Qh = self.q_proj(Q).view(B, SQ, self.num_heads, d).transpose(1, 2)  # [B, heads, SQ, d]
        Kh = self.k_proj(K).view(B, SK, self.num_heads, d).transpose(1, 2)  # [B, heads, SK, d]
        Vh = self.v_proj(V).view(B, SK, self.num_heads, d).transpose(1, 2)  # [B, heads, SK, d]

        # 2) 注意力
        attn = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(d)        # [B, heads, SQ, SK]
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 3) 聚合
        out = torch.matmul(attn, Vh)                                        # [B, heads, SQ, d]
        out = out.transpose(1, 2).contiguous().view(B, SQ, H)               # [B, SQ, H]
        out = self.out_proj(out)                                            # [B, SQ, H]
        return out

    def forward(self, modal_list: List[torch.Tensor], q_index: int):
        """
        modal_list: 長度 M 的 list，每個元素 [B,K,N,H]
        q_index:    本層選哪個模態當 Query（0..M-1）
        回傳同長度 list；只有 q_index 那個被更新（殘差+FFN）
        """
        M = len(modal_list)
        assert M >= 2, "至少需要兩個模態"
        B, K, N, H = modal_list[0].shape
        for m in range(1, M):
            assert modal_list[m].shape == (B, K, N, H), "所有模態形狀需一致 [B,K,N,H]"

        # 取出 Q 與 其他模態
        Xq = modal_list[q_index]                                           # [B,K,N,H]
        others = [modal_list[m] for m in range(M) if m != q_index]         # M-1 個 [B,K,N,H]

        # 攤平成序列
        S = K * N
        Q = Xq.reshape(B, S, H)                                            # [B, S, H]
        KV = torch.cat([x.reshape(B, S, H) for x in others], dim=1)        # [B, (M-1)*S, H]

        # 前置 LayerNorm（PreNorm 樣式）
        Qn  = self.pre_attn(Q)
        KVn = self.pre_attn(KV)

        # MHSA：Q 來自 q_index 模態；K/V 來自其餘模態串接
        h = self._mhsa(Qn, KVn, KVn)                                       # [B, S, H]
        Xq_new = Q + h                                                     # 殘差
        Xq_new = Xq_new + self.ffn(self.pre_ffn(Xq_new))                   # MLP 殘差

        # 還原回 [B,K,N,H]
        Xq_new = Xq_new.view(B, K, N, H)

        # 只更新 q_index，其他維持不變
        out_list = []
        for m in range(M):
            out_list.append(Xq_new if m == q_index else modal_list[m])
        return out_list
    
class SwitchEncoder(nn.Module):
    def __init__(self, hidden_dim: int, depth: int = 6, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwitchMultiModalBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.depth = depth

    def forward(self, modal_list: List[torch.Tensor]):
        """
        modal_list: [X_price, X_fin, X_event, X_news], 每個 [B,K,N,H]
        回傳:
          - updated_list: 每層輪流更新後的各模態 [B,K,N,H]
          - fused:        融合特徵 [B,K,N,H]（四模態平均）
        """
        M = len(modal_list)
        xs = modal_list
        for l, block in enumerate(self.blocks):
            q_idx = l % M
            xs = block(xs, q_index=q_idx)

        # 融合：平均（也可改成 concat 再線性）
        fused = torch.stack(xs, dim=0).mean(dim=0)  # [B,K,N,H]
        return xs, fused
