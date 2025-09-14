# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class ESGDataset(Dataset):

    """
    每個樣本 = (單一公司, 單一年份)
    回傳keys:
    - price: (12,dp)
    - finance: (12, df)
    - news: (12, dn)
    - event: (12, de)
    - label_E: (1,)
    - label_S: (1,)
    - label_G: (1,)
    - symbol:str
    - year: int
    """
    def __init__(
        self,
        root_price,
        root_finance,
        root_news,
        root_event,
        root_label=None,
        root_year_symbols=None,
        years=range(2015, 2025),
        has_label=True,
        event_dim=64,
        strict_check=True,
        fill_missing_event="zeros",
        label_mode = "all"
    ):
        self.root_price = root_price
        self.root_finance = root_finance
        self.root_news = root_news
        self.root_event = root_event
        self.root_label = root_label
        self.root_year_symbols = root_year_symbols
        self.years = list(years)
        self.has_label = has_label
        self.event_dim = event_dim
        self.strict_check = strict_check
        self.fill_missing_event = fill_missing_event
        self.label_mode = label_mode

        self.samples = []
        for y in years:
            year_samples = self._load_one_year(y)
            self.samples.extend(year_samples)
        
        # dimension initial
        first = self.samples[0]
        self.price_dim = first["price"].shape[1]
        self.finance_dim = first["finance"].shape[1]
        self.news_dim = first["news"].shape[1]
        self.event_dim = first["event"].shape[1]
    
    def _load_year_symbols(self, year):
        if self.root_year_symbols is None:
            return None
        path = os.path.join(self.root_year_symbols, f"{year}_symbol.csv")
        syms = pd.read_csv(path)["symbol"].astype(str).tolist()
        return syms

    def _stack_event(self, year, N):
        months = [f"{m:02d}" for m in range(1, 13)]
        seq = []
        for mm in months:
            fpath = os.path.join(self.root_event, f"event_{year}-{mm}_pca64.npy")
            if os.path.exists(fpath):
                arr = np.load(fpath)  # (N, 64)
                if arr.shape[1] != self.event_dim:
                    raise ValueError(f"[event] dim mismatch at {fpath}, got {arr.shape}")
                if arr.shape[0] != N:
                    raise ValueError(f"[event] N mismatch at {fpath}: got {arr.shape[0]} vs expected {N}")
                seq.append(arr)
            else:
                if self.fill_missing_event == "zeros":
                    seq.append(np.zeros((N, self.event_dim), dtype=np.float32)) # 月份資料不見補0
                else:
                    seq.append(np.full((N, self.event_dim), np.nan, dtype=np.float32))
        return np.stack(seq, axis=0)  # (12, N, d_e)
    
    def min_max(arr, eps=1e-8):
        min_val = arr.min(axis=0, keepdims=True)
        max_val = arr.max(axis=0, keepdims=True)
        return (arr - min_val) / (max_val-min_val + eps) 

    def _load_one_year(self, year):
        symbols = self._load_year_symbols(year)
        
        # news
        news  = np.load(os.path.join(self.root_news,  f"news_{year}.npy"))     # (12, N, 100)

        # price + min-max norm
        price = np.load(os.path.join(self.root_price, f"price_{year}.npy"))    # (12, N, 4)
        min_val = price.min(axis=0, keepdims=True)   # (1, N, 4)
        max_val = price.max(axis=0, keepdims=True)   # (1, N, 4)
        price = (price - min_val) / (max_val - min_val + 1e-8)


        # finance + z-score
        finance_dict = np.load(os.path.join(self.root_finance, f"financial_{year}.npy"), allow_pickle=True).item()
        finance = finance_dict["finance"]  # (12, N, 24)

        mean = np.nanmean(finance, axis=0, keepdims=True)  
        std  = np.nanstd(finance, axis=0, keepdims=True)   
        finance = (finance - mean) / (std + 1e-8)

        N = price.shape[1]

        # event
        event = self._stack_event(year, N)

        # label
        labels = None
        if self.has_label and self.root_label is not None:
            label_path = os.path.join(self.root_label, f"{year}_esg.npy")
            if os.path.exists(label_path):
                labels = np.load(label_path)  # (N, 3)
                labels = labels / 100.0

        # 拆成單公司
        samples = []
        for i in range(N):
            symbol = symbols[i] if symbols is not None else str(i)
            sample = {
                "price":   torch.tensor(price[:, i, :],   dtype=torch.float32),
                "finance": torch.tensor(finance[:, i, :], dtype=torch.float32),
                "news":    torch.tensor(news[:, i, :],    dtype=torch.float32),
                "event":   torch.tensor(event[:, i, :],   dtype=torch.float32),
                "symbol":  symbol,
                "year":    year,
            }

            if labels is not None:
                e, s, g = labels[i]
                if self.label_mode == "all":
                    sample["label"] = torch.tensor([e, s, g], dtype=torch.float32)  # (3,)
                elif self.label_mode == "E":
                    sample["label"] = torch.tensor([e], dtype=torch.float32)        # (1,)
                elif self.label_mode == "S":
                    sample["label"] = torch.tensor([s], dtype=torch.float32)
                elif self.label_mode == "G":
                    sample["label"] = torch.tensor([g], dtype=torch.float32)
                else:
                    raise ValueError(f"Unknown label_mode {self.label_mode}")

            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

