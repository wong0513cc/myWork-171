# train.py
import os
import math
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

from dataset import ESGDataset
from model import SlotESGSingleTaskSimple  # 單任務、無圖、保留 slot 的模型

# ---------------- Utils ----------------
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def parse_year_span(span: str) -> range:
    # "2015-2019" -> range(2015, 2020)
    a, b = span.split("-")
    return range(int(a), int(b) + 1)

def sMAPE(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    with torch.no_grad():
        p = pred.detach().cpu()
        t = target.detach().cpu()
        return (100 * torch.mean(torch.abs(p - t) / ((torch.abs(p) + torch.abs(t)) / 2 + eps))).item()

def create_loader(ds_or_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=False):
    return DataLoader(
        ds_or_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

def filter_indices_by_year_and_label(dataset: ESGDataset, years: range) -> List[int]:
    idx = []
    for i, s in enumerate(dataset.samples):
        if s["year"] in years and ("label" in s) and (s["label"] is not None):
            t = s["label"]
            if torch.is_tensor(t):
                if torch.isnan(t).any():  # 跳過 NaN 標籤
                    continue
            idx.append(i)
    return idx

def build_year_loaders(dataset: ESGDataset, years: range, batch_size: int,
                       num_workers=4, pin_memory=True) -> Dict[int, DataLoader]:
    """為每個年份建立一個 DataLoader，方便逐年評估。"""
    out = {}
    for y in years:
        idx = [i for i, s in enumerate(dataset.samples) if s["year"] == y and s.get("label") is not None]
        if len(idx) == 0:
            continue
        subset = Subset(dataset, idx)
        out[y] = create_loader(subset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory)
    return out

# ---------------- Loss / Train Step ----------------
def forward_and_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_regs: bool,
    lambd_intra: float,
    gamma_attn: float,
    tau_temp: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    price   = batch["price"].to(device)     # [B, K, d_p]
    finance = batch["finance"].to(device)   # [B, K, d_f]
    event   = batch["event"].to(device)     # [B, K, d_e]
    news    = batch.get("news", None)
    if news is not None and torch.is_tensor(news):
        news = news.to(device)

    out   = model(price, finance, event, news)
    pred  = out["pred"]                     # [B,1]
    label = batch["label"].to(device)       # [B,1]

    # --- loss ---
    mse_loss = F.mse_loss(pred, label)

    total = mse_loss   # 沒有 regs，直接 total=mse

    with torch.no_grad():
        mse  = mse_loss.item()
        rmse = math.sqrt(max(mse, 0.0))
        mae  = F.l1_loss(pred, label).item()
        smape_val = sMAPE(pred, label)

    metrics = {
        "total": total.item(),
        "mse": mse, "rmse": rmse, "mae": mae, "smape": smape_val,
        "L_intra": 0.0, "L_attn": 0.0, "L_temp": 0.0
    }
    return total, metrics

def run_one_epoch(model, loader, optimizer, device, train: bool, amp: bool,
                  use_regs: bool, lambd_intra: float, gamma_attn: float, tau_temp: float,
                  grad_clip: float = 1.0) -> Dict[str, float]:
    model.train(mode=train)
    scaler = GradScaler(enabled=amp)

    agg = defaultdict(float)
    n = 0
    for batch in loader:
        with autocast(enabled=amp):
            loss, m = forward_and_loss(model, batch, device, use_regs, lambd_intra, gamma_attn, tau_temp)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        # 累積
        for k, v in m.items():
            agg[k] += float(v)
        n += 1

    return {k: (v / max(n, 1)) for k, v in agg.items()}

@torch.no_grad()
def evaluate_by_year(model, year_loaders: Dict[int, DataLoader], device,
                     use_regs: bool, lambd_intra: float, gamma_attn: float, tau_temp: float) -> Dict[int, Dict[str, float]]:
    model.eval()
    results = {}
    for y, loader in sorted(year_loaders.items()):
        agg = defaultdict(float); n = 0
        for batch in loader:
            with autocast(enabled=False):
                _, m = forward_and_loss(model, batch, device, use_regs, lambd_intra, gamma_attn, tau_temp)
            for k, v in m.items():
                agg[k] += float(v)
            n += 1
        results[y] = {k: (v / max(n, 1)) for k, v in agg.items()}
    return results

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    # 路徑
    parser.add_argument("--root_price", required=True)
    parser.add_argument("--root_finance", required=True)
    parser.add_argument("--root_news", required=True)
    parser.add_argument("--root_event", required=True)
    parser.add_argument("--root_label", required=True)
    parser.add_argument("--root_year_symbols", required=True)

    # 年份切分
    parser.add_argument("--train_years", default="2015-2019")
    parser.add_argument("--val_years",   default="2020-2021")
    parser.add_argument("--test_years",  default="2022-2024")

    # 任務 / 資料
    parser.add_argument("--task", choices=["E", "S", "G"], required=True, help="單任務訓練：E / S / G")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    # 模型
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_slots", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bounded_output", action="store_true")

    # 訓練
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # 規則化（先建議設 0，穩定後逐步開）
    parser.add_argument("--lambd_intra", type=float, default=0.0)  # slot 去相關
    parser.add_argument("--gamma_attn",  type=float, default=0.0)  # 注意力熵
    parser.add_argument("--tau_temp",    type=float, default=0.0)  # 時間平滑
    parser.add_argument("--use_regs",    action="store_true")

    # 輸出
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--log_prefix", default="single_task")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 建 Dataset（單公司、單任務 label）----
    # label_mode 要與 task 對齊
    label_mode = args.task
    all_years = range(int(args.train_years.split("-")[0]), int(args.test_years.split("-")[1]) + 1)
    dataset = ESGDataset(
        root_price=args.root_price,
        root_finance=args.root_finance,
        root_news=args.root_news,
        root_event=args.root_event,
        root_label=args.root_label,
        root_year_symbols=args.root_year_symbols,
        years=all_years,
        has_label=True,
        label_mode=label_mode,        # "E"|"S"|"G" -> label shape [1]
    )
    print(f"Total samples: {len(dataset)}")

    # ---- 切分（依年份 + 去掉無標/NaN 樣本）----
    years_train = parse_year_span(args.train_years)
    years_val   = parse_year_span(args.val_years)
    years_test  = parse_year_span(args.test_years)

    idx_train = filter_indices_by_year_and_label(dataset, years_train)
    idx_val   = filter_indices_by_year_and_label(dataset, years_val)
    idx_test  = filter_indices_by_year_and_label(dataset, years_test)

    train_set = Subset(dataset, idx_train)
    val_set   = Subset(dataset, idx_val)
    test_set  = Subset(dataset, idx_test)

    train_loader = create_loader(train_set, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader   = create_loader(val_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader  = create_loader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)

    # 每年 dataloader（for per-year metrics）
    test_year_loaders = build_year_loaders(dataset, years_test, batch_size=args.batch_size,
                                           num_workers=args.num_workers, pin_memory=args.pin_memory)

    print(f"Train/Val/Test = {len(train_set)}/{len(val_set)}/{len(test_set)}")

    # ---- Model / Optim ----
    model = SlotESGSingleTaskSimple(
        price_dim=dataset.price_dim,
        finance_dim=dataset.finance_dim,
        event_dim=dataset.event_dim,
        news_dim=getattr(dataset, "news_dim", None),
        hidden_dim=args.hidden_dim,
        num_slots=args.num_slots,
        dropout=args.dropout,
        task=args.task,
        bounded_output=args.bounded_output,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Train Loop ----
    best_val = float("inf"); best_ep = -1
    ckpt_path = os.path.join(args.save_dir, f"{args.log_prefix}_{args.task}_best.pt")

    hist = {
        "epoch": [],
        "train_total": [], "train_mse": [], "train_rmse": [], "train_mae": [], "train_smape": [],
        "val_total": [],   "val_mse": [],   "val_rmse": [],   "val_mae": [],   "val_smape": [],
        "L_intra": [], "L_attn": [], "L_temp": []
    }

    for ep in range(1, args.epochs + 1):
        tr = run_one_epoch(
            model, train_loader, optimizer, device,
            train=True, amp=args.amp, use_regs=args.use_regs,
            lambd_intra=args.lambd_intra, gamma_attn=args.gamma_attn, tau_temp=args.tau_temp,
            grad_clip=1.0
        )
        vl = run_one_epoch(
            model, val_loader, optimizer, device,
            train=False, amp=False, use_regs=args.use_regs,
            lambd_intra=args.lambd_intra, gamma_attn=args.gamma_attn, tau_temp=args.tau_temp
        )
        scheduler.step()

        print(f"[Epoch {ep:03d}][{args.task}] "
              f"train_total={tr.get('total', float('nan')):.5f} "
              f"val_total={vl.get('total', float('nan')):.5f}  "
              f"(val_mse={vl.get('mse', float('nan')):.5f}  val_rmse={vl.get('rmse', float('nan')):.5f}  "
              f"val_mae={vl.get('mae', float('nan')):.5f}  val_smape={vl.get('smape', float('nan')):.5f})")

        # 記錄
        hist["epoch"].append(ep)
        for k in ["total","mse","rmse","mae","smape","L_intra","L_attn","L_temp"]:
            hist_key = ("train_" + k) if k in tr else None
        hist["train_total"].append(tr.get("total", float("nan")))
        hist["train_mse"].append(tr.get("mse", float("nan")))
        hist["train_rmse"].append(tr.get("rmse", float("nan")))
        hist["train_mae"].append(tr.get("mae", float("nan")))
        hist["train_smape"].append(tr.get("smape", float("nan")))
        hist["val_total"].append(vl.get("total", float("nan")))
        hist["val_mse"].append(vl.get("mse", float("nan")))
        hist["val_rmse"].append(vl.get("rmse", float("nan")))
        hist["val_mae"].append(vl.get("mae", float("nan")))
        hist["val_smape"].append(vl.get("smape", float("nan")))
        hist["L_intra"].append(tr.get("L_intra", 0.0))
        hist["L_attn"].append(tr.get("L_attn", 0.0))
        hist["L_temp"].append(tr.get("L_temp", 0.0))

        # 儲存最好
        val_total = vl.get("total", float("inf"))
        if val_total < best_val - 1e-9:
            best_val = val_total; best_ep = ep
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_total": val_total
            }, ckpt_path)
            print(f">>> Save best @ epoch {ep}: val_total={val_total:.5f} -> {ckpt_path}")

        torch.cuda.empty_cache()

    # ---- 存歷史/曲線 ----
    df_hist = pd.DataFrame(hist)
    csv_path = os.path.join(args.save_dir, f"{args.log_prefix}_{args.task}_history.csv")
    df_hist.to_csv(csv_path, index=False)
    print(f"Saved history -> {csv_path}")

    plt.figure(figsize=(10,5))
    plt.plot(df_hist["epoch"], df_hist["train_mse"], label="Train MSE")
    plt.plot(df_hist["epoch"], df_hist["val_mse"],   label="Val MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title(f"{args.task} - Train vs Val MSE")
    plt.grid(True); plt.legend()
    fig_path = os.path.join(args.save_dir, f"{args.log_prefix}_{args.task}_curves.png")
    plt.savefig(fig_path, dpi=300); plt.close()
    print(f"Saved curves -> {fig_path}")

    # ---- 測試 ----
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded best @ epoch={ckpt['epoch']}  val_total={ckpt['val_total']:.5f}")

    te = run_one_epoch(
        model, test_loader, optimizer=None, device=device,
        train=False, amp=False, use_regs=args.use_regs,
        lambd_intra=args.lambd_intra, gamma_attn=args.gamma_attn, tau_temp=args.tau_temp
    )
    print(f"[TEST {args.task}] total={te.get('total', float('nan')):.5f}  "
          f"mse={te.get('mse', float('nan')):.5f}  rmse={te.get('rmse', float('nan')):.5f}  "
          f"mae={te.get('mae', float('nan')):.5f}  smape={te.get('smape', float('nan')):.5f}")

    # ---- 每年 metrics ----
    year_results = evaluate_by_year(
        model, test_year_loaders, device,
        use_regs=args.use_regs, lambd_intra=args.lambd_intra, gamma_attn=args.gamma_attn, tau_temp=args.tau_temp
    )
    rows = []
    print(f"\n[TEST per-year — task={args.task}]")
    for y, m in year_results.items():
        rows.append({"year": y, "mse": m.get("mse", float("nan")),
                     "rmse": m.get("rmse", float("nan")),
                     "mae": m.get("mae", float("nan")),
                     "smape": m.get("smape", float("nan"))})
        print(f"Year {y}: mse={m.get('mse', float('nan')):.5f}  rmse={m.get('rmse', float('nan')):.5f}  "
              f"mae={m.get('mae', float('nan')):.5f}  smape={m.get('smape', float('nan')):.5f}")
    df_year = pd.DataFrame(rows).sort_values("year")
    per_year_csv = os.path.join(args.save_dir, f"{args.log_prefix}_{args.task}_per_year.csv")
    df_year.to_csv(per_year_csv, index=False)
    print(f"Saved per-year -> {per_year_csv}")

    # ---- 散點圖 (Output vs Label) ----
    model.eval()
    outs, tgts = [], []
    with torch.no_grad():
        for batch in test_loader:
            price   = batch["price"].to(device)
            finance = batch["finance"].to(device)
            event   = batch["event"].to(device)
            news    = batch.get("news", None)
            if news is not None and torch.is_tensor(news): news = news.to(device)
            out  = model(price, finance, event, news)
            pred = out["pred"]
            label= batch["label"].to(device)
            outs.append(pred.detach().cpu().view(-1))
            tgts.append(label.detach().cpu().view(-1))
    outs = torch.cat(outs).numpy()
    tgts = torch.cat(tgts).numpy()

    plt.figure(figsize=(6,6))
    plt.scatter(tgts, outs, alpha=0.5)
    plt.xlabel("True Label"); plt.ylabel("Prediction")
    plt.title(f"Scatter — {args.task}")
    mn, mx = min(outs.min(), tgts.min()), max(outs.max(), tgts.max())
    plt.plot([mn, mx], [mn, mx], "--")
    scatter_path = os.path.join(args.save_dir, f"{args.log_prefix}_{args.task}_scatter.png")
    plt.grid(True); plt.savefig(scatter_path, dpi=300); plt.close()
    print(f"Saved scatter -> {scatter_path}")

if __name__ == "__main__":
    main()
