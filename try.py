from model import MGMMultiModel
from dataset_v2 import GraphESGDataset
# 假設：
# price:   torch.FloatTensor [B,K,N,4]
# finance: torch.FloatTensor [B,K,N,26]
# event:   torch.FloatTensor [B,K,N,64]
# news:    torch.FloatTensor [B,K,N,100]
# label:   torch.FloatTensor [B,1,N,1]  (年度) 或 [B,K,N,1] (月度)

model = MGMMultiModel(
    D=128,
    use_event_ffn=False,        # True 則 Event 用 FFN encoder
    global_pool='attn',         # 'attn' or 'mean'
    fusion='gated',             # 目前提供 gated 融合
    loss_kind='mse',            # 'mse' / 'mae' / 'smape'
    lambda_uni=0.5,
    lambda_icl=0.1,
    lambda_temp=0.2,
    contrastive_temp=0.07,
    object_window=3,
    object_stride=1,
    coattn_rounds=1,
    predict_yearly=True         # 年度預測 -> heads 會做 K 上的 pooling
)

root_paths = {
    "price": "/home/sally/dataset/data_preprocessing/price_percentage",
    "finance": "/home/sally/dataset/data_preprocessing/financial",
    "news": "/home/sally/dataset/data_preprocessing/news/monthly_embeddings",
    "event": "/home/sally/dataset/data_preprocessing/event_type_PCA",
    "graph": "/home/sally/dataset/gkg_data/monthly_graph_new",
    "label": "/home/sally/dataset/data_preprocessing/esg_label/esg_npy",
    "year_symbols": "/home/sally/dataset/ticker/nyse/yearly_symbol"
}

sample_ds = GraphESGDataset(
    root_price=root_paths["price"],
    root_finance=root_paths["finance"],
    root_news=root_paths["news"],
    root_event=root_paths["event"],
    root_graph=root_paths["graph"],
    root_label=root_paths["label"],
    root_year_symbols=root_paths["year_symbols"],
    years=range(2015, 2025),
    has_label=True,
)

out = model(price, finance, event, news, label=label, contrastive_t=None)
pred_multi = out["pred_multi"]         # [B,1,N,1] if yearly
pred_uni   = out["pred_uni"]           # dict of [B,1,N,1]
losses     = out.get("losses", None)   # 包含 total / 各項