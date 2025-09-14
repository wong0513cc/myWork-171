from torch.utils.data import DataLoader, Subset
from collections import Counter
from dataset import ESGDataset

def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=False
):
    """
    建立一個 DataLoader，支援自訂參數

    Args:
        dataset: PyTorch Dataset 物件
        batch_size (int): 每個 batch 的樣本數
        shuffle (bool): 是否在每個 epoch 打亂樣本
        drop_last (bool): 如果資料不是 batch_size 整數倍，是否丟掉最後一個不足的 batch
        num_workers (int): DataLoader 使用的子程序數量（0 表示在主程序載入）
        pin_memory (bool): 是否把張量放到固定記憶體，加快 GPU 傳輸

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def split_dataset_by_year(dataset, years_train, years_val, years_test):
    """
    依年份切分 dataset，回傳 train/val/test Subset
    """
    idx_train = [i for i, s in enumerate(dataset.samples) if s["year"] in years_train]
    idx_val   = [i for i, s in enumerate(dataset.samples) if s["year"] in years_val]
    idx_test  = [i for i, s in enumerate(dataset.samples) if s["year"] in years_test]

    train_set = Subset(dataset, idx_train)
    val_set   = Subset(dataset, idx_val)
    test_set  = Subset(dataset, idx_test)

    return train_set, val_set, test_set


def show_year_distribution(dataset):
    """
    印出每年樣本數，方便 debug
    """
    cnt = Counter([s["year"] for s in dataset.samples])
    for year in sorted(cnt.keys()):
        print(f"{year}: {cnt[year]} 筆樣本")
    print("總樣本數:", len(dataset))



dataset = ESGDataset(
    root_price="/home/sally/dataset/data_preprocessing/price_data",
    root_finance="/home/sally/dataset/data_preprocessing/financial",
    root_news="/home/sally/dataset/data_preprocessing/news/monthly_embeddings",
    root_event="root_event",
    root_label="/home/sally/dataset/data_preprocessing/esg_label/esg_npy",
    root_year_symbols="/home/sally/dataset/ticker/nyse/yearly_symbol",
    years=range(2015, 2025),
    has_label=True,
    label_mode="S",       
)


print(dataset[0])
show_year_distribution(dataset)