from dataset import ESGDataset

dataset = ESGDataset(
    root_price   = "/home/sally/dataset/data_preprocessing/price_data",
    root_finance = "/home/sally/dataset/data_preprocessing/financial",
    root_news    = "/home/sally/dataset/data_preprocessing/news/monthly_embeddings",
    root_event   = "/home/sally/dataset/data_preprocessing/event_type_PCA",
    root_label   = "/home/sally/dataset/data_preprocessing/esg_label/esg_npy",
    root_year_symbols = "/home/sally/dataset/ticker/nyse/yearly_symbol",
    years = range(2015, 2025),
    has_label = True,
    label_mode = "all"
)

sample = dataset[9112]
print(sample["price"].shape)   # (12, d_p)
print(sample["finance"].shape)
print(sample["news"].shape)
print(sample["event"].shape)
print(sample["label"])       # tensor([0.xx])
print(sample["price"])
print(sample["symbol"])
print(sample["year"])
print(sample["label"])
print()
# print(f"price: {sample['price']}")
# print(f"finance: {sample['finance']}")
# print(f"news: {sample['news']}")
# print(f"event: {sample['event']}")

"""
之後可以考慮event的前4個月補0
"""