import torch
import pickle
from datasets import load_dataset


class MSMARCO(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
    ):
        self.triplets_df = load_dataset(f"amyf/ms-marco-triplets-{split}")[
            "train"  # index is train because data is already split
        ].to_pandas()
        self.encoded_queries = pickle.load(open(f"data/encoded_queries_{split}.pkl", "rb"))
        self.encoded_documents = pickle.load(
            open(f"data/encoded_documents_{split}.pkl", "rb")
        )

    def __len__(self):
        return len(self.triplets_df)

    def __getitem__(self, idx):
        triplet = self.triplets_df.iloc[idx]
        query_id = triplet["query_id"]
        pos_doc_id = triplet["pos_doc_id"]
        neg_doc_id = triplet["neg_doc_id"]

        query = self.encoded_queries[query_id]
        pos_doc = self.encoded_documents[pos_doc_id]
        neg_doc = self.encoded_documents[neg_doc_id]

        return {
            "query": query,
            "pos_doc": pos_doc,
            "neg_doc": neg_doc,
        }
