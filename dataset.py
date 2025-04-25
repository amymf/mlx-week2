import torch
import pickle
from datasets import load_dataset
import pandas as pd

class MSMARCO(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
    ):
        self.triplets_df = load_dataset(f"amyf/ms-marco-triplets-{split}")[
            "train"  # index is train because data is already split
        ].to_pandas()
        self.encoded_queries = pickle.load(
            open(f"data/encoded_queries_{split}.pkl", "rb")
        )
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


class MSMARCO_RNN(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
    ):
        self.triplets_df = load_dataset(f"amyf/ms-marco-triplets-{split}")[
            "train"  # index is train because data is already split
        ].to_pandas()
        self.queries = pd.read_csv(f"data/queries_{split}.tsv", sep="\t")
        self.documents = pd.read_csv(f"data/documents_{split}.tsv", sep="\t")
        self.query_id_to_text = dict(
            zip(self.queries["query_id"], self.queries["query_text"])
        )
        self.doc_id_to_text = dict(
            zip(self.documents["doc_id"], self.documents["doc_text"])
        )

    def __len__(self):
        return len(self.triplets_df)

    def __getitem__(self, idx):
        triplet = self.triplets_df.iloc[idx]
        query_id = triplet["query_id"]
        pos_doc_id = triplet["pos_doc_id"]
        neg_doc_id = triplet["neg_doc_id"]

        query = self.query_id_to_text[query_id]
        pos_doc = self.doc_id_to_text[pos_doc_id]
        neg_doc = self.doc_id_to_text[neg_doc_id]

        return {
            "query": query,
            "pos_doc": pos_doc,
            "neg_doc": neg_doc,
        }
