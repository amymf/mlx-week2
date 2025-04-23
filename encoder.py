import pickle
from gensim.utils import simple_preprocess
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

def sentence_to_w2v(sentence):
    tokens = simple_preprocess(sentence)
    vectors = [model[w] for w in tokens if w in model]
    if not vectors:
        return np.zeros(300)
    return np.mean(vectors, axis=0)


queries_df = (
    pd.read_csv("queries.tsv", sep="\t")
)
docs_df = (
    pd.read_csv("documents.tsv", sep="\t")
)

# Generate query embeddings
encoded_queries = {
    row["query_id"]: sentence_to_w2v(row["query_text"])
    for _, row in queries_df.iterrows()
}

# Generate document embeddings
encoded_documents = {
    row["doc_id"]: sentence_to_w2v(row["doc_text"]) for _, row in docs_df.iterrows()
}

with open("encoded_queries.pkl", "wb") as f:
    pickle.dump(encoded_queries, f)

with open("encoded_documents.pkl", "wb") as f:
    pickle.dump(encoded_documents, f)

# import torch 
# from model import QueryTower, DocumentTower

# if __name__ == "__main__":
#     # Test the embedding functions
#     test_query = "Who wrote An Inspector Calls?"
#     test_doc = "An Inspector Calls is a play written by J.B. Priestley."

#     query_tower = QueryTower()
#     doc_tower = DocumentTower()
#     query_tower.eval()
#     doc_tower.eval()

#     query_embedding = sentence_to_w2v(test_query)
#     doc_embedding = sentence_to_w2v(test_doc)
#     query_embedding = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
#     doc_embedding = torch.tensor(doc_embedding, dtype=torch.float32).unsqueeze(0)
#     # query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
#     # doc_embedding = torch.nn.functional.normalize(doc_embedding, p=2, dim=1)
#     query_embedding = query_tower(query_embedding)
#     doc_embedding = doc_tower(doc_embedding)
#     # pylint: disable=not-callable
#     cosine_similarity = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding, dim=1)

#     # print("Query embedding:", query_embedding)
#     # print("Document embedding:", doc_embedding)
#     print("Cosine similarity:", cosine_similarity.item())
