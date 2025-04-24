import faiss
import torch
import pandas as pd
from model import QueryTower
from encoder import sentence_to_w2v

# N.B. this script sometimes crashes and works better if using faiss-gpu

# Load document data
index = faiss.read_index("data/document_index.faiss")
all_docs_df = pd.read_csv("data/all_documents.tsv", sep="\t")

# Initialize tower
query_tower = QueryTower()
query_tower.load_state_dict(torch.load("models/query_tower.pt"))
query_tower.eval()

def evaluate(query, top_k=5):
    query_embedding = sentence_to_w2v(query)
    query_embedding = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
    query_embedding = query_tower(query_embedding)  # normalized

    query_embedding_np = query_embedding.detach().cpu().numpy().astype("float32")
    _, I = index.search(query_embedding_np.reshape(1, -1), k=top_k)
    return all_docs_df.iloc[I[0]]


query = "What does DNA stand for?"
top_k_docs = evaluate(query, top_k=5)

for _, row in top_k_docs.iterrows():
    doc = row["doc_id"]
    doc_text = row["doc_text"]
    print(f"Document ID: {doc}, Text: {doc_text[:100]}...")
