import torch
import pandas as pd
import torch.nn.functional as F
from model import QueryTower, DocumentTower
from encoder import sentence_to_w2v
import pickle
from tqdm import tqdm

# Load your document data
docs_df = pd.read_csv("data/documents_test.tsv", sep="\t")

# Initialize towers
query_tower = QueryTower()
doc_tower = DocumentTower()

query_tower.load_state_dict(torch.load("models/query_tower.pt"))
doc_tower.load_state_dict(torch.load("models/doc_tower.pt"))

# Set towers to evaluation mode
query_tower.eval()
doc_tower.eval()

with open("data/encoded_documents_test.pkl", "rb") as f:
    encoded_documents = pickle.load(f)


def evaluate(query, top_k=5):
    query_embedding = sentence_to_w2v(query)
    query_embedding = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
    query_embedding = query_tower(query_embedding) # normalized

    # Calculate cosine similarity between query and all documents
    similarities = {}
    for doc_id, doc_embedding in tqdm(
        encoded_documents.items(), total=len(encoded_documents)
    ):
        doc_embedding = torch.tensor(doc_embedding, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            doc_embedding = doc_tower(doc_embedding)  # normalized

        # pylint: disable=not-callable
        similarity = F.cosine_similarity(query_embedding, doc_embedding, dim=1)
        # pylint: enable=not-callable
        similarities[doc_id] = similarity.item()

    # Sort documents by similarity and get the top_k
    ranked_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return ranked_docs


query = "What is the largest mammal?"
top_k_docs = evaluate(query, top_k=5)

for doc, sim in top_k_docs:
    doc_text = docs_df.loc[docs_df["doc_id"] == doc, "doc_text"].values[0]
    print(f"Document ID: {doc}, Similarity: {sim:4f}, Text: {doc_text[:100]}...")
