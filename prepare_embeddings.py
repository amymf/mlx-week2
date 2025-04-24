from tqdm import tqdm
import faiss
import pandas as pd
import torch
import numpy as np
from model import DocumentTower
from encoder import sentence_to_w2v

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load & combine all document splits
splits = ["train", "validation", "test"]
dfs = [pd.read_csv(f"data/documents_{split}.tsv", sep="\t") for split in splits]
all_docs_df = pd.concat(dfs).drop_duplicates(subset="doc_text").reset_index(drop=True)
all_docs_df["doc_id"] = all_docs_df.index.astype(str)  # Reassign unique IDs

# Load trained document tower
doc_tower = DocumentTower()
doc_tower.load_state_dict(torch.load("models/doc_tower.pt", map_location=device))
doc_tower = doc_tower.to(device)
doc_tower.eval()

# Generate embeddings
embeddings = []
doc_ids = []

for _, row in tqdm(all_docs_df.iterrows(), total=len(all_docs_df)):
    doc_embedding = sentence_to_w2v(row["doc_text"])
    doc_embedding = torch.tensor(doc_embedding, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        doc_embedding = doc_tower(doc_embedding).squeeze().cpu().numpy()

    embeddings.append(doc_embedding)
    doc_ids.append(row["doc_id"])

embeddings = np.vstack(embeddings)

# Save index
index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine similarity
faiss.normalize_L2(embeddings)
index.add(embeddings)

faiss.write_index(index, "data/document_index.faiss")
all_docs_df.to_csv("data/all_documents.tsv", sep="\t", index=False)
