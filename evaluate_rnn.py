import torch
import pandas as pd
import torch.nn.functional as F
import model
import pickle
from tqdm import tqdm
import tokeniser 
from typing import List, Dict, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_to_int = pickle.load(open("vocab_to_int.pkl", "rb"))

# Load your document data
docs_df = pd.read_csv("data/all_documents.tsv", sep="\t")

max_length = 20

# Initialize towers
vocab_size = len(vocab_to_int)
query_tower = model.RNNTower(vocab_size=vocab_size, embed_dim=128, hidden_dim=128).to(device)
doc_tower = model.RNNTower(vocab_size=vocab_size, embed_dim=128, hidden_dim=128).to(device)

query_tower.load_state_dict(torch.load("models/rnn_query_tower.pt"))
doc_tower.load_state_dict(torch.load("models/rnn_doc_tower.pt"))

# Set towers to evaluation mode
query_tower.eval()
doc_tower.eval()

def tokenize_and_pad(texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    
    tokenized = [tokeniser.tokenize(t, vocab_to_int) for t in texts]
    tokenized = [seq[:max_length] for seq in tokenized]
    lengths = torch.tensor([len(seq) for seq in tokenized])
    padded = tokeniser.pad_sequences(tokenized, max_length=max_length)
    tensor = torch.tensor(padded).long().to(device)
    return tensor, lengths


def evaluate(query, top_k=5):
    
    tokenized_query = tokeniser.tokenize(query, vocab_to_int)
    query_length = torch.tensor([len(tokenized_query)]).cpu()
    padded_query = torch.tensor(tokeniser.pad_sequences([tokenized_query], max_length=max_length)).long().to(device)
    
    query_embedding = query_tower(padded_query, query_length)

    doc_embeddings = pickle.load(open("data/encoded_documents_test.pkl", "rb"))

    # Compute cosine similarity
    similarities = F.cosine_similarity(query_embedding.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2).squeeze(0)    
    print(similarities)
    topk_scores, topk_indices = torch.topk(similarities, top_k)
    return topk_indices


query = "what is the the capital of argentina"
topk_indices = evaluate(query, top_k=5)

for doc in topk_indices:
    print(doc.item())
    text = docs_df.iloc[doc.item()]["doc_text"]
    print(f"Document ID: {doc.item()}, Text: {text[:100]}...")