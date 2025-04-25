import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import model
from dataset import MSMARCO_RNN
import wandb
import tokeniser

wandb.init(
    project="msmarco-dual-encoder-rnn",
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "margin": 0.2,
    },
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = MSMARCO_RNN(split="train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

validate_dataset = MSMARCO_RNN(split="validation")
validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)

vocab_size = len(pickle.load(open("vocab_to_int.pkl", "rb")))
query_tower = model.RNNTower(vocab_size=vocab_size, embed_dim=128, hidden_dim=128).to(device)
doc_tower = model.RNNTower(vocab_size=vocab_size, embed_dim=128, hidden_dim=128).to(device)

query_tower.train()
doc_tower.train()

optimizer = torch.optim.Adam(
    list(query_tower.parameters()) + list(doc_tower.parameters()), lr=0.001
)

vocab_to_int = pickle.load(open("vocab_to_int.pkl", "rb"))

margin = torch.tensor(0.2).to(device)

def process_batch(batch, max_length=20):
    def tokenize_and_pad(texts):
        tokenized = [tokeniser.tokenize(t, vocab_to_int) for t in texts]
        tokenized = [seq[:max_length] for seq in tokenized]
        lengths = torch.tensor([len(seq) for seq in tokenized])
        padded = tokeniser.pad_sequences(tokenized, max_length=max_length)
        tensor = torch.tensor(padded).long().to(device)
        return tensor, lengths

    query, query_lengths = tokenize_and_pad(batch["query"])
    pos_doc, pos_doc_lengths = tokenize_and_pad(batch["pos_doc"])
    neg_doc, neg_doc_lengths = tokenize_and_pad(batch["neg_doc"])

    query = query.to(device)
    pos_doc = pos_doc.to(device)
    neg_doc = neg_doc.to(device)
    query_lengths = query_lengths.cpu()
    pos_doc_lengths = pos_doc_lengths.cpu()
    neg_doc_lengths = neg_doc_lengths.cpu()
    query_vec = query_tower(query, query_lengths)
    pos_vec = doc_tower(pos_doc, pos_doc_lengths)
    neg_vec = doc_tower(neg_doc, neg_doc_lengths)

    # pylint: disable=not-callable
    dist_pos = F.cosine_similarity(query_vec, pos_vec, dim=1)
    dist_neg = F.cosine_similarity(query_vec, neg_vec, dim=1)
    # pylint: enable=not-callable

    return F.relu(margin - (dist_pos - dist_neg)).mean()


num_epochs = 20
for epoch in range(num_epochs):

    # Training step
    query_tower.train()
    doc_tower.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        loss = process_batch(batch).to(device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        wandb.log({"train/epoch_loss": loss.item()})

    # Validation step
    query_tower.eval()
    doc_tower.eval()
    validate_loss = 0.0
    with torch.no_grad():
        for batch in validate_loader:
            loss = process_batch(batch).to(device)
            validate_loss += loss.item()
            wandb.log({"validate/batch_loss": loss.item()})

    avg_train_loss = train_loss / len(train_loader)
    avg_validate_loss = validate_loss / len(validate_loader)
    wandb.log(
        {
            "train/epoch_loss": avg_train_loss,
            "validate/epoch_loss": avg_validate_loss,
            "epoch": epoch + 1,
        }
    )
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validate Loss: {avg_validate_loss:.4f}"
    )

# Save the model
torch.save(query_tower.state_dict(), "models/rnn_query_tower.pt")
torch.save(doc_tower.state_dict(), "models/rnn_doc_tower.pt")
