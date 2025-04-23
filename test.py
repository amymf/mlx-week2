import torch
from torch.utils.data import DataLoader
import wandb
import model
from dataset import MSMARCO

wandb.init(project="msmarco-dual-encoder")

test_dataset = MSMARCO(split="test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

query_tower = model.QueryTower()
doc_tower = model.DocumentTower()

query_tower.load_state_dict(torch.load("query_tower.pt"))
doc_tower.load_state_dict(torch.load("doc_tower.pt"))

query_tower.eval()
doc_tower.eval()

margin = torch.tensor(0.2)
test_loss = 0.0
num_correct = 0
num_total = 0

for batch in test_loader:

    query = batch["query"].float()
    pos_doc = batch["pos_doc"].float()
    neg_doc = batch["neg_doc"].float()

    query = query_tower(query)
    pos_doc = doc_tower(pos_doc)
    neg_doc = doc_tower(neg_doc)

    # pylint: disable=not-callable
    dist_pos = torch.nn.functional.cosine_similarity(query, pos_doc, dim=1)
    dist_neg = torch.nn.functional.cosine_similarity(query, neg_doc, dim=1)
    # pylint: enable=not-callable

    loss = torch.nn.functional.relu(margin - (dist_pos - dist_neg)).mean()

    test_loss += loss.item()

    # Calculate accuracy
    correct = (dist_pos > dist_neg).sum().item()
    total = query.size(0)
    num_correct += correct
    num_total += total


avg_loss = test_loss / len(test_loader)
wandb.log({"test/epoch_loss": avg_loss})
print(f"Test Loss: {avg_loss:.4f}")

accuracy = num_correct / num_total
wandb.log({"test/accuracy": accuracy})
print(f"Test Accuracy: {accuracy:.4f}")
