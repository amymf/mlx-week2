import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import model
from dataset import MSMARCO
import wandb

wandb.init(
    project="msmarco-dual-encoder",
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "margin": 0.2,
    },
)


train_dataset = MSMARCO(split="train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

query_tower = model.QueryTower()
doc_tower = model.DocumentTower()

query_tower.load_state_dict(torch.load("query_tower.pt"))
doc_tower.load_state_dict(torch.load("doc_tower.pt"))

query_tower.train()
doc_tower.train()

optimizer = torch.optim.Adam(
    list(query_tower.parameters()) + list(doc_tower.parameters()), lr=0.001
)

num_epochs = 20
margin = torch.tensor(0.2)
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()

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

        dist_diff = dist_pos - dist_neg

        loss = F.relu(margin - (dist_pos - dist_neg)).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1})

    avg_loss = total_loss / len(train_loader)
    wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the model
torch.save(query_tower.state_dict(), "query_tower.pt")
torch.save(doc_tower.state_dict(), "doc_tower.pt")
