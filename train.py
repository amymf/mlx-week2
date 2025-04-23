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

validate_dataset = MSMARCO(split="validation")
validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)

query_tower = model.QueryTower()
doc_tower = model.DocumentTower()

query_tower.train()
doc_tower.train()

optimizer = torch.optim.Adam(
    list(query_tower.parameters()) + list(doc_tower.parameters()), lr=0.001
)

num_epochs = 24
margin = torch.tensor(0.2)
for epoch in range(num_epochs):
    # Training step
    query_tower.train()
    doc_tower.train()
    train_loss = 0.0
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

        loss = F.relu(margin - (dist_pos - dist_neg)).mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        wandb.log({"train/batch_loss": loss.item(), "epoch": epoch + 1})

    # Validation step
    query_tower.eval()
    doc_tower.eval()
    validate_loss = 0.0
    with torch.no_grad():
        for batch in validate_loader:
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

            loss = F.relu(margin - (dist_pos - dist_neg)).mean()
            validate_loss += loss.item()
            wandb.log({"validate/batch_loss": loss.item(), "epoch": epoch + 1})

    avg_train_loss = train_loss / len(train_loader)
    avg_validate_loss = validate_loss / len(validate_loader)
    wandb.log(
        {
            "train/epoch_loss": avg_train_loss,
            "validate/epoch_loss": avg_validate_loss,
            "epoch": epoch + 1,
        }
    )
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validate Loss: {avg_validate_loss:.4f}")

# Save the model
torch.save(query_tower.state_dict(), "models/query_tower.pt")
torch.save(doc_tower.state_dict(), "models/doc_tower.pt")
