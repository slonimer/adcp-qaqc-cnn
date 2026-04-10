import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs=20,
    scheduler=None,
    eval_metric_fn=compute_f1,
):
    model.to(device)

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        early_stopping = EarlyStopping(patience=5, save_path="best_model.pt")

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(torch.argmax(logits, dim=-1))
            train_targets.append(y)

        # Compute training metrics
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_f1 = eval_metric_fn(train_targets, train_preds)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                val_loss += loss.item()
                val_preds.append(torch.argmax(logits, dim=-1))
                val_targets.append(y)

        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_f1 = eval_metric_fn(val_targets, val_preds)

        if scheduler:
            scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss / len(train_loader):.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss / len(val_loader):.4f} | F1: {val_f1:.4f}")
        
        
        # End of epoch logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_f1": train_f1,
            "val_f1": val_f1,
        })

        early_stopping(val_loss / len(val_loader), model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
