import os
import torch
import numpy as np
import random
from torch.nn import functional as F
from sklearn.metrics import f1_score
#import wandb

# 1. Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 2. Compute class weights for imbalance
def get_class_weights(dataset, num_classes):
    from collections import Counter
    label_counts = Counter()

    #for _, labels in dataset:   #From when just had x,y
    for _, labels, _ in dataset: #Updated for x,y,meta
        label_counts.update(labels.tolist())

    total = sum(label_counts.values())
    #num_classes = max(label_counts.keys()) + 1
    weights = [0] * num_classes
    for cls in range(num_classes):
        #count = label_counts[cls] if cls in label_counts else 1
        #weights[cls] = total / (num_classes * count)
        #Note: num_classes is in the denominator to help normalize the weights
        count = label_counts[cls]
        if count>0:
            weights[cls] = total / (num_classes * count)
        else: 
            weights[cls] = 0
          
    return torch.tensor(weights, dtype=torch.float)

# 3. Combined loss: CrossEntropy + Dice
def combined_loss(class_weights, alpha=0.5):
    #loss_fn is produced by: loss_fn = combined_los(...)
    #This is so that it is called BEFORE "train_model", so that weights are set in place and dont need to be included in the train_model loop

    def loss_fn(outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, weight=class_weights.to(outputs.device))
        
        # Dice loss (for multi-class, softmaxed)
        probs = F.softmax(outputs, dim=1)
        #print(targets.shape) #576
        #print(outputs.shape[1]) #6
        
        do_vanilla_diceloss = 0
        if do_vanilla_diceloss:
             # One-hot encode targets
            targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).float()  # Shape: (B*T, C)
            #targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 2, 1).float()
            
            intersection = (probs * targets_one_hot).sum(dim=0) # sum(dim=2)
            union = probs.sum(dim=0) + targets_one_hot.sum(dim=0) # sum(dim=2)
            dice_loss = 1 - (2. * intersection / (union + 1e-8)).mean()
        else:
            #Do Graduated diceloss
            targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).float().to(outputs.device)  # (B*T, C)

            # Compute numerator and denominator for GDL
            intersection = (probs * targets_one_hot).sum(dim=0)  # sum over batch*time
            probs_sq = probs.pow(2).sum(dim=0)
            targets_sq = targets_one_hot.pow(2).sum(dim=0)
            denominator = probs_sq + targets_sq

            # Class weights (same as used for CE), normalize to sum to 1 for stability
            class_weights_norm = class_weights.to(outputs.device)
            class_weights_norm = class_weights_norm / class_weights_norm.sum()

            # Graduated (Generalized) Dice Loss
            dice_score = 2 * intersection / (denominator + 1e-8)
            dice_loss = 1 - (class_weights_norm * dice_score).sum()

        #Return the combines losses
        return alpha * ce_loss + (1 - alpha) * dice_loss
    return loss_fn

# 4. Training loop
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=20, patience=5, USE_WANDB=0, best_model_path = "best_model.pt"):
    model = model.to(device)
    best_f1 = 0
    #best_model_path = "best_model.pt"
    patience_counter = 0

    #USE_WANDB = 1   
    if not(USE_WANDB):
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": []
        }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        n = 1 #For debugging
        print('Starting Training on Epoch # ',epoch)

        for x_batch, y_batch, _ in train_loader:
            #print("x_batch shape:", x_batch.shape)
            #print(x_batch.dtype)
            
            x_batch = x_batch.to(device) # (B, Channels, T, range)
            y_batch = y_batch.to(device) # (B, T)

            optimizer.zero_grad()
            outputs = model(x_batch)     # (B, T, num_classes)
            #print("output shape:", outputs.shape)
            #print("y shape:", y_batch.shape)

            #Need to reshape the outputs, and y_batch for loss_fn to work properly:
            outputs = outputs.reshape(-1, outputs.shape[-1])  # (B*T, num_classes)
            y_batch = y_batch.view(-1)                        # (B*T, )

            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu())
            all_targets.append(y_batch.cpu())

            print('Loss (& total) on Batch #{}: {} ({})'.format(n, loss, total_loss))
            n+=1

        train_loss = total_loss / len(train_loader) # #Calculate average loss
        train_f1 = f1_score(torch.cat(all_targets), torch.cat(all_preds), average='macro')

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        print('Starting Validation on Epoch # ',epoch)
        with torch.no_grad():
            for x_val, y_val, _ in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                outputs = model(x_val)
                #Need to reshape the outputs, and y_batch for loss_fn to work properly:
                outputs = outputs.reshape(-1, outputs.shape[-1])  # (B*T, num_classes)
                y_val = y_val.view(-1)                        # (B*T, )

                loss = loss_fn(outputs, y_val)
                #val_loss += loss.item()
                val_loss += loss.item() * x_val.size(0)  # Total loss for this batch - Needed when using small number of examples, and last batch size might not match

                preds = torch.argmax(outputs, dim=1)
                val_preds.append(preds.cpu())
                val_targets.append(y_val.cpu())

        #Calculate average loss
        val_loss /= len(val_loader.dataset) # Total loss for this batch - Needed when using small number of examples, and last batch size might not match
        #val_loss /= len(val_loader)
        val_f1 = f1_score(torch.cat(val_targets), torch.cat(val_preds), average='macro')

        if USE_WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_f1": val_f1
            })
        else:
            # Save to history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            #"unwrap" model if run on parallel GPUs, makes loading easier
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), best_model_path)
            #torch.save(model.state_dict(), best_model_path)
            best_f1 = val_f1
            patience_counter = 0
            print("✅ New best model saved.")
        else:
            #If the validation loss isn't improving for 5 epochs, quit early
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    if USE_WANDB:
        return model
    else:
        return model, history
