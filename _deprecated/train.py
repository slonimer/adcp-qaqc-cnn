# train.py
import argparse
import torch
from torch.utils.data import DataLoader
from dataset_loader import ADCPDataset
from model import TemporalCNN
from utils import (
    train_model,
    compute_f1,
    combined_loss,
    get_class_weights,
    seed_everything,
)
import wandb

def main(args):
    seed_everything(42)

    # Load dataset
    train_set = ADCPDataset(args.train_dir)
    val_set = ADCPDataset(args.val_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Model + Loss
    model = TemporalCNN(input_channels=3, num_classes=args.num_classes)
    class_weights = get_class_weights(train_set)
    loss_fn = combined_loss(class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # wandb init
    wandb.init(project="adcp-anomaly-detection", config=vars(args))

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs=args.epochs,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=4)
    args = parser.parse_args()

    if args.train:
        main(args)
