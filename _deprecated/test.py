# test.py
import argparse
import torch
from torch.utils.data import DataLoader
from dataset_loader import ADCPDataset
from model import TemporalCNN
from utils import compute_f1
import numpy as np

def run_inference(model, dataloader, device):
    model.eval()
    model.to(device)
    predictions = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            predictions.append(preds.cpu())
            all_labels.append(y)

    return torch.cat(predictions), torch.cat(all_labels)

def main(args):
    dataset = ADCPDataset(args.test_dir)
    loader = DataLoader(dataset_loader, batch_size=1)

    model = TemporalCNN(input_channels=3, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint))
    preds, targets = run_inference(model, loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"Test F1 Score: {compute_f1(targets, preds):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run classification on test data")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--num_classes", type=int, default=4)
    args = parser.parse_args()

    if args.test:
        main(args)
