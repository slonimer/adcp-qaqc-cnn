#!/usr/bin/env python3
"""
train_adcp_model.py

Train and optionally evaluate an ADCP anomaly detection model using a temporal ResNet-based classifier.

UNTESTED = WORK IN PROGRESS

Dependencies:
    - PyTorch
    - scikit-learn
    - wandb (optional)
    - Your local modules: dataset_loader, resnet_temporal, utils
"""

import os
from types import SimpleNamespace
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dataset_loader import ADCPDataset
from resnet_temporal import ResNetTemporalClassifier
from utils import seed_everything, get_class_weights, combined_loss, train_model

# ------------------------------
# Global settings
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED_NUM = 42
seed_everything(SEED_NUM)

# Hard-coded paths
DATA_FOLDER = "/scratch/slonimer/ML_ADCP/BACAX_24hr_h5/"
ANNOTATED_FILE = "annotated_files.txt"
BEST_MODEL_PATH = "best_model.pt"


def train_adcp_model(
    data_folder: str = DATA_FOLDER,
    anomaly_list_path: str = ANNOTATED_FILE,
    resnet_variant: str = "resnet50",
    epochs: int = 100,
    batch_size: int = 16,
    use_wandb: bool = True,
    debug_mode: bool = False,
    reload_best_model: bool = False,
) -> Optional[ResNetTemporalClassifier]:
    """
    Train an ADCP anomaly detection model.

    Parameters
    ----------
    data_folder : str
        Directory containing .h5 ADCP files.
    anomaly_list_path : str
        Path to text file listing anomaly-labeled files.
    resnet_variant : str
        ResNet variant to use: 'resnet18', 'resnet34', or 'resnet50'.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    use_wandb : bool
        Whether to log training with WandB.
    debug_mode : bool
        If True, limit dataset size and disable some optimizations for debugging.
    reload_best_model : bool
        If True, load best saved model and evaluate on test set after training.

    Returns
    -------
    model : Optional[ResNetTemporalClassifier]
        Trained model. If `reload_best_model` is True and model is reloaded, returns evaluated model.
    """
    # ------------------------------
    # WandB setup
    # ------------------------------
    if use_wandb:
        import wandb

        wandb.init(
            project="adcp-anomaly-detection",
            config={
                "model": "ResNetTemporalClassifier",
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": 1e-3,
                "loss_alpha": 0.5,
                "optimizer": "Adam",
            },
        )
        config = wandb.config
    else:
        config = SimpleNamespace(
            model="ResNetTemporalClassifier",
            epochs=epochs,
            batch_size=batch_size,
            lr=1e-3,
            loss_alpha=0.5,
            optimizer="Adam",
        )

    # ------------------------------
    # Load dataset
    # ------------------------------
    file_list = os.listdir(data_folder)
    h5_files = sorted([f for f in file_list if os.path.splitext(f)[1] == ".h5"])
    h5_paths = [os.path.join(data_folder, f) for f in h5_files]

    # Load anomaly files
    with open(anomaly_list_path, "r") as f:
        anomaly_files = set(line.strip() for line in f if line.strip())

    anomaly_paths = [p for p in h5_paths if os.path.basename(p) in anomaly_files]

    if debug_mode:
        # Reduce dataset for debugging
        h5_paths = h5_paths[:200]

    normal_paths = [p for p in h5_paths if os.path.basename(p) not in anomaly_files]

    # Split anomaly and normal files into train/val/test
    an_train, an_temp = train_test_split(
        anomaly_paths, test_size=0.3, random_state=SEED_NUM
    )
    an_val, an_test = train_test_split(an_temp, test_size=0.33, random_state=SEED_NUM)

    n_train, n_temp = train_test_split(
        h5_paths, test_size=0.3, random_state=SEED_NUM
    )
    n_val, n_test = train_test_split(n_temp, test_size=0.33, random_state=SEED_NUM)

    train_files = an_train + n_train
    val_files = an_val + n_val
    test_files = an_test + n_test

    # Create datasets
    train_dataset = ADCPDataset(train_files)
    val_dataset = ADCPDataset(val_files)
    test_dataset = ADCPDataset(test_files)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    print("Datasets loaded: train/val/test")

    # ------------------------------
    # Initialize model, loss, optimizer
    # ------------------------------
    num_classes = 6
    model = ResNetTemporalClassifier(
        num_classes=num_classes,
        pretrained=True,
        variant=resnet_variant,
        resize=(224, 224),
    ).to(DEVICE)

    class_weights = get_class_weights(train_dataset, num_classes)
    print(f"Class weights: {class_weights}")

    loss_fn = combined_loss(class_weights, alpha=config.loss_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # ------------------------------
    # Optional debug environment tweaks
    # ------------------------------
    if debug_mode:
        os.environ["TORCH_DISABLE_MKL"] = "1"
        os.environ["ONEDNN_VERBOSE"] = "0"
        os.environ["DNNL_VERBOSE"] = "0"
        torch.backends.mkldnn.enabled = False

    # ------------------------------
    # Train the model
    # ------------------------------
    if use_wandb:
        model = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            DEVICE,
            num_epochs=config.epochs,
            patience=5,
            USE_WANDB=use_wandb,
        )
    else:
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            DEVICE,
            num_epochs=config.epochs,
            patience=5,
            USE_WANDB=use_wandb,
        )

    # ------------------------------
    # Optional reload and evaluate
    # ------------------------------
    if reload_best_model and os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.eval()

        all_preds = []
        all_labels = []

        for x, y in test_loader:
            x = x.to(DEVICE)
            with torch.no_grad():
                out = model(x)
                out = out.reshape(-1, out.shape[-1])
                preds = torch.argmax(out, dim=1)

            y = y.view(-1)
            all_preds.append(preds.cpu())
            all_labels.append(y)

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        print("Classification Report on Test Set:")
        print(classification_report(y_true, y_pred))

    return model


# ------------------------------
# Main block for CLI execution
# ------------------------------
if __name__ == "__main__":
    #Example
    # python train_adcp_model.py --epochs 50 --batch_size 8 --resnet 34
    
    import argparse

    parser = argparse.ArgumentParser(description="Train ADCP anomaly detection model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--resnet",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
        help="ResNet variant to use",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--reload_best", action="store_true", help="Reload and evaluate best model"
    )

    args = parser.parse_args()

    train_adcp_model(
        resnet_variant=args.resnet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_wandb=args.wandb,
        debug_mode=args.debug,
        reload_best_model=args.reload_best,
    )
