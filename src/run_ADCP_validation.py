#Kernel: torch_env25

#Session Settings on JupyterHub
#Memory: 32000
#Cores: 4
#GPUs: 1x 20 or 2x 10

#if DRAC is being buggy, try from ssh:
# $ salloc --account=def-kmoran --time=02:00:00 --mem=16G --cpus-per-task=4 --gres=gpu:h100:1


# Cell 1: Imports & Setup

import os
import sys 
import torch
from torch.utils.data import DataLoader, random_split



from dataset_loader import ADCPDataset  # your custom dataset
from resnet_temporal import ResNetTemporalClassifier # CNNClassifier  # your model
# from model import TemporalCNN # CNNClassifier  # your model
from utils import seed_everything, get_class_weights, combined_loss, train_model

from types import SimpleNamespace

#For Model training/selection
from sklearn.model_selection import train_test_split

#For report on Test Set
from sklearn.metrics import classification_report

#For CSV report
from pathlib import Path
import csv
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------
# Global settings
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_num = 42
seed_everything(seed_num)

#batch_size = 16 #64 is too big for 10 GB GPU

# Hard-coded paths
data_folder = "/lustre10/scratch/slonimer/BACAX_24hr_h5/"
#data_folder = "/scratch/slonimer/ML_ADCP/BACAX_24hr_h5/"
#data_folder= r"F:\Documents\Projects\ML\ADCP_ML\h5_24h_files\\"
#variant = 'resnet50'   # or 'resnet18', 'resnet34', etc.


def evaluate_model(model, data_loader, device, model_path, log_file_path):
    """
    Evaluate a PyTorch model on a given DataLoader.

    Args:
        model (torch.nn.Module): Trained model
        data_loader (DataLoader): Validation or test DataLoader
        device (torch.device): Device to run evaluation on

    Returns:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        report (str): sklearn classification report
        acc (float): accuracy
        f1 (float): F1-score (macro)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            # Supports (x, y) or (x, y, *) batches
            x = batch[0].to(device)
            y = batch[1].to(device)

            out = model(x)
            out = out.reshape(-1, out.shape[-1])  # Flatten if temporal
            preds = torch.argmax(out, dim=1)

            y = y.view(-1)  # Flatten labels to match preds

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    report = classification_report(y_true, y_pred, digits=4)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # Write to file
    with open(log_file_path, "a") as f:
        f.write(f"=== Model: {model_path} ===\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n\n")

    print(f"Metrics for {model_path} written to {log_file_path}")

    return y_true, y_pred, report, acc, f1




def run_ADCP_validation(data_folder: str = data_folder,
    anomaly_list_path: str = "annotated_files.txt",
    resnet_variant: str = "resnet50",
    batch_size: int = 16,
    debug_mode: bool = False,
    reload_best_model: bool = False,
    run_id: int = None,
    best_model_path: str = None,):


    
    #best_model_path = f"best_model_{resnet_variant}.pt"
    if best_model_path is None:
        if run_id is not None:
            best_model_path = f"best_model_{run_id}_{resnet_variant}_bs{batch_size}.pt"
        else:
            best_model_path = f"best_model_{resnet_variant}.pt"
    

    if run_id is not None:
        log_file = f"/lustre10/scratch/slonimer/logs/run_{run_id}_{resnet_variant}_bs{batch_size}.log"
        sys.stdout = open(log_file, "w")
        sys.stderr = sys.stdout

    
    # Cell 3: Load Dataset
    print(data_folder)
    #Get a list of files from the directory:
    file_list = os.listdir(data_folder)
    h5_files = {k for k in file_list if os.path.splitext(k)[1] == ".h5"}
    #Files are inherently NOT in order in python! So if you want them in order, need to do this:
    h5_files = sorted(h5_files)  # Sorts alphabetically
    
    h5_paths = []
    for filename in h5_files:
        h5_paths.append(data_folder + filename) 
    
    # Load anomaly filenames from a text file
    #with open(annotated_file, "r") as f:
    with open(anomaly_list_path, "r") as f:
        anomaly_files = set(line.strip() for line in f if line.strip())
        
    #Define anomaly paths before truncating h5_paths
    anomaly_paths = [p for p in h5_paths if os.path.basename(p) in anomaly_files]
    
    normal_paths = [p for p in h5_paths if os.path.basename(p) not in anomaly_files]
    
    
    # Example: 70% train, 20% val, 10% test
    #total_size = len(full_dataset)
    #train_size = int(0.7 * total_size)
    #val_size = int(0.20 * total_size)
    #test_size = total_size - train_size - val_size  # ensure all samples are used
    
    #train_dataset, val_dataset, test_dataset = random_split(
    #    full_dataset, [train_size, val_size, test_size]
    
    #train_size = int(0.8 * len(full_dataset))
    #val_size = len(full_dataset) - train_size
    #train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Split anomaly files
    an_train, an_temp = train_test_split(anomaly_paths, test_size=0.3, random_state=seed_num) #70/30 split
    an_val, an_test = train_test_split(an_temp, test_size=0.33, random_state=seed_num) #20/10 split 
    
    # Split normal files
    n_train, n_temp = train_test_split(h5_paths, test_size=0.3, random_state=seed_num) #70/30 split
    n_val, n_test = train_test_split(n_temp, test_size=0.33, random_state=seed_num) #20/10 split 
    
    # Combine
    train_files = an_train + n_train
    val_files = an_val + n_val
    test_files = an_test + n_test
    
    #Create the datasets:
    # train_dataset = ADCPDataset(train_files)
    val_dataset = ADCPDataset(val_files)
    test_dataset = ADCPDataset(test_files)
    
    print('val/test datasets grabbed')
    
    #Set num_workers to 2x cpu cores (NO, get a warning with that), and pin_memory when using a GPU
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    
    
    # Cell 4: Initialize Model and Loss - Resnet50
    num_classes = 6 # full_dataset.num_classes
    # model = TemporalCNN(input_channels=3, num_classes=num_classes)
    
    model = ResNetTemporalClassifier(
        num_classes=num_classes,
        pretrained=False,      # set False if you want to train from scratch
        variant=resnet_variant,     # options: 'resnet50', 'resnet101', 'resnet152' (but only 18,34,50 are avail pre-trained for now)
        resize=(224, 224)        # input size for ResNet
    )
    #In the "model" initialization above, 
    #I've set the pre-trained to false, since no internet, but will load from pre-trained models I got on my VM
    
    if resnet_variant == 'resnet50':
        # --- Load pretrained weights manually (offline) ---
        pretrained_path = f"/lustre10/scratch/slonimer/models/{resnet_variant}.pth"
        state_dict = torch.load(pretrained_path, map_location='cpu')
        # Filter out the fc layer weights (1000-class classifier)
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        # only load matching keys (to ignore classifier layer mismatches)
        missing, unexpected = model.backbone.load_state_dict(filtered_state_dict, strict=False)
        print(f"✅ Loaded pretrained weights with {len(missing)} missing and {len(unexpected)} unexpected keys")




    # Load the saved model for evaluation
    print("\n=== Starting Evaluation on Test Set ===")
    state_dict = torch.load(best_model_path, map_location=device)

    # Detect if the saved state_dict was from DataParallel
    #dp_saved = any(k.startswith("module.") for k in state_dict.keys())

    # Clean up keys: remove any "module." prefixes for single-GPU
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        while new_k.startswith("module."):
            new_k = new_k[len("module."):]
        cleaned_state_dict[new_k] = v

    #Load the cleaned model paramters
    model.load_state_dict(cleaned_state_dict)

    # Wrap model in DataParallel if using multiple GPUs AND saved weights had 'module.'
    use_multi_gpu = torch.cuda.device_count() > 1
    if use_multi_gpu: #and dp_saved:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model.to(device) #May need to comment this out
    model.eval()

    #Print to log file
    log_file_path = "/lustre10/scratch/slonimer/eval_results.txt"

    # Evaluate on validation set
    y_true_val, y_pred_val, report_val, acc_val, f1_val = evaluate_model(model, val_loader, device, best_model_path, log_file_path)
    print("=== Validation Metrics ===")
    print(report_val)
    print(f"Accuracy: {acc_val:.4f}, F1 (macro): {f1_val:.4f}")

    # Evaluate on test set
    y_true_test, y_pred_test, report_test, acc_test, f1_test = evaluate_model(model, test_loader, device, best_model_path, log_file_path)
    print("=== Test Metrics ===")
    print(report_test)
    print(f"Accuracy: {acc_test:.4f}, F1 (macro): {f1_test:.4f}")

    '''
    all_preds = []
    all_labels = []

    #for x, y in val_loader:
    for x, y, _ in test_loader:
        x = x.to(device)
        with torch.no_grad():
            out = model(x)
            #Need to reshape the outputs, and y to match dimensions:
            out = out.reshape(-1, out.shape[-1])  # (B*T, num_classes)
            #The prediction is the class with largest score per sample
            preds = torch.argmax(out, dim=1)

        #Need to reshape the outputs, and y to match dimensions:
        y = y.view(-1)                        # (B*T, )

        #Append the results
        all_preds.append(preds.cpu())
        all_labels.append(y)

    # Flattened predictions and labels
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    #from sklearn.metrics import classification_report
    #print(classification_report(y_true, y_pred))

    basic_metrics = 1
    if basic_metrics:
        from sklearn.metrics import classification_report, f1_score, accuracy_score
        print("=== TEST SET RESULTS ===")
        print(classification_report(y_true, y_pred))
        print(f"F1-score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    else:
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Print metrics
        print(f"Run ID: {run_id}, Variant: {resnet_variant}, Batch: {batch_size}")
        print(f"Accuracy: {acc:.4f}, F1-macro: {f1_macro:.4f}, F1-weighted: {f1_weighted:.4f}")
        
        # Write/append summary CSV
        Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
        file_exists = Path(summary_csv).exists()
        
        with open(summary_csv, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "run_id", "variant", "batch_size", "accuracy", "f1_macro", "f1_weighted", "best_model_path"
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "run_id": run_id,
                "variant": resnet_variant,
                "batch_size": batch_size,
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "best_model_path": best_model_path
            })
    '''

# ------------------------------
# Main block for CLI execution
# ------------------------------
if __name__ == "__main__":
    #Example
    # python run_ADCP_validation.py --batch_size 16 --resnet resnet34 --best_model_path best_model_1_resnet18_bs16.pt
    
    import argparse

    parser = argparse.ArgumentParser(description="Train ADCP anomaly detection model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--resnet", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload_best", action="store_true", help="Reload and evaluate best model")
    parser.add_argument("--run_id", type=int, default=None, help="Optional run ID for logging and model filenames")
    parser.add_argument("--best_model_path", type=str, default=None, help="Optional model path")


    args = parser.parse_args()

    run_ADCP_validation(
        resnet_variant=args.resnet,
        batch_size=args.batch_size,
        debug_mode=args.debug,
        reload_best_model=args.reload_best,
        run_id=args.run_id,
        best_model_path=args.best_model_path,
        )

