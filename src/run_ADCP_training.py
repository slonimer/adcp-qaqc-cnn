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


def run_ADCP_training(data_folder: str = data_folder,
    anomaly_list_path: str = "annotated_files.txt",
    resnet_variant: str = "resnet50",
    epochs: int = 100,
    batch_size: int = 16,
    use_wandb: bool = False,
    debug_mode: bool = False,
    reload_best_model: bool = False,
    run_id: int = None,):

    
    #For WandB
    USE_WANDB = False # not working on rorqual, so setting to false
    # USE_WANDB = True  # TODO: Set to True to enable logging - Keep false while in development/debugging
    DEBUG_MODE = False
    
    #best_model_path = f"best_model_{resnet_variant}.pt"
    if run_id is not None:
        best_model_path = f"best_model_{run_id}_{resnet_variant}_bs{batch_size}.pt"
    else:
        best_model_path = f"best_model_{resnet_variant}.pt"
    

    if run_id is not None:
        log_file = f"/lustre10/scratch/slonimer/logs/run_{run_id}_{resnet_variant}_bs{batch_size}.log"
        sys.stdout = open(log_file, "w")
        sys.stderr = sys.stdout



    if USE_WANDB:
        import wandb
    
    
        
    if USE_WANDB:
        wandb.init(project="adcp-anomaly-detection", #dir="/scratch/ML_ADCP/wandb_runs", 
                config={
                    "model": "ResNetTemporalClassifier",
                    "epochs": epochs,
                    "batch_size": batch_size, #16, 64 too big
                    "lr": 1e-3,
                    "loss_alpha": 0.5,
                    "optimizer": "Adam"
                })
        config = wandb.config
    else:
        config = SimpleNamespace(**{
            "model": "ResNetTemporalClassifier",
            "epochs": epochs,
            "batch_size": batch_size, #16
            "lr": 1e-3,
            "loss_alpha": 0.5,
            "optimizer": "Adam"})
    
    
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

    
    DEBUG_MODE = 0
    if DEBUG_MODE:
        #WHILE DEBUGGING, ONLY USE A FEW FILES
        #NEED THIS FOR DEBUGGING - POTENTIALLY USES 13 GB OF MEMORY
        num_files = 200 # 1000 makes the kernel crash w 2400 MB memory
        h5_paths = h5_paths[:num_files]    
    
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
    train_dataset = ADCPDataset(train_files)
    val_dataset = ADCPDataset(val_files)
    test_dataset = ADCPDataset(test_files)
    
    print('train/val/test datasets grabbed')
    
    #Set num_workers to 2x cpu cores (NO, get a warning with that), and pin_memory when using a GPU
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    
    
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


    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    #class_weights = torch.tensor([1.6761e-01, 3.8812e+01, 1.4015e+02, 9.8140e+02, 0.0000e+00, 0.0000e+00])
    class_weights = get_class_weights(train_dataset, num_classes)                    # FIX THIS LATER -  UNCOMMENT THIS OR DEFINE MANUALLY BUT CORRECT WEIGHTS
    print(class_weights)
    
    #Original with 6 classes
    #tensor([1.8522e-01, 2.0864e+00, 2.3713e+01, 9.4286e+01, 2.3760e+04, 1.4505e+01]) # For 200 files, is inverse of [5.3990, 0.4793, 0.0422, 0.0106, 0.0000,0.0689]
    # tensor([2.2450e-01, 3.8812e+01, 1.4015e+02, 9.8140e+02, 1.9692e+00, 9.9600e-01]) # For full dateset (or 70% anyways)
    
    loss_fn = combined_loss(class_weights, alpha=config.loss_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    
    
    # Cell 5: Train
    
    #I was having bugs in this, where it was saying it failed creating a primitive. This was found to solve the issue (but shouldn't be used when proper training/running)
    DEBUG_MODE = 0
    if DEBUG_MODE:
        #import os
        os.environ["TORCH_DISABLE_MKL"] = "1"  # optional: disables MKL
        os.environ["ONEDNN_VERBOSE"] = "0"
        os.environ["DNNL_VERBOSE"] = "0"
        torch.backends.mkldnn.enabled = False
    
    if USE_WANDB:
        model = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=config.epochs, patience=5, USE_WANDB=USE_WANDB, best_model_path = best_model_path)
    else:
        model, history = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=config.epochs, patience=5, USE_WANDB=USE_WANDB, best_model_path = best_model_path)



# ------------------------------
# Main block for CLI execution
# ------------------------------
if __name__ == "__main__":
    #Example
    # python run_ADCP_training.py --epochs 100 --batch_size 16 --resnet resnet34
    
    import argparse

    parser = argparse.ArgumentParser(description="Train ADCP anomaly detection model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--resnet", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload_best", action="store_true", help="Reload and evaluate best model")
    parser.add_argument("--run_id", type=int, default=None, help="Optional run ID for logging and model filenames")


    args = parser.parse_args()

    run_ADCP_training(
        resnet_variant=args.resnet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_wandb=args.wandb,
        debug_mode=args.debug,
        reload_best_model=args.reload_best,
        run_id=args.run_id,)

