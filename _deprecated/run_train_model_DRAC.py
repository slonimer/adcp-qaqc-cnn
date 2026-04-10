#Kernel: SSAMBA Kernel on Scratch

#Session Settings on JupyterHub
#Memory: 15000
#Cores: 4
#GPUs: 1

#if DRAC is being buggy, try from ssh:
# $ salloc --account=def-kmoran --time=02:00:00 --mem=16G --cpus-per-task=4 --gres=gpu:h100:1


# Cell 1: Imports & Setup

import os
import torch
from torch.utils.data import DataLoader, random_split

from dataset_loader import ADCPDataset  # your custom dataset
from resnet_temporal import ResNetTemporalClassifier # CNNClassifier  # your model
# from model import TemporalCNN # CNNClassifier  # your model
from utils import seed_everything, get_class_weights, combined_loss, train_model

#For WandB
from types import SimpleNamespace

#For Model training/selection
from sklearn.model_selection import train_test_split

#For report on Test Set
from sklearn.metrics import classification_report

# ------------------------------
# Global settings
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_num = 42
seed_everything(seed_num)

# Hard-coded paths
data_folder = "/scratch/slonimer/ML_ADCP/BACAX_24hr_h5/"
annotated_file = "annotated_files.txt"
best_model_path = "best_model.pt"





def run_train_model_DRAC(data_folder: str = data_folder,
    anomaly_list_path: str = annotated_file,
    resnet_variant: str = "resnet50",
    epochs: int = 100,
    batch_size: int = 16,
    use_wandb: bool = True,
    debug_mode: bool = False,
    reload_best_model: bool = False,):

    #IMPORTANT NOTE: ALSO NEED TO ADJUST THIS IN "utils.py" TO MATCH, in "def train_model()"

    #USE_WANDB = True  # TODO: Set to True to enable logging - Keep false while in development/debugging
    if use_wandb:
        import wandb
        
    #DEBUG_MODE = False




    # Cell 2: Initialize wandb
    #53842d9970aafed0ab407079e403fe03469dcb33

    if use_wandb:
        wandb.init(project="adcp-anomaly-detection", #dir="/scratch/ML_ADCP/wandb_runs", 
                config={
                    "model": "ResNetTemporalClassifier",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": 1e-3,
                    "loss_alpha": 0.5,
                    "optimizer": "Adam"
                })
        config = wandb.config
    else:
        config = SimpleNamespace(**{
            "model": "ResNetTemporalClassifier",
            "epochs": epochs,
            "batch_size": batch_size, #2,
            "lr": 1e-3,
            "loss_alpha": 0.5,
            "optimizer": "Adam"})


    # Cell 3: Load Dataset

    #Get a list of files from the directory:
    data_folder = "/scratch/slonimer/ML_ADCP/BACAX_24hr_h5/"
    #data_folder= r"F:\Documents\Projects\ML\ADCP_ML\h5_24h_files\\"
    file_list = os.listdir(data_folder)
    h5_files = {k for k in file_list if os.path.splitext(k)[1] == ".h5"}

    #Files are inherently NOT in order in python! So if you want them in order, need to do this:
    h5_files = sorted(h5_files)  # Sorts alphabetically

    #print(os.path.splitext(file_list[0]))
    #print(h5_files)

    h5_paths = []
    for filename in h5_files:
        h5_paths.append(data_folder + filename) 

    #print(len(h5_paths))

    # Load anomaly filenames from a text file
    #with open("annotated_files.txt", "r") as f:
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
    train_dataset = ADCPDataset(train_files)
    val_dataset = ADCPDataset(val_files)
    test_dataset = ADCPDataset(test_files)

    print('train/val/test datasets grabbed')


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)



    # Cell 4: Initialize Model and Loss
    num_classes = 6 # full_dataset.num_classes
    # model = TemporalCNN(input_channels=3, num_classes=num_classes)
    model = ResNetTemporalClassifier(
        num_classes=num_classes,
        pretrained=True,      # set False if you want to train from scratch
        variant=resnet_variant,     # options: 'resnet50', 'resnet101', 'resnet152'
        resize=(224, 224)        # input size for ResNet
    )

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

    #DEBUG_MODE = 0
    if debug_mode:
        import os
        os.environ["TORCH_DISABLE_MKL"] = "1"  # optional: disables MKL
        os.environ["ONEDNN_VERBOSE"] = "0"
        os.environ["DNNL_VERBOSE"] = "0"
        torch.backends.mkldnn.enabled = False

    if use_wandb:
        model = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=config.epochs, patience=5, USE_WANDB=use_wandb)
    else:
        model, history = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=config.epochs, patience=5, USE_WANDB=use_wandb)

    # ([2, 3, 288, 102])
    # => [batch, channels, time, range]
        
    #Best result using ce and dice-loss:
    #Epoch 7/20 | Train Loss: 0.7003 | Val Loss: 0.9850 | Val F1: 0.4931
    #?? Early stopping triggered.
        
    # With ce and graduated dice-loss:
    # Starting Validation on Epoch #  6
    # Epoch 7/20 | Train Loss: 0.7778 | Val Loss: 0.9043 | Val F1: 0.4931
    # #
    # Starting Validation on Epoch #  7
    # Epoch 8/20 | Train Loss: 0.7598 | Val Loss: 0.9866 | Val F1: 0.4931
    # ?? Early stopping triggered.
        

    # Cell 6: Load Best Model and Evaluate on the TEST set
    model.load_state_dict(torch.load(best_model_path))
    #model.load_state_dict(torch.load("best_model_20250508.pt"))
    #model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    all_preds = []
    all_labels = []

    #for x, y in val_loader:
    for x, y in test_loader:
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

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    #from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred))

# ------------------------------
# Main block for CLI execution
# ------------------------------
if __name__ == "__main__":
    #Example
    # python run_train_model_DRAC.py --epochs 100 --batch_size 16 --resnet resnet34
    
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

    run_train_model_DRAC(
        resnet_variant=args.resnet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_wandb=args.wandb,
        debug_mode=args.debug,
        reload_best_model=args.reload_best,
    )


