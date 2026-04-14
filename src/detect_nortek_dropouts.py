
import os

import torch
from torch.utils.data import DataLoader, random_split

from src.dataset_loader_noEmbed import ADCPDataset  # your custom dataset
#from src.dataset_loader import ADCPDataset  # your custom dataset
from src.resnet_temporal import ResNetTemporalClassifier # Resnet Models
from src.model import TemporalCNN # CNNClassifier  # Original model
from src.utils import seed_everything, get_class_weights, combined_loss, train_model

import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime

import src.convert_monthly_mat_to_h5 as convert_monthly_mat_to_h5
#import src.split_h5_to_24hr_files as split_h5_to_24hr_files
import src.split_h5_to_24hr_files_noEmbed as split_h5_to_24hr_files_noEmbed

from concurrent.futures import ThreadPoolExecutor, as_completed

from collections import OrderedDict #Use this for ordering batches of multiple files

import re

#This was developed with the kernel "adcp_anomaly_env"

def init_model(model_path):
    #Load the model to use for classification, created with "ADCP_Anomaly_Training.ipynb" (or other)

    #Set this up before loading the model:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional: disable some backend optimizations for CPU if needed
    #Was trying to have this only run conditionally, but not working, so just make it run always, and fix later if needed
    #if device == "cpu":
    os.environ["TORCH_DISABLE_MKL"] = "1"  # optional: disables MKL
    os.environ["ONEDNN_VERBOSE"] = "0"
    os.environ["DNNL_VERBOSE"] = "0"
    torch.backends.mkldnn.enabled = False

    #model_path = r"F:\Documents\GitHub\ml_development\ADCP_ML\\" + "best_model_20250508.pt"
    num_classes = 6 

    # Determine model type from filename
    filename = os.path.basename(model_path).lower()
    if "resnet" in filename:
        # Extract variant from filename, e.g., "resnet34" or "resnet50"
        match = re.search(r"resnet(\d+)", filename)
        resnet_variant = match.group(0) if match else "resnet50"

        print(f"📌 Initializing ResNet model ({resnet_variant})")
        model = ResNetTemporalClassifier(
            num_classes=num_classes,
            pretrained=False,
            variant=resnet_variant,
            resize=(224, 224)
        )

        if resnet_variant == 'resnet50':
            # --- Load pretrained weights manually (offline) ---
            
            #pretrained_path = model_path
            pretrained_path = r"F:\Documents\GitHub\ml_development\ADCP_ML\src\\models\\" + f"{resnet_variant}.pth"
            #pretrained_path = f"/lustre10/scratch/slonimer/models/{resnet_variant}.pth"
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Filter out the fc layer weights (1000-class classifier)
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
            # only load matching keys (to ignore classifier layer mismatches)
            missing, unexpected = model.backbone.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ Loaded pretrained weights with {len(missing)} missing and {len(unexpected)} unexpected keys")


    else:
        print("📌 Initializing TemporalCNN model")
        model = TemporalCNN(input_channels=3, num_classes=num_classes)

    # Load weights
    #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu') ))

    # Load the saved model for evaluation
    print("\n=== Starting Evaluation on Test Set ===")
    state_dict = torch.load(model_path, map_location=device)

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
    
    #model.load_state_dict(torch.load(model_path, map_location=device))
    #model.to(device)
    model.eval()
    return model 


def classify_test_data(model, h5_24hr_file, store_x=False, batch_size=3):

    #Ensure that batch_size is a multiple of 3 (number of beams), since we want to keep the beams together in the batches, and not split them up
    if batch_size % 3 != 0:
        raise ValueError("batch_size must be a multiple of 3 (number of beams) to keep beams together in batches")

    #Set this up:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Create a generic function for classifying files
    print(f'Loading: {h5_24hr_file}')
    test_file_dataset = ADCPDataset(h5_24hr_file)
    #test_loader = DataLoader(test_file_dataset, batch_size=3, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_file_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_x = []
    all_preds = []
    all_labels = []
    all_meta = []

    n_debug = 0

    for x, y, meta in test_loader:
        x = x.to(device)
        model = model.to(device)  # ← Add this line
        with torch.no_grad():
            out = model(x)
            #Need to reshape the outputs, and y to match dimensions:
            out = out.reshape(-1, out.shape[-1])  # (B*T, num_classes)
            #The prediction is the class with largest score per sample
            preds = torch.argmax(out, dim=1)

        #Need to reshape the outputs, and y to match dimensions:
        y = y.view(-1)                        # (B*T, )

        #Append x (aka input ADCP data)
        if store_x:
            #Only store the input data if specified, since it can take a lot of memory, 
            # and is only really needed if making plots
            all_x.append(x.cpu())
        else:
            all_x.append(None) #Placeholder if not storing x
        
        #Append results
        all_preds.append(preds.cpu())
        all_labels.append(y)
        all_meta.append(meta)

        n_debug += 1
        print(f"Processed batch {n_debug} with shape {x.shape}, predictions shape {preds.shape}, labels shape {y.shape}")
        
    #return x, y, preds, meta
    return all_x, all_labels, all_preds, all_meta


# import h5py
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import numpy as np
# import datetime


def get_segments(annotations, num_classes=4):
    segments = []
    annotations = np.asarray(annotations)

    for cls in range(1, num_classes):  # Skip class 0 (background)
        mask = annotations == cls
        if not np.any(mask):
            continue

        diffs = np.diff(mask.astype(int))
        start_indices = np.where(diffs == 1)[0] + 1
        end_indices = np.where(diffs == -1)[0] + 1

        if mask[0]:
            start_indices = np.r_[0, start_indices]
        if mask[-1]:
            end_indices = np.r_[end_indices, len(annotations)-1]

        segments.extend((start, end, cls) for start, end in zip(start_indices, end_indices))

    return segments

'''
def get_segments(annotations, ann):
    #This is used by "plot_results"
    if ann==1: #For annotations (may be multiclass)
        mask = np.diff(annotations) != 0 # Create a mask of where changes in annotations are non-zero
        #diffs = mask.astype(int)
    elif ann==0: #For predictions
        mask = annotations != 0 # Create a mask of where annotations are non-zero
        
    diffs = np.diff(mask.astype(int)) # Find the changes in the mask
    start_indices = np.where(diffs == 1)[0] + 1 # Start indices: where diff == 1 (0 → 1)
    end_indices = np.where(diffs == -1)[0] + 1 # End indices: where diff == -1 (1 → 0)
    # Handle edge cases: 
    if mask[0]: #if it starts with a non-zero 
        start_indices = np.r_[0, start_indices]

    if mask[-1]: # if it ends with a non-zero
        end_indices = np.r_[end_indices, len(annotations)-1]

    anomaly_segments = list(zip(start_indices, end_indices)) # Zip together
    return anomaly_segments
'''

def plot_results(x, annotations, predictions, filename, meta, outdir = None) :
    x = x.cpu()
    annotations = annotations.cpu()
    predictions = predictions.cpu()

    # If saving to an output folder, normalize filename to basename only
    if outdir is not None:
        filename = os.path.basename(filename)

    n_beams = x.shape[0]#[2]
    n_channels = x.shape[1]#[2]

    #time_data = meta['time'][0]
    time_data = [datetime.datetime.utcfromtimestamp(t.item()) for t in meta['time'][0]]

    for beam in range(n_beams):
        fig, axs = plt.subplots(n_channels, 1, sharex=True, figsize=(12, 2.5*n_channels))
        if n_channels == 1:
            axs = [axs]

        #Get the annotations for this beam
        anomaly_segments = get_segments(annotations[beam].cpu().numpy(), num_classes = 4)
        pred_segments = get_segments(predictions[beam].cpu().numpy(), num_classes = 4)
        #anomaly_segments = get_segments(annotations[beam].cpu().numpy(),ann = 1)
        #pred_segments = get_segments(predictions[beam].cpu().numpy(), ann = 0)

        #print(anomaly_segments)
        #print(pred_segments)

        #Determine if any annotations present in this beam:
        cls_str = '' #Initialize as nothing
        ann = annotations[beam].cpu().numpy()
        if np.any(ann>0):
            cls = np.median(ann[ann>0])
            cls_str = ', class: {}'.format(int(cls))


        #Plot Velocity, backscatter, and correlation, for each beam
        for ch in range(n_channels):
            #Plot the Complex Data
            im = axs[ch].imshow(
                x[beam,ch,:,:].T, aspect='auto', origin='lower',
                #extent=[extent[0], extent[1], extent[2], extent[3]],
                extent=[time_data[0], time_data[-1], 0, x.shape[3]-1],
                interpolation='nearest',
                cmap='jet',
            )

            #Set the figure title
            if beam == 0 and ch == 0:
                fig.suptitle('Annotations = Shaded, Predictions = --')
                #fig.suptitle('File: {}'.format(filename))
            
            #Set the subplot titles
            if ch == 0:
                axs[ch].set_title('Beam #{} {}'.format(beam+1, cls_str))   
           
            #Add labels and titles
            #axs[ch].set_ylabel("Range bin" if range_dim is not None else '')
            #axs[ch].set_title(f"{var} - Channel {ch+1}")

            #Add dashed vertical lines for predictions
            colors = ['None','black','red','blue']
            for start, end, cls in pred_segments:
                if 0 <= start < x.shape[2]:
                    axs[ch].axvline(x=time_data[start], color=colors[cls], linestyle='dashed', alpha=0.7)
                    #axs[ch].axvline(x=time_data[start], color='black', linestyle='dashed', alpha=0.7)
                    #axs[ch].axvline(x=start, color='red', linestyle='dashed', alpha=0.7)
                if 0 <= end < x.shape[2]:
                    axs[ch].axvline(x=time_data[end], color=colors[cls], linestyle='dashed', alpha=0.7)
                    #axs[ch].axvline(x=time_data[end], color='black', linestyle='dashed', alpha=0.7)
                    #axs[ch].axvline(x=end, color='red', linestyle='dashed', alpha=0.7)

            #Add shading for annotations
            for start, end, cls in anomaly_segments:
                if 0 <= start < x.shape[2] and 0 <= end <= x.shape[2]:
                    axs[ch].axvspan(time_data[start], time_data[end], color=colors[cls], alpha=0.3)
                    #axs[ch].axvspan(time_data[start], time_data[end], color='black', alpha=0.3)
                    #axs[ch].axvspan(start, end, color='black', alpha=0.3)

            #Add a colorbar
            fig.colorbar(im, ax=axs[ch], label=meta['channels'][ch][0])
            #fig.colorbar(im, ax=axs[ch], label='color')

            #Add y-axis label
            axs[ch].set_ylabel('{}'.format('Range [m]') )


        # -- Date formatting for X --
        axs[-1].xaxis_date()  # tells matplotlib to interpret x as dates
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()  # Makes dates pretty (auto-rotates, etc.)

        #Add x-axis label
        axs[-1].set_xlabel('{} [HH:MM] UTC'.format(time_data[0].strftime('%Y-%m-%d')) )

        #axs[-1].set_xlabel(time_dt[0].astype('datetime64[D]').astype(str))   # 'yyyy-mm-dd' date for xlabel
        #axs[-1].set_xlabel("Time (hours since start)")
        #fig.suptitle(f"{var} (shape={arr.shape})")
        plt.tight_layout()
        
        if outdir:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            stem = os.path.splitext(filename)[0]
            plt.savefig(f"{outdir}\\{stem}_beam{beam+1}.png", dpi=300)
            #plt.savefig(f"{outdir}\\{filename[:-3]}_beam{beam+1}.png", dpi=300)
        #if show:
        #    plt.show()
        plt.show()
        #plt.close()



def classify_and_plot(model, h5_files, h5_24hr_folder, log_path, create_plots = 1, batch_size=3):

    #Ensure that batch_size is a multiple of 3 (number of beams), since we want to keep the beams together in the batches, and not split them up
    if batch_size % 3 != 0:
        raise ValueError("batch_size must be a multiple of 3 (number of beams) to keep beams together in batches")
    
    #Define the path for tracking detections:
    #log_path = os.path.join(h5_24hr_folder, "dropout_detections.txt")

    #Combine paths into a list
    h5_24hr_files = []
    for h5_file in h5_files:
        h5_24hr_files.append(h5_24hr_folder + h5_file) 
        #print(h5_24hr_folder)
        #print(h5_file)
        
    #Run the classification
    print(f'Running drop-out detection on all 24 hr files')
    if create_plots == 1:
       x, y, preds, meta = classify_test_data(model, h5_24hr_files, store_x=True, batch_size=batch_size)
    else:
        #If not creating plots, don't store the input data (x), since it can take a lot of memory, and is only really needed for plotting
        x, y, preds, meta = classify_test_data(model, h5_24hr_files, batch_size=batch_size)

    #print('printing shapes of x, and y')
    #print(len(x))
    #print(len(y))

    #Output the results for each 24 hr file
    #for h5_file, x_sample, y_sample, pred_sample, meta_sample in zip(h5_files, x, y, preds, meta):

    #Loop through each batch of outputs:
    for x_sample, y_sample, pred_sample, meta_sample in zip(x, y, preds, meta):
        
        n_in_batch = len(meta_sample["beam_number"]) # actual rows in this batch

        #Reorganize the structure of the labels and predictions
        labels= y_sample.view(n_in_batch, 288) #annotations
        predictions = pred_sample.view(n_in_batch, 288)
        #labels = y_sample.view(3,288) #annotations
        #predictions = pred_sample.view(3,288)

        #DEBUG
        #print(labels.shape)
        #print(predictions.shape)

        #Loop through contents of each batch: (each beam dataset)

        # Group sample indices by filename within this batch
        file_to_samples = OrderedDict()
        for sample in range(n_in_batch):
            filename = meta_sample["filename"][sample]
            file_to_samples.setdefault(filename, []).append(sample)

        # Process one file at a time within the batch
        for h5_file, file_samples in file_to_samples.items():

            #Initialize variables to Determine if any predictions present in any beam:
            do_plot = 0
            beam_num = []

            
            #for sample in range(n_in_batch): #for beam in range(3):
            for sample in file_samples:
                #labels_beam = labels[beam].cpu().numpy()
                pred_sample = predictions[sample].cpu().numpy()
                
                #Number of predictions per beam, per day
                n_predictions = np.sum(pred_sample>0)

                beam_number = int(meta_sample["beam_number"][sample].item())
                time_vec = meta_sample["time"][sample]  # time for this sample

                #DEBUG
                #print(f"Filename: {meta_sample['filename'][sample]}")
                #print(f"sample {sample}: n_predictions {n_predictions}, beam number {beam_number}")  

                #If more than 5 minutes (1 prediction) in a day, append to text file:
                # Only proceed if the log file already exists
                if os.path.exists(log_path) and n_predictions >= 1:

                    #Find non-zero predictions (ann=0 indicates this is for predictions, not labels)
                    segments = get_segments(pred_sample, num_classes = 4)
                    #segments = get_segments(pred_beam, ann=0) 
                    for start_idx, end_idx, cls in segments:
                        start_time = float(time_vec[start_idx])
                        end_time = float(time_vec[end_idx])
                        #start_time = float(meta_sample["time"][0][start_idx])
                        #end_time = float(meta_sample["time"][0][end_idx])
                        #beam_number = meta_sample["beam_number"]

                        # Convert to datetime if values are Unix timestamps
                        if isinstance(start_time, (int, float)):
                            start_time = datetime.datetime.utcfromtimestamp(start_time)
                            end_time = datetime.datetime.utcfromtimestamp(end_time)

                        duration = (end_time - start_time).total_seconds() / 60.0

                        # Append to log
                        with open(log_path, "a") as f:
                            f.write(f"{start_time}, {end_time}, {cls}, {beam_number}, {duration:.1f}\n")
                            #f.write(f"{start_time}, {end_time}, {cls}, {sample+1}, {duration:.1f}\n")

                #If more than one hour predicted in a day in any beam, make plots:
                n_samples_threshold = 12 # 1 hour
                # print(np.sum(pred_beam>0)) #DEBUG
                if n_predictions > n_samples_threshold:
                    do_plot = 1
                    beam_num.append(sample)

            #Print a message
            print('Diagnostic plots are only produced if more than 1 hour of drop-outs are detected')
            
            #Either plot the results, or print a message
            if do_plot == 1 and create_plots==1 and x_sample is not None:
                #Plot the results - Don't show them, just save a figure
                file_x = x_sample[file_samples]
                file_labels = labels[file_samples]
                file_predictions = predictions[file_samples]

                file_meta = {
                    "time": [meta_sample["time"][sample] for sample in file_samples],
                    "filename": h5_file,
                    "channels": meta_sample["channels"][0],
                    "beam_number": [int(meta_sample["beam_number"][sample].item()) for sample in file_samples],
                }

                plot_results(file_x, file_labels, file_predictions, h5_file, file_meta, outdir=h5_24hr_folder)
                print(f"Drop-outs found in Beam {beam_num}! In {h5_file}")

    '''
    #Run the classification
    for h5_file in h5_files:

        print(f'Running drop-out detection on {h5_file}')

        #classify_test_data expects a list of files, so if only giving 1 file at a time,
        # need to wrap it in square brackets
        x, y, preds, meta = classify_test_data(model, [h5_24hr_folder + h5_file])

        #Reorganize the structure of the labels and predictions
        labels = y.view(3,288) #annotations
        predictions = preds.view(3,288)

        #Determine if any predictions present in any beam:
        do_plot = 0
        for beam in range(3):
            labels_beam = labels[beam].cpu().numpy()
            pred_beam = predictions[beam].cpu().numpy()
            
            #If more than one hour predicted in a day in any beam:
            n_samples = 12 # 1 hour
            if np.sum(pred_beam>0)>n_samples:
                do_plot = 1
        
        #Either plot the results, or print a message
        if do_plot == 1:
            #Plot the results - Don't show them, just save a figure
            plot_results(x, labels, predictions, h5_file, meta, outdir=h5_24hr_folder)
            #And print a message
            print('Drop-outs found! In {}'.format(h5_file))
        else:
            #Print a message
            print('No drop-outs found in {}'.format(h5_file))
    '''

def prepare_24hr_from_mat(mat_path, h5_monthly_folder, h5_24hr_folder):
    """Convert monthly mat -> monthly h5 -> split into 24hr h5 files."""
    #STEP 1: Convert monthly mat to monthly h5
    convert_monthly_mat_to_h5.extract_mat_to_h5(mat_path, h5_monthly_folder)
    #STEP 2: Split monthly h5 to 24hr h5 files
    filename_h5 = os.path.splitext(os.path.basename(mat_path))[0] + '.h5'
    input_file = h5_monthly_folder + filename_h5
    split_h5_to_24hr_files_noEmbed.split_h5_to_24hr_files(
        input_file,
        h5_24hr_folder,
    )

def run_classify(h5_monthly_folder, h5_24hr_folder, mat_path=None, model_path=None, log_path=None, batch_size=None, create_plots = 1):
    #Required Inputs: 
    # h5_monthly_folder # Folder for h5 files converted from mat format (input and output)
    # h5_24hr_folder = #24hr h5 files (input and output)
    # 
    #Optional Inputs
    # mat_path : Path to mat files to convert to h5 format
    # model_path: Path to a trained model, otherwise will use default
    # log_path: Path to an existing log file, otherwise will make one
    # batch_size: Default is 3. Any input options must a multiple of 3
    #
    #Examples
    # h5_monthly_folder = r'F:\Documents\Projects\ML\ADCP_ML\BACUS\h5_files\\' 
    # h5_24hr_folder = r'F:\Documents\Projects\ML\ADCP_ML\BACUS\h5_24h_files\\'
    # 
    # mat_path = [r'F:\Documents\Projects\ADCP\scan_for_data\BACUS\ADCP2MHZ\\20240801\\ADCP_20240801.mat'] 
    # model_path = r"F:\Documents\GitHub\ml_development\ADCP_ML\src\\models\\" + "best_model_20250508.pt" 
    # log_path = os.path.join(h5_24hr_folder, "dropout_detections.txt")
    # batch_size = 6 

    '''
    #Optional method for running this code
    
    #Create a text file for keeping track of detections:
    log_path = os.path.join(h5_24hr_folder, "dropout_detections.txt")
    # Create or clear the file
    with open(log_path, "w") as f:
        f.write("start_time, end_time, class, beam, duration_minutes\n")
        
    #PART 1: Convert monthly Mat to monthly h5:
    for folder in folder_list:
        # Path to your .mat file
        file_list = os.listdir(data_parent + folder)
        mat_files = {k for k in file_list if os.path.splitext(k)[1] == ".mat"}
        print(mat_files)

        #Combine the filenames into a proper path:
        mat_paths = []
        for filename in mat_files:
            mat_paths.append(data_parent + folder + '\\' + filename) 

        #Run the extraction
        for mat_path in mat_paths:
            detect_nortek_dropouts.run_classify(model_path, mat_path, h5_monthly_folder, h5_24hr_folder) 
    '''
    if model_path is None:
        model_path = r"F:\Documents\GitHub\ml_development\ADCP_ML\src\\models\\" + "best_model_20250508.pt"

    #Ensure that batch_size is a multiple of 3 (number of beams), since we want to keep the beams together in the batches, and not split them up
    if batch_size is not None and batch_size % 3 != 0:
        raise ValueError("batch_size must be a multiple of 3 (number of beams) to keep beams together in batches")


    #If an output text logfile (for keeping track of detections), is not specified, create one
    if log_path is None:
        # Determine model type from filename
        filename_model = os.path.basename(model_path).lower()
        if "resnet" in filename_model:
            # Extract variant from filename, e.g., "resnet34" or "resnet50"
            match = re.search(r"resnet(\d+)", filename_model)
            model_variant = match.group(0) if match else "resnet50"
        else:
            model_variant = 'CNN'

        log_path = os.path.join(h5_24hr_folder, f"dropout_detections_{model_variant}_{datetime.datetime.now().strftime('%Y%m%dT%H%M')}.txt")
        with open(log_path, "w") as f:
            f.write("start_time, end_time, class, beam, duration_minutes\n")

    #Initialize the classification model
    model = init_model(model_path)

    #data_parent = r'F:\Documents\Projects\ADCP\scan_for_data\BACUS\ADCP2MHZ\\'
    #folder_list = ['20240801']
    #folder_list = os.listdir(data_parent)

    ######################################
    #PART 1: Convert monthly Mat to monthly h5, and then to 24hr h5 files
    # Build full list of .mat files
    if mat_path:    
        mat_paths = []
        for folder in mat_path:
            file_list = os.listdir(folder)
            mat_files = {k for k in file_list if os.path.splitext(k)[1] == ".mat"}
            print(mat_files)
            for filename in mat_files:
                mat_paths.append(folder + '\\' + filename)


        # Parallel prep step (I/O-heavy); classify once afterward
        max_workers = min(4, max(1, len(mat_paths)))
        if mat_paths:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(prepare_24hr_from_mat, mat_path, h5_monthly_folder, h5_24hr_folder) for mat_path in mat_paths]
                for future in as_completed(futures):
                    try:
                        done_path = future.result()
                        print(f"Prepared 24hr files from: {os.path.basename(done_path)}")
                    except Exception as e:
                        print(f"Failed to prepare one mat file: {e}")


        #split_h5_to_24hr_files_noEmbed.split_h5_to_24hr_files(
        #    input_file,             # your big HDF5 source (created with import_monthly_mat_to_h5.py)
        #    h5_24hr_folder,          # output dir for 24hr files
        #)
        
        #split_h5_to_24hr_files.split_h5_to_24hr_files_with_ann(
        #    input_file,             # your big HDF5 source (created with import_monthly_mat_to_h5.py)
        #    h5_24hr_folder,          # output dir for 24hr files
        #    annotations_file,        # your .mat annotations file
        #)

    ######################################
    #PART 3: Plot if any detections are found

    #Load, classify, and plot the h5 24hr files,
    file_list = os.listdir(h5_24hr_folder)
    h5_files = sorted(k for k in file_list if os.path.splitext(k)[1] == ".h5")
    #print(h5_files)

    if batch_size is None:
        classify_and_plot(model, h5_files, h5_24hr_folder, log_path, create_plots)
    else:
        classify_and_plot(model, h5_files, h5_24hr_folder, log_path, create_plots, batch_size=batch_size)

    ######################################
    #PART 4: Delete the temporary h5 files
    for h5_folder in {h5_monthly_folder, h5_24hr_folder}:
        file_list = os.listdir(h5_folder)
        h5_files = sorted(k for k in file_list if k.endswith(".h5"))
        for h5_file in h5_files:
            full_path = os.path.join(h5_folder, h5_file)
            try:
                os.remove(full_path)
                print(f"Deleted {h5_file}")
            except Exception as e:
                print(f"Failed to delete {h5_file}: {e}")
    