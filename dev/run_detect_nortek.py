# TODO: Integrate this code into "detect_nortek_dropouts.run_classify"
#        and then delete this file
#
#

#This code implements the full pipeline for detecting drop-outs in monthly .mat files 
# 
# #This code is designed to run through all the monthly mat files, convert them to h5, split into 24hr h5 files,
# and classify those 24hr files with the trained model. It also logs the detections to a text file. 
# Adjust the batch size and deletion settings as needed based on your system's memory constraints and
# whether you want to keep the intermediate 24hr files.
#
# This code uses the most recent "noEmbed" method


import os
import sys
import importlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import datetime 

# Ensure repo root is available when running this cell independently
#repo_root = Path().resolve().parent
#if str(repo_root) not in sys.path:
#    sys.path.append(str(repo_root))

#from dev import detect_nortek_dropouts
#from src import detect_nortek_dropouts
import detect_nortek_dropouts

# For debugging hot-reload during notebook iteration
#importlib.reload(detect_nortek_dropouts)


# Monthly Mat data to convert (directory path, it will find the mat files)
data_parent = r'F:\Documents\Projects\ADCP\scan_for_data\BACUS\ADCP2MHZ\\'

# For one specific folder:
#folder_list = ['20240801']

# folder_list = ['20131101', '20141201', '20161101', '20170501',
#                 '20220101', '20220201', '20220601', '20220801',
#                 '20221001', '20230201', '20230701', '20240201',
#                 '20240801', '20250201']

#folder_list = ['20250701']
folder_list = ['20251201']


# Inputs:
variant = 'resnet50'
#variant = 'TemporalCNN'

if variant == 'TemporalCNN':
    model_path = r"F:\Documents\GitHub\ml_development\ADCP_ML\src\\models\\" + "best_model_20250508.pt"
else:
    model_path = r"F:\Documents\GitHub\ml_development\ADCP_ML\src\\models\\" + f"best_model_{variant}.pt"

h5_monthly_folder = r'F:\Documents\Projects\ML\ADCP_ML\BACUS\h5_files\\'  # Monthly h5 output
h5_24hr_folder = r'F:\Documents\Projects\ML\ADCP_ML\BACUS\h5_24h_files\\'   # 24hr h5 output


#Batch size: Specify number of 24-hr files to push through the model at once.
#Too few, and it takes a long time to classify all the files. Too many, and it runs out of memory.
num_beams = 3 # DO NOT CHANGE THIS
num_days = 7 # Number of 24hr files to classify at once (adjust based on memory constraints)
batch_size = num_beams*num_days

#Set this to 1 to delete the 24 hr h5 files that are created from each monthly file (recomended) 
do_delete = 1


# Create a text file for keeping track of detections
log_path = os.path.join(h5_24hr_folder, f"dropout_detections_{variant}_{datetime.datetime.now().strftime('%Y%m%dT%H%M')}.txt")
with open(log_path, "w") as f:
    f.write("start_time, end_time, class, beam, duration_minutes\n")

# Build full list of .mat files
mat_paths = []
for folder in folder_list:
    file_list = os.listdir(data_parent + folder)
    mat_files = {k for k in file_list if os.path.splitext(k)[1] == ".mat"}
    print(mat_files)
    for filename in mat_files:
        mat_paths.append(data_parent + folder + '\\' + filename)

def prepare_24hr_from_mat(mat_path):
    """Convert monthly mat -> monthly h5 -> split into 24hr h5 files."""
    detect_nortek_dropouts.convert_monthly_mat_to_h5.extract_mat_to_h5(mat_path, h5_monthly_folder)
    filename_h5 = os.path.splitext(os.path.basename(mat_path))[0] + '.h5'
    input_file = h5_monthly_folder + filename_h5
    #detect_nortek_dropouts.split_h5_to_24hr_files.split_h5_to_24hr_files_with_ann(
    detect_nortek_dropouts.split_h5_to_24hr_files_noEmbed.split_h5_to_24hr_files(
        input_file,
        h5_24hr_folder,
    )
    return mat_path

# Parallel prep step (I/O-heavy); classify once afterward
max_workers = min(4, max(1, len(mat_paths)))
if mat_paths:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(prepare_24hr_from_mat, mat_path) for mat_path in mat_paths]
        for future in as_completed(futures):
            try:
                done_path = future.result()
                print(f"Prepared 24hr files from: {os.path.basename(done_path)}")
            except Exception as e:
                print(f"Failed to prepare one mat file: {e}")

# Classify all generated 24hr files once
model = detect_nortek_dropouts.init_model(model_path)
file_list = os.listdir(h5_24hr_folder)
h5_files = sorted(k for k in file_list if os.path.splitext(k)[1] == ".h5")
detect_nortek_dropouts.classify_and_plot(model, h5_files, h5_24hr_folder, log_path, create_plots=0, batch_size=batch_size)

# Optional cleanup once complete
if do_delete:
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