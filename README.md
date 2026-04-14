## Overview

Harnessing the power of deep learning to accurately and efficiently detect anomalies in ADCP data.

## Data

Data was downloaded via the ONC Open API in MAT-format

Annotations were created manually using a custom tool created in MATLAB (not included here)

ADCP data was downloaded in 1 month chunks averaged to 5 minute intervals

* Data is converted to h5 format. 

* Data is split into 24 hour h5 format files, and annotations are embedded in the files.

## Code Development

2025-04:

dev/split_h5_to_24hour_add_annotation.ipynb: Notebook to develop and test h5 code below 
src/convert_monthly_mat_to_h5.py : Convert from monthly MAT format to h5 format
src/split_h5_to_24hr_files.py : Split monthly h5 files to 24 hour h5 files, and embedd the manual annotations
 
notebooks/split_monthly_mat_to_24hour_h5.ipynb : Code to implement and run the python files noted above

dev/ADCP_Anomaly_Training.ipynb : Original jupyter notebook used for development on local laptop (no GPU).  Uses a simple 5 layer CNN


2025-05: 

src/utils.py : Utility functions, including seed_everything(), get_class_weights(), combined_loss(), train_model()
src/dataset_loader.py : Custom ADCP dataset loader
src/model.py : Code containing the architecture of the simple 5 layer CNN

annotated_files.txt : List of files containing anomalies.  Created to ensure examples are included in the train, test, and val datasets

dev/ADCP_Anomaly_Training_DRAC.ipynb: Nearly identical to "ADCP_Anomaly_Training.ipynb", but with GPU functionality 


2025-06: 

src/detect_nortek_dropouts.py : Tool to implement the trained model for anomaly detection

notebooks/ADCP_Anomaly_Detection.ipynb : Example code showing how to implement "detect_nortek_dropouts.py"


2025-10:

src/resnet_temporal.py: ResNet model architectures that can fit the existing framework

notebooks/ADCP_Anomaly_Training_resnet.ipynb: Builds on ADCP_Anomaly_Training.ipynb, for developing model training code
notebooks/ADCP_Anomaly_Training_DRAC_resnet.ipynb: Nearly identical to code above, but for using GPU resources 


2026-04:

Created several new files to seperate the annotation handling and file splitting so that annotations are NOT embedded in the h5 formatted files.  This will make training easier, but making it simpler to adjust the annotations without needing to re-extract / re-generate every training h5 file each time.  

src/split_h5_to_24hr_files_noEmbed.py: New code to split mat formatted to hourly h5 files, without embedding annotations
src/parse_annotations.py: New code to parse annotations 
src/dataset_load_noEmbed.py: New dataset loader that uses parse_annotations to annotate the h5 files, on the fly

notebooks/split_monthly_mat_to_24hour_h5_noEmbed.ipynb: Example code for running the file splitting
notebooks/ADCP_Anomaly_Detection.ipynb: Has an example to show how to run the detection model on new monthly mat files





 
