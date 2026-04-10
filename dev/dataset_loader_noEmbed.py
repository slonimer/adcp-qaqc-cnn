import torch
from torch.utils.data import Dataset, DataLoader
import parse_annotations
import h5py
import numpy as np
import os
import re
from datetime import datetime, timedelta

import importlib
importlib.reload(parse_annotations)

class ADCPDataset(Dataset):
    def __init__(self, file_list, annotations_mat_file='', transform=None, normalize=True, expected_shape=(3, 288, 102)):
        """
        Args:
            file_list (List[str]): List of paths to .h5 files
            transform (callable, optional): Optional transforms to apply to input
            normalize (bool): Whether to normalize inputs
            
            Class labels:
            Dropout A: Something is obstructing the beam, shifting CM to near the transducer
            Dropout B: Something seems to be obstructing the beam, but CM is not shifted
            Dropout C: Something is very strongly covering the beam, but still picking up scatter beyond it (not sure it's even an issue) 
            MINOR A: Interference in the backscatter
            MINOR B: A beam is generally weaker
            I don't want to train on MINOR A or MINOR B, I only want to train on classes [0 to 3]
        """
        self.file_list = file_list
        self.transform = transform
        self.normalize = normalize
        self.label_strings = np.array(['Dropout A', 'Dropout B', 'Dropout C', 'MINOR A', 'MINOR B'])

        channel_names = ['velocity',
                         'backscatter',
                         'correlation']

        # Step 1: Load annotations and convert their datetimes to UNIX seconds
        #This loads ALL available annotations for the data set
        if annotations_mat_file != '':
            annotations_all = parse_annotations.load_matlab_annotations(annotations_mat_file)
        else:
            annotations_all = []
            
            
        # Preload all beam samples and labels
        self.samples = []
        for path in file_list:
            #print(path) #Debug
            #Load the beam data (vel, backscatter, corr), and time vector
            beam_data, timestamps = self._load_h5(path, channel_names)

            #Create an annotation array specific to each file
            ann_array = parse_annotations.subset_annotations(timestamps, annotations_all) 
            #Parse the annotation array for each file
            annotations = self._parse_annotations(ann_array)
        
            #Prepare beam data (Replace NaNs with 0s, create labels, collect meta data)
            for i, beam_stack in enumerate(beam_data):
                if beam_stack.shape != expected_shape:
                    print(f"[Skipped] {path} beam {i+1} has shape {beam_stack.shape}, expected {expected_shape}")
                    continue
                    
                # Replace NaNs with 0s before normalization
                beam_stack = np.nan_to_num(beam_stack, nan=0.0)
                if self.normalize:
                    beam_stack = self._normalize(beam_stack)
                labels = self._create_labels(timestamps, annotations, beam_number=i+1)
                # Skip if any labels are in excluded classes (4 or higher)
                if np.any(labels > 3):
                    print(f"[Skipped] {path} beam {i+1} has labels from excluded classes ")
                    continue
                    
                #Add a new variable, with "meta" data, not used for training, but useful for plots, etc
                # including beam number ensures correct beam number is matched to the annotations/predictions
                meta = {
                    'time': timestamps,
                    'filename': path,
                    'channels': channel_names, 
                    'beam_number': i+1, 
                }

                #Add data to the collection of samples
                self.samples.append((beam_stack, labels, meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, meta = self.samples[idx]

        if self.transform:
            x = self.transform(x)

        x = torch.tensor(x, dtype=torch.float32)   # shape [3, time, range]
        y = torch.tensor(y, dtype=torch.long)      # shape [time]
        
        return x, y, meta


    def _load_h5(self, path, channel_names):
        with h5py.File(path, 'r') as f:
            data = f['/data']
            beam_data = []

            for beam in [0, 1, 2]:
                v = data[channel_names[0]][beam]      # shape (time, range)
                b = data[channel_names[1]][beam]
                c = data[channel_names[2]][beam]
                #v = data['velocity'][beam]      # shape (time, range)
                #b = data['backscatter'][beam]
                #c = data['correlation'][beam]

                beam_stack = np.stack([v, b, c], axis=0)  # [3, time, range]
                beam_data.append(beam_stack)

            timestamps = f['/data/time'][:]
            return beam_data, timestamps

    def _normalize(self, x):
        return (x - np.mean(x, axis=(1, 2), keepdims=True)) / (np.std(x, axis=(1, 2), keepdims=True) + 1e-8)

    def _parse_annotations(self, annotations_array):
        annotations = []
        #has_comments = 'comment' in annotations_array # Check if a comments field exists
        
        #if has_comments:
        
        for i in range(len(annotations_array['start_index'])):
            #Extract class name
            class_name = annotations_array['class'][i].decode('utf-8') if isinstance(annotations_array['class'][i], bytes) else annotations_array['class'][i]
            # Extract beam number using regex
            comment = annotations_array['comment'][i]
            comment = comment.decode('utf-8') if isinstance(comment, bytes) else comment #decode from bytes object (common in h5) to python string, if necessary

            # Look for *all* beam mentions (could be more than one)
            beam_matches = re.findall(r'\bbeam\s*(\d)', comment, re.IGNORECASE)
            beam_nums = sorted(set(int(b) for b in beam_matches)) if beam_matches else [1, 2, 3]  # Assign to all if no valid beam found

            # In case the comment is empty, or doesn't specify a beam, 
            #  check for "beam" key in the annotations dict
            if not beam_nums:
                beam_nums = annotations_array['beam'][i]
                
            # Look for 1 beam mentioned
            # match = re.search(r'\bbeam\s*(\d)', comment, re.IGNORECASE)
            # beam_num = int(match.group(1)) if match else None

            # Add one annotation entry per beam
            for beam_num in beam_nums:
                annotations.append({
                    'start_idx': int(annotations_array['start_index'][i]),
                    'end_idx': int(annotations_array['end_index'][i]),
                    'class': class_name,
                    'beam': beam_num
                })
                
        return annotations

    def _create_labels(self, timestamps, annotations, beam_number):
        labels = np.zeros(len(timestamps), dtype=np.int64)

        # Only include annotations for the current beam
        filtered_annotations = [ann for ann in annotations if ann['beam'] == beam_number]

        for ann in filtered_annotations:
            try:
                cls = np.where(ann['class'] == self.label_strings)[0][0] + 1
                if cls<4: #Only add labels for Drop-outs for now.  Otherwise we have multi-class stuff cropping up
                  labels[ann['start_idx']:ann['end_idx']] = cls #Label all data between start and end indices
            except IndexError:
                print(f"[Warning] Unknown class label: {ann['class']}")
        return labels