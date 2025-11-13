"""
split_h5_to_24hr_files.py
Split HDF5 monthly files into daily files and attach annotations.

To use:
import split_h5_to_24hr_file

split_h5_to_24hr_file.split_h5_to_24hr_files_with_ann(
    input_file,             # your big HDF5 source (created with import_monthly_mat_to_h5.py)
    h5_24h_folder,          # output dir for 24hr files
    annotations_file,        # your .mat annotations file
)

"""
import h5py
import numpy as np
import scipy.io
from datetime import datetime, timezone
from dateutil import parser as dateparse
import os

#NOTES:
#Split up h5 monthly files
#Time handling: 
#Final block: Even if data starts (or ends) at 16:00, you'll get a block 16:00 to 15:59 (24 hours), so there may be overlap between different files
#Annotation adjustment: It slices/copies only relevant annotations for each chunk, adjusting start/end indices to local chunk coordinates.
#If you want random non-overlapping 24h blocks or a stride different from 24h, adjust the interval generator.

#Summary
#This process will output one .h5 per (possibly-overlapping) 24h window.
#Each .h5 has its own data/time, other data fields, and relevant annotations.
#ML models can now easily train on chunked, uniform shape input files.



# ---- Splitter ----
def split_h5_to_24hr_files(input_filename, out_dir, time_ds_path='/data/time'):

    with h5py.File(input_filename, 'r') as h5in:
        # Step 2: Load main time vector & fields
        time = h5in[time_ds_path][:]
        
        if np.issubdtype(time.dtype, np.floating):  # likely MATLAB datenum
            time_sec = np.round((time - 719529) * 86400).astype('int64') #719529 is 1970,01,01
            #def datenum_to_epochsecs(dn):
            #    return np.round((dn - 719529) * 86400).astype('int64')
            #time_sec = datenum_to_epochsecs(time)
        else:
            if np.issubdtype(time.dtype, np.datetime64):
                # If HDF stores as np.datetime64, convert to int seconds
                time_sec = time.astype('datetime64[s]').astype('int64')
            else:
                time_sec = time.astype('int64')  # already in seconds

        # Now, time_sec is ALWAYS seconds (int64)
        # If you wish, also get as datetime64 for some operations:
        # time_dt64 = time_sec.astype('datetime64[s]')

        #Verify that the file is:
        # 1) At least 24 hours long
        # 2) Has a 5 minute sample interval
                
        # 1. Check duration
        dur_s = time_sec[-1] - time_sec[0]      # total duration in seconds
        if dur_s < 86400:
            raise ValueError(f"Input file spans less than 24 hours ({dur_s/3600:.2f} hours)!")

        # 2. Check (average) interval
        diffs = np.diff(time_sec)
        median_dt = np.median(diffs)
        if not np.allclose(median_dt, 300, atol=2):   # 2s tolerance for rounding
            raise ValueError(
                f"Input file does not have a 5-minute interval (median step = {median_dt} s)"
            )
        
        # Load all other /data datasets
        data_group = h5in['/data']
        #main_fields = {k: data_group[k][:] for k in data_group if k != "time"}
        

        # Step 3: Compute the split points (midnight to midnight UTC, last segment aligned at end)
        start_all = time_sec[0]
        end_all = time_sec[-1]
        #time_dt64 = time_sec.astype('datetime64[s]')
        #start_all = time_dt64[0]
        #end_all   = time_dt64[-1]

        # Use UTC midnights for the intervals
        start_day = datetime.fromtimestamp(start_all, tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start_day_sec = int(start_day.timestamp())
        intervals = []
        current_start = start_day_sec

        # Round start to nearest midnight after/before as appropriate
        #day0 = datetime.fromtimestamp(time_sec[0].UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        #day0 = datetime.utcfromtimestamp(time_sec[0]).replace(hour=0, minute=0, second=0, microsecond=0)
        #day0 = np.datetime64(day0, 's')
        #intervals = []
        #current_start = day0

        # Iterate 24h windows, these may have overlap (start and end) but that's okay!
        
        #First, ensure that the first entry is 24 hours, even if not starting UTC midnight,
        if current_start < start_all:
            # Output a "first chunk" from start_all for 24h
            intervals.append((start_all, start_all + 86400 - 1))
            current_start += 86400

        # then do all other intervals so that they are utc midnight
        while current_start + 86400 - 1 <= end_all:
            intervals.append((current_start, current_start + 86400 - 1))
            current_start += 86400

        # Add tail if needed that is also 24 hours
        if end_all > intervals[-1][1]:
            intervals.append((end_all - 86400 + 1, end_all))

        # Step 4: For each interval, slice data and embed annotations
        os.makedirs(out_dir, exist_ok=True)
        for segment_start, segment_end in intervals:
            # Indices to slice data arrays
            left_idx = np.searchsorted(time_sec, segment_start, side="left")
            right_idx = np.searchsorted(time_sec, segment_end, side="right")
            
            # Guarantee length is 24h = 288 samples (minutes) if possible, else pad last
            #required_len = end_sec - start_sec + 1
            # ---- slice and pad main arrays ---- => NO PADDING!
            #Slice time 
            seg_time = time_sec[left_idx:right_idx] 
            sample_dt = np.median(np.diff(time_sec))   # e.g., 300
            samples_per_window = int(np.round((segment_end - segment_start) / sample_dt)) #e.g. 288 # + 1  # e.g., 289
            #samples_per_window = int(np.round((end_sec - start_sec) / sample_dt)) 
            shift_len = samples_per_window - len(seg_time)
            #pad_len = samples_per_window - len(seg_time)

            if right_idx != len(time_sec): #if right index is not the last index in the month-long file, extend the segment to the right
                right_idx = right_idx + shift_len
            else: #elif left_idx == 0:     #otherwise, extend the index to the left.  
                left_idx = left_idx - shift_len
           
            #Update sliced time:
            seg_time = time_sec[left_idx:right_idx] 

            #Check pad_len
            pad_len = samples_per_window - len(seg_time)
            if pad_len != 0:
                raise ValueError(f"dA hours)!")

            #Slice and pad Data - Will find whichever axis is same length as time:
            main_fields = {}
            for k in data_group:
                dset = data_group[k]
                if dset.shape == ():  # scalar
                    main_fields[k] = dset[()]  # store as is
                    continue
                # Find the axis whose length matches time
                matching_axes = [i for i, sz in enumerate(dset.shape) if sz == len(time_sec)]
                if len(matching_axes) == 1:
                    time_axis = matching_axes[0]
                    # Prepare slices: [:, :, :, ...] but time_axis gets slice(left_idx, right_idx)
                    slicer = [slice(None)] * dset.ndim
                    slicer[time_axis] = slice(left_idx, right_idx)
                    main_fields[k] = dset[tuple(slicer)]
                elif len(matching_axes) == 0:
                    # Not time dependent; save as is
                    main_fields[k] = dset[()]
                    print(f"Retaining static dataset: {k} shape {dset.shape}")
                else:
                    print(f"Warning: {k} has multiple axes matching len(time); skipping for safety")
                    # Handle as you see fit

            #seg_fields = {k: arr[left_idx:right_idx] for k, arr in main_fields.items()}
            
            #if pad_len > 0:
            #    seg_time = np.pad(seg_time, (0, pad_len), 'edge')
            #    for k in main_fields:
            #        arr = main_fields[k]
            #        if isinstance(arr, np.ndarray) and arr.shape[0] == len(seg_time) - pad_len:
            #            main_fields[k] = np.pad(arr, ((0, pad_len),), 'edge')

                #for k in seg_fields:
                #    seg_fields[k] = np.pad(seg_fields[k], (0, pad_len), 'edge')
            

            # ---- SAVE output ----
            outfn = os.path.join(
                out_dir, 
                f"{datetime.utcfromtimestamp(segment_start):%Y%m%dT%H%M%S}_"
                f"{datetime.utcfromtimestamp(segment_end):%Y%m%dT%H%M%S}.h5"
            )
            with h5py.File(outfn, 'w') as h5out:
                #Save the data group
                dgrp = h5out.create_group('data')
                #Save the time variable
                dgrp.create_dataset("time", data=seg_time)  ### <---- Use "time" in seconds, not the MATLAB time
                for k, arr in main_fields.items():
                    if np.isscalar(arr):
                        # Save scalars as attributes
                        dgrp.attrs[k] = arr
                    elif k != "time":                       ### <---- Don't write time again
                        #Save all other variables
                        dgrp.create_dataset(k, data=arr)
               
                #Add the other descriptor datasets
                h5in.copy('/meta', h5out)
                h5in.copy('/units', h5out)
                h5in.copy('/config', h5out)
                #h5out.create_dataset("meta", h5in['/meta'])
                #h5out.create_dataset("units", h5in['/units'])
                #h5out.create_dataset("config", h5in['/config'])

#if __name__ == "__main__":
#    split_h5_to_24hr_files_with_ann(
#        args.input_filename, args.out_dir, args.annotation_mat_file, args.time_ds_path
