import numpy as np
import scipy.io
from datetime import datetime, timezone
from dateutil import parser as dateparse

#Note: This used to exist within "split_h5_to_24hr_files.py" but was moved into it's own function when created "split_h5_to_24hr_files_noEmbed.m"

# ---- Annotation extraction and conversion from MATLAB ----
def load_matlab_annotations(mat_path):
    mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    ann_struct = mat['annotations']
    
    # Ensure iterable
    if ann_struct.dtype.names:  # single annotation
        ann_struct = [ann_struct]

    annotations = []
    for a in ann_struct:
        #Get and parse dates:
        start_str = str(getattr(a, 'startDate'))
        end_str   = str(getattr(a, 'endDate'))
        start_dt = dateparse.parse(start_str.replace('.0','')).replace(tzinfo=timezone.utc)
        end_dt   = dateparse.parse(end_str.replace('.0','')).replace(tzinfo=timezone.utc)
        
        #Populate an annotation dict:
        annotation_dict = {}
        annotation_dict['class'] = str(getattr(a, 'class'))
        annotation_dict['startDate'] = start_str # str(getattr(a, 'startDate'))
        annotation_dict['endDate'] = end_str # str(getattr(a, 'endDate'))
        annotation_dict['comment'] = str(getattr(a, 'comment'))
        #dct['status'] = str(getattr(a, 'status'))
        # Parse datetime (assumes UTC, adjust if needed)
        annotation_dict['start_datetime'] = start_dt # dateparse.parse(annotation_dict['startDate'].replace('.0','')).replace(tzinfo=timezone.utc)
        annotation_dict['end_datetime']   = end_dt # dateparse.parse(annotation_dict['endDate'].replace('.0','')).replace(tzinfo=timezone.utc)
        # Convert to int seconds since epoch (UTC, for fast search)
        annotation_dict['start_time_sec'] = int(start_dt.timestamp()) # int(annotation_dict['start_datetime'].timestamp())
        annotation_dict['end_time_sec'] = int(end_dt.timestamp()) # int(annotation_dict['end_datetime'].timestamp())
        
        #Append the annotation
        annotations.append(annotation_dict)
        
    return annotations




def subset_annotations(time, annotations):
    
    #Given a 24h time vector and a list of annotations,
    #return annotation array with indices relative to this file.
    
    #Using the time vector from a 24hr h5 file, get indices corresponding to the annotation times
    #This will ensure that if an annotations starts before or ends after the file start/end times, it's properly handled

    #NOTE; This was copied/altered from split_h5_to_24hr_files.split_h5_to_24hr_files_with_ann

    # Step 1: Convert time to int64 seconds
    if np.issubdtype(time.dtype, np.floating):  # likely MATLAB datenum
        time_sec = np.round((time - 719529) * 86400).astype('int64') #719529 is 1970,01,01
    elif np.issubdtype(time.dtype, np.datetime64):
        # If HDF stores as np.datetime64, convert to int seconds
        time_sec = time.astype('datetime64[s]').astype('int64')
    else:
        time_sec = time.astype('int64')  # already in seconds

    # Now, time_sec is ALWAYS seconds (int64)
    # If you wish, also get as datetime64 for some operations:
    # time_dt64 = time_sec.astype('datetime64[s]')

    # --- Optional: verify sampling interval is ~5 min---
    diffs = np.diff(time_sec)
    if len(diffs) > 0:
        median_dt = np.median(diffs)
        if not np.allclose(median_dt, 300, atol=2):   # 2s tolerance for rounding
            raise ValueError(
                f"Input file does not have a 5-minute interval (median step = {median_dt} s)"
            )

    # Step 3: Get the start and end times of the file (aka each 24 hr segment)
    seg_time = time_sec
    segment_start = time_sec[0]
    segment_end   = time_sec[-1]
    
    # Step 4: Get relevant info for the annotations
    ann_rows = []
    for a in annotations:
        # If annotation overlaps this file (or segment)
        if a['end_time_sec'] >= segment_start and a['start_time_sec'] < segment_end:
            # Restrict (clip) indices to chunk bounds, relative to seg_time
            sidx = np.searchsorted(seg_time, a['start_time_sec'], side='left')
            eidx = np.searchsorted(seg_time, a['end_time_sec'], side='right') - 1
            
            sidx = max(sidx, 0)
            eidx = min(eidx, len(seg_time) - 1)
            
            ann_rows.append((
                a['class'].encode('utf8'),
                a['start_time_sec'],
                a['end_time_sec'],
                sidx,
                eidx,
                a['comment'].encode('utf8')
            ))

    # Structured dtype for HDF5
    ann_dtype = np.dtype([
        ('class','S32'),
        ('start_time','i8'),
        ('end_time','i8'),
        ('start_index','i8'),
        ('end_index','i8'),
        ('comment','S255'),
        #('status','S16')
    ])
    ann_array = np.array(ann_rows, dtype=ann_dtype) if ann_rows else np.zeros((0,), dtype=ann_dtype)

    return ann_array

        


'''
I think this is deprecated, but don't want to delete until after testing code to be sure it all works

def parse_annotations(self, annotations_group):
    annotations = []
    #has_comments = 'comment' in annotations_group # Check if a comments field exists
    
    #if has_comments:
    
    for i in range(len(annotations_group['start_index'])):
        #Extract class name
        class_name = annotations_group['class'][i].decode('utf-8') if isinstance(annotations_group['class'][i], bytes) else annotations_group['class'][i]
        # Extract beam number using regex
        comment = annotations_group['comment'][i]
        comment = comment.decode('utf-8') if isinstance(comment, bytes) else comment #decode from bytes object (common in h5) to python string, if necessary

        # Look for *all* beam mentions (could be more than one)
        beam_matches = re.findall(r'\bbeam\s*(\d)', comment, re.IGNORECASE)
        beam_nums = sorted(set(int(b) for b in beam_matches)) if beam_matches else [1, 2, 3]  # Assign to all if no valid beam found

        # Look for 1 beam mentioned
        # match = re.search(r'\bbeam\s*(\d)', comment, re.IGNORECASE)
        # beam_num = int(match.group(1)) if match else None

        # Add one annotation entry per beam
        for beam_num in beam_nums:
            annotations.append({
                'start_idx': int(annotations_group['start_index'][i]),
                'end_idx': int(annotations_group['end_index'][i]),
                'class': class_name,
                'beam': beam_num
            })
            
    return annotations
'''