"""

convert_monthly_mat_to_h5.py
CONVERT MAT to h5

To use:
import convert_monthly_mat_to_h5

extract_mat_to_h5(mat_path, output_folder) 
"""

import os
import scipy.io
import numpy as np
import h5py

# Helper function to convert MATLAB structs to Python dicts
def matlab_struct_to_dict(matobj):
    d = {}
    for fieldname in matobj._fieldnames:
        value = getattr(matobj, fieldname)
        if isinstance(value, np.ndarray) and value.dtype.names:
            # Nested struct array
            d[fieldname] = [matlab_struct_to_dict(el) for el in value]
        elif hasattr(value, '_fieldnames'):
            d[fieldname] = matlab_struct_to_dict(value)
        else:
            d[fieldname] = value
    return d

# 2. Preview each field (meta, data, units, config)
def preview_struct(struct, name):
    print(f"\n== {name.upper()} ==")
    if hasattr(struct, '_fieldnames'):
        for fname in struct._fieldnames:
            value = getattr(struct, fname)
            if isinstance(value, np.ndarray):
                print(f"  {fname}: array shape {value.shape}, dtype {value.dtype}")
            elif hasattr(value, '_fieldnames'):
                print(f"  {fname}: nested struct")
            else:
                print(f"  {fname}: {type(value)} ({value})")
    else:
        print(struct)


def save_dict(h5group, d):
    #Helper function for saving the outputs
    for k, v in d.items():
        if isinstance(v, dict):
            g = h5group.create_group(k)
            save_dict(g, v)
        elif isinstance(v, np.ndarray) and v.dtype.names:  # structured array
            for name in v.dtype.names:
                h5group.create_dataset(f"{k}/{name}", data=v[name])
        else:
            try:
                h5group.create_dataset(k, data=v)
            except TypeError:
                try:
                    h5group.create_dataset(k, data=str(v))
                except Exception:
                    pass


def extract_mat_to_h5(mat_path, output_folder):
    # 1. Load the .mat file
    mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    for key in ['meta', 'data', 'units', 'config']:
        preview_struct(mat[key], key)

    # 3. Extract fields into Python objects
    meta = matlab_struct_to_dict(mat['meta'])
    data = matlab_struct_to_dict(mat['data'])
    units = matlab_struct_to_dict(mat['units'])
    config = matlab_struct_to_dict(mat['config'])

    # Quick preview main data time series
    print("\n=== DATA FIELD SAMPLES ===")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape {v.shape}, dtype {v.dtype}")
        else:
            print(f"{k}: {type(v)} ({v})")

    # 4. Save to HDF5 for future use
            
    # Create folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate new filename with .h5 extension
    filename = os.path.basename(mat_path)
    output_filename = os.path.splitext(filename)[0] + '.h5'
    h5_output_path = os.path.join(output_folder, output_filename)
    #print(f"Output path: {output_path}")
    #h5_output_filename = 'organized_data.h5'

    with h5py.File(h5_output_path, 'w') as h5f:
        save_dict(h5f.create_group('meta'), meta)
        save_dict(h5f.create_group('data'), data)
        save_dict(h5f.create_group('units'), units)
        save_dict(h5f.create_group('config'), config)

    print("\n✅ Data extracted and saved as organized_data.h5!")