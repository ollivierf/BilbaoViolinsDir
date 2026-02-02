import h5py
import numpy as np
import os
from filelock import FileLock

# Filepath for the HDF5 file
h5_filepath = 'MACCoef.h5'

# Initialize the HDF5 file with a resizable dataset and a status map
if not os.path.exists(h5_filepath):
    with h5py.File(h5_filepath, 'w') as f:
        # Create a resizable dataset with unlimited dimensions
        dset = f.create_dataset(
            'matrix',
            shape=(0, 0, 0, 0),  # Initial shape: (n, n, f, f)
            maxshape=(None, None, None, None),  # Unlimited growth
            chunks=(1, 1, 10, 10),  # Chunk size for efficient I/O
            dtype='float64'
        )
        # Create a status map with the same dimensions as the first two axes of 'matrix'
        status_map = f.create_dataset(
            'status_map',
            shape=(0, 0),  # Initial shape: (n, n)
            maxshape=(None, None),  # Unlimited growth
            chunks=(1, 1),  # Chunk size for efficient I/O
            dtype='int8'  # 0: unprocessed, 1: processed
        )

def get_frequency_index(frequency, frequencies):
    return np.argmin(np.abs(frequencies - frequency))

# Function to update a specific patch in the HDF5 file
def update_patch(i, j, patch_data, frequencies, f_start, f_end):
    lock = FileLock(f"{h5_filepath}.lock")  # Create a lock file
    with lock:  # Ensure only one process writes at a time
        with h5py.File(h5_filepath, 'a', libver='latest', swmr=True) as f:
            dset = f['matrix']
            status_map = f['status_map']

            # Resize the datasets if necessary
            n, _, f1, f2 = dset.shape
            new_n = max(n, i + 1, j + 1)
            new_f_start = get_frequency_index(f_start, frequencies)
            new_f_end = get_frequency_index(f_end, frequencies)
            new_f = max(f1, new_f_end + 1)
            dset.resize((new_n, new_n, new_f, new_f))
            status_map.resize((new_n, new_n))

            # Write the patch data
            dset[i, j, new_f_start:new_f_start + patch_data.shape[0], new_f_start:new_f_start + patch_data.shape[1]] = patch_data

            # Mark the patch as processed in the status map
            status_map[i, j] = 1

# Function to check unprocessed patches
def get_unprocessed_patches():
    with h5py.File(h5_filepath, 'r') as f:
        status_map = f['status_map'][:]
        unprocessed = np.argwhere(status_map == 0)  # Find indices of unprocessed patches
    return unprocessed

# Example usage
# update_patch(2, 3, patch_data, frequencies, f_start, f_end)
# unprocessed_patches = get_unprocessed_patches()
# print("Unprocessed patches:", unprocessed_patches)