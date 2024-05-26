import os
import numpy as np

def normalize_notes(notes):
    notes[:, 0] = notes[:, 0] - notes[:, 0].min()  # Normalize start times
    notes[:, 1] = notes[:, 1] - notes[:, 1].min()  # Normalize end times
    notes[:, 2] = (notes[:, 2] - notes[:, 2].min()) / (notes[:, 2].max() - notes[:, 2].min())  # Normalize pitches
    notes[:, 3] = notes[:, 3] / 127.0  # Normalize velocity
    return notes

def normalize_data(data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for npy_file in os.listdir(data_dir):
        if npy_file.endswith('.npy'):
            notes = np.load(os.path.join(data_dir, npy_file))
            normalized_notes = normalize_notes(notes)
            np.save(os.path.join(save_dir, npy_file), normalized_notes)

if __name__ == "__main__":
    data_directory = '../data_numpy/'
    save_directory = '../data_normalized/'
    normalize_data(data_directory, save_directory)
