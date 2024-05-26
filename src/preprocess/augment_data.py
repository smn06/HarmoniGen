import os
import numpy as np

def transpose_notes(notes, semitones):
    notes[:, 2] = notes[:, 2] + semitones
    return notes

def augment_data(data_dir, save_dir, transpositions):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for npy_file in os.listdir(data_dir):
        if npy_file.endswith('.npy'):
            notes = np.load(os.path.join(data_dir, npy_file))
            for semitone in transpositions:
                transposed_notes = transpose_notes(notes.copy(), semitone)
                save_path = os.path.join(save_dir, f"{npy_file.replace('.npy', '')}_transposed_{semitone}.npy")
                np.save(save_path, transposed_notes)

if __name__ == "__main__":
    data_directory = '../data_normalized/'
    save_directory = '../data_augmented/'
    transpositions = [-2, -1, 1, 2]  # Example transpositions: up and down by 1 and 2 semitones
    augment_data(data_directory, save_directory, transpositions)
