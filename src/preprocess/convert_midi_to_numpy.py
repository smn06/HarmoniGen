import os
import numpy as np
import pretty_midi

def midi_to_notes(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append([note.start, note.end, note.pitch, note.velocity])
    return np.array(notes)

def save_notes_to_numpy(data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for genre in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre)
        if os.path.isdir(genre_path):
            for midi_file in os.listdir(genre_path):
                if midi_file.endswith('.mid'):
                    midi_path = os.path.join(genre_path, midi_file)
                    notes = midi_to_notes(midi_path)
                    save_path = os.path.join(save_dir, f"{genre}_{midi_file.replace('.mid', '.npy')}")
                    np.save(save_path, notes)

if __name__ == "__main__":
    data_directory = '../data/'
    save_directory = '../data_numpy/'
    save_notes_to_numpy(data_directory, save_directory)
