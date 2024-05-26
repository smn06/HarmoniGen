import os
import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras.models import load_model

def generate_samples(generator_model_path, output_dir, latent_dim=100, n_samples=10):
    generator = load_model(generator_model_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(n_samples):
        noise = np.random.normal(0, 1, (1, latent_dim))
        generated_data = generator.predict(noise)
        save_generated_midi(generated_data, output_dir, i)

def save_generated_midi(generated_data, output_dir, index):
    piano_roll = np.squeeze(generated_data)
    midi_data = piano_roll_to_pretty_midi(piano_roll)
    output_file = os.path.join(output_dir, f"generated_sample_{index}.mid")
    midi_data.write(output_file)

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    notes, frames = piano_roll.shape
    threshold = 0.5  # Adjust as needed

    # Use numpy to find the indices of notes that are active
    for note in range(88):
        velocity_curve = piano_roll[note]
        velocity_changes = np.diff(velocity_curve, prepend=0)

        for i in range(1, len(velocity_changes)):
            if velocity_changes[i] == 1:
                start_time = i / fs
            elif velocity_changes[i] == -1:
                end_time = i / fs
                piano.notes.append(pretty_midi.Note(velocity=100, pitch=note, start=start_time, end=end_time))

    midi.instruments.append(piano)
    return midi

if __name__ == "__main__":
    generator_model_path = '../results/models/generator.h5'
    output_dir = '../results/generated_samples/music/'

    latent_dim = 100
    n_samples = 10

    generate_samples(generator_model_path, output_dir, latent_dim, n_samples)
