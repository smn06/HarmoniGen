import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

def evaluate_generator(generator_model_path, n_samples=10, latent_dim=100):
    generator = load_model(generator_model_path)
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated_samples = generator.predict(noise)
    return generated_samples

def evaluate_discriminator(discriminator_model_path, real_data, n_samples=10, latent_dim=100):
    discriminator = load_model(discriminator_model_path)
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated_data = generator.predict(noise)

    real_labels = np.ones((n_samples, 1))
    fake_labels = np.zeros((n_samples, 1))

    real_score = discriminator.evaluate(real_data[:n_samples], real_labels, verbose=0)
    fake_score = discriminator.evaluate(generated_data, fake_labels, verbose=0)
    
    print(f"Real data evaluation: Loss={real_score[0]}, Accuracy={real_score[1]*100}%")
    print(f"Fake data evaluation: Loss={fake_score[0]}, Accuracy={fake_score[1]*100}%")

def evaluate_lstm(lstm_model_path, test_data):
    lstm_model = load_model(lstm_model_path)
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    predictions = lstm_model.predict(x_test)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
    print(f"LSTM model accuracy: {accuracy * 100}%")

if __name__ == "__main__":
    latent_dim = 100
    n_samples = 10

    generator_model_path = '../results/models/generator.h5'
    discriminator_model_path = '../results/models/discriminator.h5'
    lstm_model_path = '../results/models/lstm_model.h5'
    test_data_dir = '../data_augmented/classical/'

    # Load test data
    test_data = []
    for npy_file in os.listdir(test_data_dir):
        if npy_file.endswith('.npy'):
            test_data.append(np.load(os.path.join(test_data_dir, npy_file)))
    test_data = np.array(test_data)

    # Evaluate models
    print("Evaluating Generator...")
    generated_samples = evaluate_generator(generator_model_path, n_samples, latent_dim)
    print(f"Generated {len(generated_samples)} samples.")

    print("Evaluating Discriminator...")
    evaluate_discriminator(discriminator_model_path, test_data, n_samples, latent_dim)

    print("Evaluating LSTM...")
    evaluate_lstm(lstm_model_path, test_data)
