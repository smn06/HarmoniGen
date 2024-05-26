import os
import numpy as np
import tensorflow as tf
from dcgan import build_generator, build_discriminator, compile_gan
from lstm import build_lstm_model

def load_data(data_dir):
    data = []
    for npy_file in os.listdir(data_dir):
        if npy_file.endswith('.npy'):
            data.append(np.load(os.path.join(data_dir, npy_file)))
    return np.array(data)

def train_dcgan(generator, discriminator, combined, data, epochs, batch_size):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]

        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, valid)
        d_loss_fake = discriminator.train_on_batch(generated_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = combined.train_on_batch(noise, valid)

        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")

def train_lstm(lstm_model, data, epochs, batch_size):
    x_train = data[:, :-1]
    y_train = data[:, -1]

    lstm_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    latent_dim = 100
    input_shape = (32, 32, 1)
    output_shape = 88

    generator = build_generator(latent_dim)
    discriminator = build_discriminator(input_shape)
    gan = compile_gan(generator, discriminator)

    lstm_model = build_lstm_model((32, 32), output_shape)

    data_dir = '../data_augmented/classical/'
    data = load_data(data_dir)

    epochs = 10000
    batch_size = 32

    train_dcgan(generator, discriminator, gan, data, epochs, batch_size)
    train_lstm(lstm_model, data, epochs, batch_size)
