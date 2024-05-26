import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim):
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))

    # Adding layers to reach 32 layers
    num_layers = 30  # We already have input and reshape as 2 layers
    for _ in range(num_layers // 6):
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Reshape((16, 16, 1)))

    for _ in range(num_layers // 6):
        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
    
    for _ in range(num_layers // 6):
        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))

    for _ in range(num_layers // 6):
        model.add(layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
    
    # Output layer
    model.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', activation='tanh'))

    return model

def build_discriminator(input_shape):
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    # Adding layers to reach 32 layers
    num_layers = 30  # We already have input and flatten as 2 layers
    for _ in range(num_layers // 6):
        model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
    
    for _ in range(num_layers // 6):
        model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))

    for _ in range(num_layers // 6):
        model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
    
    # Output layer
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def compile_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    discriminator.trainable = False

    z = layers.Input(shape=(100,))
    generated = generator(z)
    valid = discriminator(generated)

    combined = tf.keras.models.Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    return combined
