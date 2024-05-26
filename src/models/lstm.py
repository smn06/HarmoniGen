import tensorflow as tf
from tensorflow.keras import layers

def build_lstm_model(input_shape, output_shape):
    model = tf.keras.Sequential()

    # Adding LSTM layers to reach 32 layers
    for _ in range(16):  # 16 layers of LSTM + 16 layers of Dropout = 32 layers
        model.add(layers.LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(layers.Dropout(0.3))

    model.add(layers.LSTM(128))
    model.add(layers.Dropout(0.3))
    
    # Output layer
    model.add(layers.Dense(output_shape, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
