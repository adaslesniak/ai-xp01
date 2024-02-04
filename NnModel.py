import tensorflow as tf
print(tf.__version__)  # Check TensorFlow version
#print(tf.keras.__version__)  # Check Keras version integrated within TensorFlow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def prepare_model():
    model = Sequential([
        Dense(64, input_shape=(25,), activation='relu'),  # First hidden layer with 64 neurons
        Dense(64, activation='relu'),  # Second hidden layer with 64 neurons
        Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model