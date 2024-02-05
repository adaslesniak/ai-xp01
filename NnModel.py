from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop


def prepare_model():
    model = Sequential([
        Dense(64, input_shape=(25,), activation='relu'),  # First hidden layer with 64 neurons
        Dense(64, activation='relu'),  # Second hidden layer with 64 neurons
        Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])
    the_optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=the_optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model