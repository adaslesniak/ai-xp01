from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


def prepare_model():
    model = Sequential([
        Dense(96, input_shape=(25,), activation='elu'),  # First hidden 
        Dense(128, activation='elu'),  # Second hidden 
        Dense(96, activation='elu'),  # Third hidden 
        Dense(48, activation='elu'),  # Third hidden 
        Dense(1, activation='sigmoid')  # Output for binary classification
    ])
    sgd_optimizer = SGD(learning_rate=0.007, momentum=0.66)
    model.compile(optimizer=sgd_optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model