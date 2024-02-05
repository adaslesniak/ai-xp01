from NnModel import prepare_model
from DataGenerator import load_training_data, load_test_data, generate_data


def train():
    training_data, training_labels = load_training_data()
    model = prepare_model()
    _ = model.fit(training_data, 
                        training_labels, 
                        epochs=66, batch_size=48, validation_split=0.17)
    return model


def evaluate(trained_model):
    test_data, test_labels = load_test_data()
    test_loss, test_acc = trained_model.evaluate(test_data, test_labels)
    print(f"Test Accuracy: {test_acc*100:.2f}%, loss: {test_loss}")


#generate_data(99, 16)
the_thing = train()
evaluate(the_thing)