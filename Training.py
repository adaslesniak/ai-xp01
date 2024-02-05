from NnModel import prepare_model
from DataGenerator import load_training_data, load_test_data


def train():
    training_data, training_labels = load_training_data()
    model = prepare_model()
    history = model.fit(training_data, 
                        training_labels, 
                        epochs=48, batch_size=24, validation_split=0.16)
    print(history)
    return model


def evaluate_accuracy(trained_model):
    test_data, test_labels = load_test_data()
    test_loss, test_acc = trained_model.evaluate(test_data, test_labels)
    print(f"Test Accuracy: {test_acc*100:.2f}%, loss: {test_loss}")


the_thing = train()
evaluate_accuracy(the_thing)