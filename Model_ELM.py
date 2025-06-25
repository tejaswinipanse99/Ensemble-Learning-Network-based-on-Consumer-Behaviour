import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Evaluation import evaluate_error


class ELMClassifier:
    def __init__(self, input_size, hidden_layer_size, output_size, Act):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        # Randomly initialize input-to-hidden weights and biases
        self.input_to_hidden_weights = np.random.rand(hidden_layer_size, input_size)
        self.input_to_hidden_biases = np.random.rand(hidden_layer_size, 1)

        # Initialize hidden-to-output weights (will be updated during training)
        self.hidden_to_output_weights = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X_train, y_train, epochs=1):
        for epoch in range(epochs):
            # Calculate hidden layer outputs
            hidden_outputs = self._sigmoid(
                np.dot(self.input_to_hidden_weights, X_train.T) + self.input_to_hidden_biases)

            # Add a bias term to hidden layer outputs
            hidden_outputs = np.vstack([hidden_outputs, np.ones((1, hidden_outputs.shape[1]))])

            # Calculate output layer weights using Moore-Penrose pseudoinverse
            self.hidden_to_output_weights = np.dot(np.linalg.pinv(hidden_outputs.T), y_train)

    def predict(self, X_test):
        # Calculate hidden layer outputs for testing data
        hidden_outputs = self._sigmoid(np.dot(self.input_to_hidden_weights, X_test.T) + self.input_to_hidden_biases)

        # Add a bias term to hidden layer outputs
        hidden_outputs = np.vstack([hidden_outputs, np.ones((1, hidden_outputs.shape[1]))])

        # Calculate predicted outputs
        predicted_outputs = np.dot(hidden_outputs.T, self.hidden_to_output_weights)

        return predicted_outputs.argmax(axis=1)


def Model_ELM(X_train, y_train, X_test, y_test, Act=None, sol=None):
    if Act is None:
        Act = 'sigmoid'
    if sol is None:
        sol = [5, 5, 5]

    input_size = 20
    hidden_layer_size = int(sol[2])
    output_size = y_train.shape[-1]

    Train_Temp = np.zeros((X_train.shape[0], input_size))
    for i in range(X_train.shape[0]):
        Train_Temp[i, :] = np.resize(X_train[i], (input_size))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], input_size)

    Test_Temp = np.zeros((X_test.shape[0], input_size))
    for i in range(X_test.shape[0]):
        Test_Temp[i, :] = np.resize(X_test[i], input_size)
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], input_size)


    elm_classifier = ELMClassifier(input_size, hidden_layer_size, output_size, Act)
    elm_classifier.train(Train_X, y_train, epochs=5)
    pred = elm_classifier.predict(Test_X)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluate_error(np.reshape(pred, (-1, 1)), y_test)
    return Eval, pred
