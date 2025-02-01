# Manual Logistic Regression Using Numpy & Gradient Descent

# Import necessary libraries and datasets
import numpy as np
import sklearn


# Return the Wisconsin breast cancer dataset from scikit-learn
def get_data():
    return sklearn.datasets.load_breast_cancer()


# Create the sigmoid function
def sigmoid(x):
    # x: The input into the sigmoid function

    return 1.0/(1+np.e**-x)


# Create the cost function using log loss
def cost(Y, Y_guess, N):
    # Y_guess: The model's guess of the example's classification
    # Y: The correct classification of the training example
    # N: The number of entries in the training sample

    # Compute the cost and reduce unwanted dimensions
    return -np.sum(Y * np.log(Y_guess) + (1-Y) * np.log(1-Y_guess))/N


# Get model's guesses of an example's classification
def guess(X, W, b):
    # W: The model's weights
    # X: The input vector
    # b: The biases

    # Return the guess using the sigmoid function
    return sigmoid(np.dot(W, X) + b)


# Get the gradients of the weights and biases
def get_gradients(X, Y, Y_guess):
    # Calculate gradients
    dC_dW = np.dot(X, Y_guess-Y)/len(X)
    dC_db = np.sum(Y_guess-Y)/len(X)

    return dC_dW, dC_db


# Use gradient descent to update the weights and bias parameters
def gradient_descent(X, Y, W, b, N, learning_rate=0.0001):
    # Iterate through training examples
    for i in range(N):
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}: ")

        current_cost = 0  # Reset previous cost

        for j in range(len(X[0])):
            if (i + 1) % 10 == 0:
                # Every 10 iterations, get and print current cost
                current_cost += cost(Y[j], guess(X[j], W, b), len(X))

            # Get the model guess
            Y_guess = guess(X[j], W, b)

            # Get the current gradients for the weights and biases
            dw, db = get_gradients(X[j], Y[j], Y_guess)

            # Update weights and biases
            W -= learning_rate * dw
            b -= learning_rate * db

        if (i + 1) % 10 == 0:
            print(f"Current Cost: {current_cost}")
            print()

    # Return parameters
    return W, b


# Get model prediction for given data
def predict(X, W, b):
    model_guess = guess(W, X, b)

    # If the model's guess is greater than 0.5, set its choice to 1 (true)
    # Otherwise, set it to 0 (false)
    if model_guess >= 0.5:
        return 1
    return 0


# Main method to train and test the model
if __name__ == "__main__":
    # Get training and testing data
    data = get_data()
    X = data['data']
    Y = data['target']

    # Split data into training and testing data
    # With 80% of it in the training set
    # and 20% in the testing set
    train_X, test_X, train_Y, test_Y = sklearn.model_selection.train_test_split(X, Y, train_size=0.8)

    # Initialize weights and biases
    W, b = np.zeros(X.shape[1]), 0

    # Train the model using gradient descent with 6000 iterations
    W, b = gradient_descent(train_X, train_Y, W, b, 10000)

    # Test the model with the testing data
    # and get it's overall accuracy
    accuracy = 0
    for i in range(len(test_X)):
        if predict(test_X[i], W, b) == test_Y[i]:
            accuracy += 100.0 / len(test_X)  # Increase the accuracy for every correct prediction

    # Output model accuracy
    print(f"Accuracy: {accuracy}%")
    print(f"  # of correct predictions: {round(accuracy * len(test_X) / 100.0)}")
    print(f"  # of wrong predictions: {round((100 - accuracy) * len(test_X) / 100.0)}")
