# Manual Logistic Regression Using Numpy & Gradient Descent

# Import necessary libraries and datasets
import numpy as np
import sklearn
import time
import matplotlib.pyplot as plt


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
    return sigmoid(np.dot(X, W) + b)


# Get the gradients of the weights and biases
def get_gradients(X, Y, Y_guess):
    # Calculate gradients

    dC_dW = np.dot(X.T, Y_guess-Y)/len(X)
    dC_db = np.sum(Y_guess-Y)/len(X)

    return dC_dW, dC_db


# Use gradient descent to update the weights and bias parameters
def gradient_descent(X, Y, W, b, N, learning_rate=0.0001):
    costs = []  # Store the cost every 10 iterations for the loss curve

    # Iterate through training examples
    for i in range(N):
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}: ")

        current_cost = 0  # Reset previous cost

        for j in range(len(X)):
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
            costs.append(current_cost)

    # Return parameters
    return W, b, costs


# Use vectorized gradient descent to update the weights and bias parameters
# (uses every training example at once with matrix operations instead of a loop)
def gradient_descent_vectorized(X, Y, W, b, N, learning_rate=0.1):
    costs = []  # Store the cost every 10 iterations for the loss curve

    for i in range(N):
        # Get the model's guesses for every training example at once
        Y_guess = guess(X, W, b)

        if (i + 1) % 10 == 0:
            # Every 10 iterations, store the current cost
            costs.append(cost(Y, Y_guess, len(X)))

        # Get the current gradients for the weights and biases
        dw, db = get_gradients(X, Y, Y_guess)

        # Update weights and biases
        W -= learning_rate * dw
        b -= learning_rate * db

    # Return parameters
    return W, b, costs


# Get model prediction for given data
def predict(X, W, b):
    model_guess = guess(X, W, b)

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
    # (random_state fixes the split so results are the same every run)
    train_X, test_X, train_Y, test_Y = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state=42)

    # Standardize the features so they're all on the same scale
    # (fit on the training set only, so no information from the testing set leaks in)
    scaler = sklearn.preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # Initialize weights and biases
    W, b = np.zeros(X.shape[1]), 0

    # Train the model using gradient descent with 10000 iterations and time how long it takes
    start_time = time.time()
    W, b, costs = gradient_descent(train_X, train_Y, W, b, 10000)
    loop_time = time.time() - start_time

    # Train a second model using vectorized gradient descent with 10000 iterations
    # starting from fresh weights and biases
    start_time = time.time()
    W_vec, b_vec, costs_vec = gradient_descent_vectorized(train_X, train_Y, np.zeros(X.shape[1]), 0, 10000)
    vectorized_time = time.time() - start_time

    # Test the model with the testing data and get it's overall accuracy
    accuracy = 0
    for i in range(len(test_X)):
        if predict(test_X[i], W, b) == test_Y[i]:
            accuracy += 100.0 / len(test_X)  # Increase the accuracy for every correct prediction

    # Output model accuracy
    print(f"Accuracy: {accuracy}%")
    print(f"  # of correct predictions: {round(accuracy * len(test_X) / 100.0)}")
    print(f"  # of wrong predictions: {round((100 - accuracy) * len(test_X) / 100.0)}")
    print()

    # Get precision and recall for malignant tumours (0 = malignant, 1 = benign in this dataset)
    # since missing a malignant tumour is the costly mistake
    predictions = [predict(test_X[i], W, b) for i in range(len(test_X))]
    print(f"Precision (malignant): {sklearn.metrics.precision_score(test_Y, predictions, pos_label=0) * 100}%")
    print(f"Recall (malignant): {sklearn.metrics.recall_score(test_Y, predictions, pos_label=0) * 100}%")
    print()

    # Compare against scikit-learn's logistic regression on the same split
    sklearn_model = sklearn.linear_model.LogisticRegression().fit(train_X, train_Y)
    print(f"scikit-learn Accuracy: {sklearn_model.score(test_X, test_Y) * 100}%")
    print()

    # Test the vectorized model with the testing data
    vectorized_correct = sum(predict(test_X[i], W_vec, b_vec) == test_Y[i] for i in range(len(test_X)))
    print(f"Vectorized Accuracy: {vectorized_correct * 100.0 / len(test_X)}%")

    # Compare training times
    print(f"Loop training time: {loop_time} seconds")
    print(f"Vectorized training time: {vectorized_time} seconds")
    print(f"\nVectorized is {loop_time / vectorized_time}x faster")

    # Plot the loss curves of both models
    plt.plot(range(10, 10001, 10), costs, label="Loop (per example)")
    plt.plot(range(10, 10001, 10), costs_vec, label="Vectorized (all examples)")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (log loss)")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
