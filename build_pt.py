from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from model_pt import LinearRegressionModel
from preprocessing import load_dataset

import os


def main():
    dataset = load_dataset()

    # Divide data into features (X) and target (y)
    X = dataset.drop('MPG', axis=1)
    y = dataset['MPG']

    print(X.head())
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Check the shapes
    X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # Define input dimension (number of features) and output dimension (1)
    input_dim = X_train.shape[1]
    output_dim = 1

    # Create model
    model = LinearRegressionModel(input_dim, output_dim)

    # Initialize a scaler
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform both the training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled data to PyTorch tensors
    X_train_tensor = Variable(torch.tensor(X_train_scaled, dtype=torch.float))
    y_train_tensor = Variable(torch.tensor(y_train.values, dtype=torch.float)).view(-1, 1)
    X_test_tensor = Variable(torch.tensor(X_test_scaled, dtype=torch.float))
    y_test_tensor = Variable(torch.tensor(y_test.values, dtype=torch.float)).view(-1, 1)

    # Create a new model (to reset parameters)
    model = LinearRegressionModel(input_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Define number of epochs
    epochs = 100

    # Placeholder for loss values
    losses = []

    # Training process
    for epoch in range(epochs):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(X_train_tensor)

        # Compute loss
        loss = criterion(y_pred, y_train_tensor)
        losses.append(loss.item())

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print final loss
    print(losses[-1])


    # Compute predictions on the test set
    y_pred_test = model(X_test_tensor)

    # Compute MSE on the test set
    test_loss = criterion(y_pred_test, y_test_tensor)

    # Print test loss
    test_loss.item()
    print(test_loss.item())

    # Save the model
    path = './models'
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f'The {path} directory is created!')

    torch.save(model.state_dict(), './models/model.pt')

    # Save the scaler to disk
    joblib.dump(scaler, './models/scaler.pkl')


if __name__ == '__main__':
    main()
