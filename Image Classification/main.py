# main.py

import torch
import torch.optim as optim
import torch.nn as nn
from model import CNNModel
from data_loader import get_data_loaders
from utils import train_model, evaluate_model, save_model

def main():
    # Get data loaders for train and test
    trainloader, testloader = get_data_loaders(batch_size=64)

    # Initialize model
    model = CNNModel()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, trainloader, criterion, optimizer, num_epochs=10)

    # Evaluate the model
    accuracy = evaluate_model(model, testloader)
    print(f"Accuracy on the test set: {accuracy:.2f}%")

    # Save the trained model
    save_model(model, "saved_models/cifar10_cnn.pth")

if __name__ == '__main__':
    main()
