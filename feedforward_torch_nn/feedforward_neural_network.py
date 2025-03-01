import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer
    
    def forward(self, x):
        # Apply ReLU activation after the first layer
        x = torch.relu(self.fc1(x))
        # Output layer (no activation function is needed here for now)
        x = self.fc2(x)
        return x

# Step 2: Create an instance of the network
input_size = 3  # For example, 3 features as input
hidden_size = 5  # Number of neurons in the hidden layer
output_size = 2  # For binary classification

model = SimpleNN(input_size, hidden_size, output_size)

# Step 3: Define a Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Step 4: Dummy Input and Output Data (Example for Binary Classification)
inputs = torch.randn(10, input_size)  # 10 samples, each with 'input_size' features
labels = torch.randint(0, 2, (10,))  # 10 labels, binary values (0 or 1)

# Step 5: Train the Network (Example Training Loop)
epochs = 100

for epoch in range(epochs):
    # Forward pass
    outputs = model(inputs)
    
    # Compute the loss
    loss = loss_function(outputs, labels)
    
    # Zero the gradients before backward pass
    optimizer.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Optimize the weights
    optimizer.step()
    
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Step 6: Test the model
with torch.no_grad():  # We don't need to calculate gradients when testing
    test_inputs = torch.randn(5, input_size)  # Test with 5 new random samples
    predicted = model(test_inputs)
    predicted_class = torch.argmax(predicted, dim=1)  # Choose the class with the highest score
    print("Predicted classes:", predicted_class)
