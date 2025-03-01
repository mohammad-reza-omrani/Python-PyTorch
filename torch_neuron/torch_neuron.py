import torch

# Input tensor (e.g., features or values for a single neuron)
input_data = torch.tensor([1.0, 2.0])

# Weights for the neuron (also a tensor)
weights = torch.tensor([0.5, -0.5])

# Bias for the neuron (scalar, can also be represented as a tensor)
bias = torch.tensor(0.0)

# Weighted sum (dot product of input and weights + bias)
output = torch.matmul(input_data, weights) + bias

# Apply an activation function (e.g., ReLU or Sigmoid)
activated_output = torch.relu(output)  # You can also use torch.sigmoid(output)

print("Output of the artificial neuron:", activated_output)