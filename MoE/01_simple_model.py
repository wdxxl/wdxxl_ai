import torch
import torch.nn as nn

# Define a simple neural network class
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        # Define a linear layer
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)  # Apply the linear transformation to the input

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create an instance of the model
    input_size = 3  # e.g., 3 features
    output_size = 1  # e.g., 1 output
    model = SimpleModel(input_size, output_size)
    
    # Create a sample input tensor (batch_size=2, input_size=3)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # Pass the input through the model
    output = model(x)
    
    # Print the result
    print("Input:", x)
    print("Output:", output)