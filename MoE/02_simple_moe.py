import torch # type: ignore
import torch.nn as nn # type: ignore

# Define the Mixture of Experts model
class SimpleMoE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_experts=2):
        super(SimpleMoE, self).__init__()
        
        # Define the experts (simple linear layers)
        self.experts = nn.ModuleList([
            nn.Linear(input_size, output_size) for _ in range(num_experts)
        ])
        
        # Define the gating network (input -> num_experts)
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)  # Outputs weights summing to 1
        )
    
    def forward(self, x):
        # Get gating weights (batch_size, num_experts)
        gate_weights = self.gate(x)
        
        # Get outputs from each expert (batch_size, output_size) for each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # (batch_size, output_size, num_experts)
        
        # Weighted sum of expert outputs
        # (batch_size, output_size, 1) = (batch_size, output_size, num_experts) * (batch_size, 1, num_experts)
        output = torch.bmm(expert_outputs, gate_weights.unsqueeze(2)).squeeze(2)
        
        return output

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    input_size = 4   # e.g., 4 input features
    hidden_size = 8  # Hidden size for the gate
    output_size = 2  # e.g., 2 output classes
    num_experts = 2  # Number of experts
    
    # Create the MoE model
    model = SimpleMoE(input_size, hidden_size, output_size, num_experts)
    
    # Sample input (batch_size=3, input_size=4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]])
    
    # Forward pass
    output = model(x)
    
    # Print results
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output:", output)
    
 
# Input shape: torch.Size([3, 4])
# Output shape: torch.Size([3, 2])
# Output: tensor([[ 2.6415,  0.7172],
#         [ 7.2707,  1.1315],
#         [11.9621,  1.1594]], grad_fn=<SqueezeBackward1>)
