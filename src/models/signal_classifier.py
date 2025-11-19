import torch
import torch.nn as nn
from typing import Tuple

class SignalClassifier(nn.Module):
    # Architecture: Input (1000) -> FC (256) -> ReLU -> Dropout(0.2) ->
    # FC (128) -> ReLU -> Dropout(0.2) -> FC (3) -> Output
    
    def __init__(
        self,
        input_size: int = 1000,
        hidden_sizes: Tuple[int, ...] = (256, 128),
        num_classes: int = 3,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_sizes[1], num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the network to return logits of shape
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc_out(x)
        return x
    
    def get_num_parameters(self) -> int:
        # Total number of trainable parameters
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = SignalClassifier()
    print(f"Model parameters: {model.get_num_parameters():,}")