import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# DQN
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # Conv1: 4 input channels (stacked frames), 32 output channels (filters), 8x8 kernel size, 4 stride
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # Conv2: 32 input channels (from previous layer), 64 output channels, 4x4 kernel size, 2 stride
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # Conv3: 64 input channels (from previous layer), 64 output channels, 3x3 kernel size, 1 stride
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the number of input features for the first fully connected layer:
        # After conv1: (84 - 8) / 4 + 1 = 20 -> output shape: (32, 20, 20)
        # After conv2: (20 - 4) / 2 + 1 = 9  -> output shape: (64, 9, 9)
        # After conv3: (9 - 3) / 1 + 1 = 7   -> output shape: (64, 7, 7)
        # Total features = 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(3136, 512)  # Fully connected layer: 3136 input features, 512 output features
        self.fc2 = nn.Linear(512, num_actions)  # Output layer: 512 input features, num_actions output features

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU activation after conv1
        x = F.relu(self.conv2(x))  # Apply ReLU activation after conv2
        x = F.relu(self.conv3(x))  # Apply ReLU activation after conv3
        x = x.view(x.size(0), -1)  # Flatten the output for the dense layers
        x = F.relu(self.fc1(x))    # Apply ReLU activation after fc1
        x = self.fc2(x)            # Output layer
        return x