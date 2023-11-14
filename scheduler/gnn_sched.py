import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GINConv, global_mean_pool
from torch_geometric.data import Batch
import random

from collections import deque

class GNNPolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNPolicyNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        #x, edge_index, batch = data.x, data.edge_index, data.batch


        if not isinstance(data, Batch):
            # If it's a single graph, wrap it in a Batch for compatibility
            data = Batch.from_data_list([data])

        if hasattr(data, 'batch'):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            # Handle the case where data.batch doesn't exist (e.g., single instance)
            x, edge_index = data.x, data.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)


        # Graph Convolutional Layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Global Pooling Layer
        x = global_mean_pool(x, batch)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class GNNPolicyNetwork2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNPolicyNetwork2, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        if not isinstance(data, Batch):
            data = Batch.from_data_list([data])

        if hasattr(data, 'batch'):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index = data.x, data.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw scores (logits)
        return x
    

class DQN(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_outputs)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for layers with ReLU activation
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.01)  # Initialize biases, if needed

    def forward(self, x):        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    '''
    the purpose of the replay buffer is to break the 
    temporal correlations between consecutive experiences, 
    allowing the model to learn from a more diverse set of experiences.
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Append the new experience, deque will automatically discard the oldest if over capacity
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def pop(self):
        # Remove and return the oldest item
        return self.buffer.popleft()

    def __len__(self):
        return len(self.buffer)


# Example usage:
#model = GNNPolicyNetwork(input_dim=your_input_dim, hidden_dim=64, output_dim=10)
