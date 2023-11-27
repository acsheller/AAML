import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GINConv, global_mean_pool
from torch_geometric.data import Batch
import random

from collections import deque

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    def __init__(self, num_inputs, num_outputs,num_hidden):
        super().__init__()
        # Increased depth: Adding additional layers
        self.fc1 = nn.Linear(num_inputs, int(num_hidden*2))  # First layer
        self.fc2 = nn.Linear(int(num_hidden*2), int(num_hidden*2))         # Second layer
        self.fc3 = nn.Linear(int(num_hidden*2), num_hidden)          # Third layer
        self.fc4 = nn.Linear(num_hidden, num_hidden)           # Fourth layer
        self.fc5 = nn.Linear(num_hidden, int(num_hidden//2))           # Fifth layer
        self.fc6 = nn.Linear(int(num_hidden//2), int(num_hidden//2))           # Sixth layer
        self.fc7 = nn.Linear(int(num_hidden//2), num_outputs)  # Output layer

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.001)  # Initialize biases

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)  # No activation function on output layer
        return x


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs,num_hidden):
        super().__init__()
        # Increased depth: Adding additional layers
        self.fc1 = nn.Linear(num_inputs, int(num_hidden*2))  # First layer
        self.fc2 = nn.Linear(int(num_hidden*2), int(num_hidden*2))         # Second layer
        self.fc3 = nn.Linear(int(num_hidden*2), num_hidden)          # Third layer
        self.fc4 = nn.Linear(num_hidden, num_hidden)           # Fourth layer
        self.fc5 = nn.Linear(num_hidden, int(num_hidden//2))           # Fifth layer
        self.fc6 = nn.Linear(int(num_hidden//2), int(num_hidden//2))           # Sixth layer
        self.fc7 = nn.Linear(int(num_hidden//2), num_outputs)  # Output layer

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.001)  # Initialize biases

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.softmax(self.fc7(x),dim=-1)
        return x


class Critic(nn.Module):
    def __init__(self, num_inputs, num_outputs,num_hidden):
        super().__init__()
        # Increased depth: Adding additional layers
        self.fc1 = nn.Linear(num_inputs, int(num_hidden*2))  # First layer
        self.fc2 = nn.Linear(int(num_hidden*2), int(num_hidden*2))         # Second layer
        self.fc3 = nn.Linear(int(num_hidden*2), num_hidden)          # Third layer
        self.fc4 = nn.Linear(num_hidden, num_hidden)           # Fourth layer
        self.fc5 = nn.Linear(num_hidden, int(num_hidden//2))           # Fifth layer
        self.fc6 = nn.Linear(int(num_hidden//2), int(num_hidden//2))           # Sixth layer
        self.fc7 = nn.Linear(int(num_hidden//2), 1)  # Output layer

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.001)  # Initialize biases

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x



import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

# Actor Network
class ActorGNN(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs,num_hidden):
        super(ActorGNN, self).__init__()
        # Network architecture remains similar
        self.conv1 = GCNConv(num_inputs, num_hidden)
        self.bn1 = torch.nn.BatchNorm1d(num_hidden)
        self.conv2 = GCNConv(num_hidden, num_hidden)
        self.bn2 = torch.nn.BatchNorm1d(num_hidden)
        self.fc1 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_outputs)

        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.001)


    def forward(self, data):
        # Forward pass remains largely unchanged

        if not isinstance(data, Batch):
            data = Batch.from_data_list([data])

        if hasattr(data, 'batch'):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index = data.x, data.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)


        #x, edge_index, batch = self.process_input(data)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)  # Output as a probability distribution
        return x


# Critic Network
class CriticGNN(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super(CriticGNN, self).__init__()
        # Similar architecture but the output is scalar
        self.conv1 = GCNConv(num_inputs, num_hidden)
        self.bn1 = torch.nn.BatchNorm1d(num_hidden)
        self.conv2 = GCNConv(num_hidden, num_hidden)
        self.bn2 = torch.nn.BatchNorm1d(num_hidden)
        self.fc1 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, 1)  # Scalar output

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.001)


    def forward(self, data):
        # Forward pass similar to the actor network but with a scalar output
        if not isinstance(data, Batch):
            data = Batch.from_data_list([data])

        if hasattr(data, 'batch'):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index = data.x, data.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        #x, edge_index, batch = self.process_input(data)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # <-- Scalar value
        return x




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

