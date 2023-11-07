from kubernetes import client, config, watch
from kinfo import KubeInfo
import redis
import json
import random
import networkx as nx

import torch
from torch_geometric.data import Data
import torch.optim as optim

from cluster_env import ClusterEnvironment
from gnn_sched import GNNPolicyNetwork, ReplayBuffer


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Action:
    def __init__(self,node,pod):
        self.node = node
        self.pod = pod




class CustomScheduler:
    '''
    Implementation of RL agent using a GNN as the Deep protion of Deep Reinforcement Learning.
    '''

    def __init__(self,scheduler_name ="custom-scheduler",replay_buffer_size=100,learning_rate=1e-4,gamma=0.99,epsilon=1.0):
        self.scheduler_name = scheduler_name
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.watcher = watch.Watch()
        self.kube_info = KubeInfo()
        self.env = ClusterEnvironment()
        self.gnn_input = []
        self.BATCH_SIZE = 16
        
        # Need to get the input size starting out so to have
        # something as the input size of the GNN.
        self.input_size = self.env.graph_to_torch_data(self.env.create_graph()).x.size(1)
        # Do the same for output size
        self.output_size = len(self.api.list_node().items) -1
        self.gnn_model = GNNPolicyNetwork(input_dim=self.input_size,hidden_dim=64,output_dim=self.output_size)

        logging.info("AGENT :: GNN Model Created")

        # Set up the optimizer
        self.optimizer = optim.Adam(self.gnn_model.parameters(), lr=learning_rate)
        
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        logging.info("AGENT :: Replay Buffer Created")


        # Set hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Starting value for epsilon in epsilon-greedy action selection

        logging.info("AGENT :: scheduling Agent constructed.")


    def needs_scheduling(self, pod):
        return (
            pod.status.phase == "Pending" and
            not pod.spec.node_name and
            pod.spec.scheduler_name == self.scheduler_name
        )


    def select_best_random_node(self,nodes,pod):

        '''
        #TODO Consider Deleting as this was used early to see how scheduling worked or leave it as part of explanation
        '''
        selected_node = random.choice(nodes)
        return selected_node

    def update_node_index_mapping(self):
        '''
        TODO Consider deleting as not used
        '''
        self.node_index_to_name_mapping = {}
        self.node_name_to_index_mapping = {}
        for index, node in enumerate(self.nodes):
            node_name = node.metadata.name
            self.node_index_to_name_mapping[index] = node_name
            self.node_name_to_index_mapping[node_name] = index

    def select_action(self, state,pod):
        '''
        GNN used here!!!
        '''
        sample = random.random()
        eps_threshold = self.epsilon

        # TODO eps_threshold is too big currently.
        if sample > eps_threshold:
            with torch.no_grad():
                # Use the model to choose the best action
                action_index = self.gnn_model(state).max(1)[1].view(1, -1).item()
        else:
            # Choose a random action
            action_index = random.randrange(1,self.output_size-1)

        # Map the action index to a node name using the environment's mapping
        node_name = self.env.kube_info.node_index_to_name_mapping[action_index]
        return Action(node_name,pod)



    def select_action2(self, pod):
        '''
        TODO Delete this as its likely not needed.
        '''
        nodes = self.api.list_node().items

        agents = []

        G = self.env.create_graph()
        self.gnn_input.append(self.env.graph_to_torch_data(G))

        return Action(best_node.metadata.name,pod)


    def add_to_buffer(self, state, action, reward, next_state):
        """
        Add a new experience to the buffer. If the buffer is full, remove the oldest experience.
        """
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state))


    def update_policy_network(self, experiences):
        # Unpack experiences
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.stack(next_states)
        
        # Compute predicted Q-values (Q_expected) from policy network
        Q_expected = self.gnn_model(states).gather(1, actions.unsqueeze(-1))
        
        # Compute target Q-values (Q_targets) from next states
        # If using a target network, it would be involved here
        Q_targets_next = self.gnn_model(next_states).detach().max(1)[0].unsqueeze(-1)
        Q_targets = rewards + (self.gamma * Q_targets_next)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network, if necessary
        if self.use_target_network and self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.gnn_model.state_dict())


    def run(self):
        '''
        
        It receives node selection to bind the pod to.
        It then takes the action and calulates the reward based on that action.
        
        '''
        while True:
            try:
                for event in self.watcher.stream(self.api.list_pod_for_all_namespaces):
                    pod = event['object']
                    if self.needs_scheduling(pod):
                        current_state= self.env.get_state()
                        action = self.select_action(current_state,pod)
                        logging.info(f"AGENT :: Action selected: assign pod to  {action.node}")
                        new_state, reward, done = self.env.step(action)

                        # Store the experience in the replay buffer
                        self.replay_buffer.push(current_state, action, reward, new_state,done)

                        # Periodically update the policy network
                        if len(self.replay_buffer) > self.BATCH_SIZE:
                            experiences = self.replay_buffer.sample(self.BATCH_SIZE)
                            self.update_policy_network(experiences)

                        # Reset the environment if the episode is done
                        if done:
                            self.env.reset()

            except client.exceptions.ApiException as e:
                if e.status == 410:
                    logging.warning("AGENT :: Watch timed out or resourceVersion too old. Restarting watch...")
                    self.watcher = watch.Watch() # TODO Do I need to restart the watcher?
                else:
                    logging.error(f"AGENT :: Unexpected API exception: {e}")
            except Exception as e:
                logging.error(f"AGENT :: Unexpected error: {e}")

if __name__ == "__main__":
    scheduler = CustomScheduler()
    scheduler.run()
