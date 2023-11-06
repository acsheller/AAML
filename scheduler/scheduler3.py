from kubernetes import client, config, watch
from kinfo import KubeInfo
import redis
import json
import random
import networkx as nx

import torch
from torch_geometric.data import Data
import torch.optim as optim
from gnn_sched import GNNPolicyNetwork, ReplayBuffer


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Action:
    def __init__(self,node,pod):
        self.node = node
        self.pod = pod

class ClusterEnvironment:
    def __init__(self,namespace='default'):
        self.namespace = namespace
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.app_api= client.AppsV1Api()
        self.kube_info = KubeInfo()
        self.nodes = self.api.list_node().items
        self.watcher = watch.Watch()

    def reset(self):
        # Reset the environment to an initial state
        # and return the initial observation
        # TODO Need to reset the cluster from here.  wipe it clean -- need to check this.
        try:
            self.app_api.delete_collection_namespaced_deployment(namespace=self.namespace)
            logging.info(f"ENV :: All deployments in the '{namespace}' namespace have been deleted.")
        
            # Delete all pods in the specified namespace
            self.api.delete_collection_namespaced_pod(namespace=self.namespace)
            logging.info(f"ENV :: All pods in the '{namespace}' namespace have been deleted.")
        
        except client.rest.ApiException as e:
            logging.error(f"ENV :: Exception when calling Kubernetes API: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        
        G = self.create_graph()
        initial_state = self.graph_to_torch_data(G)
        return initial_state



    def step(self, action):
        # Apply the action to the environment (schedule a pod)
        # Update the state (create a new graph)
        # Calculate the reward
        # Check if the episode is done
        node_name = self.apply_action(action)
        new_state = self.graph_to_torch_data(self.create_graph())
        reward = self.calc_reward()
        done = self.done()
        return new_state, reward, done

    def get_state(self):
        # This method would return the current state of the cluster
        G = self.create_graph()
        data = self.graph_to_torch_data(G)
        return data

    def apply_action(self,action):
        self.bind_pod_to_node(action.pod,action.node)

    def bind_pod_to_node(self, pod, node_name):
        if node_name is None:
            logging.error("ENV :: Node name is None. Cannot bind pod.")
            return
        binding = {
            "apiVersion": "v1",
            "kind": "Binding",
            "metadata": {
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace
            },
            "target": {
                "apiVersion": "v1",
                "kind": "Node",
                "name": node_name
            }
        }
        logging.info(f"ENV :: Binding object: {binding}")
        try:
            self.api.create_namespaced_binding(namespace=pod.metadata.namespace, body=binding,_preload_content=False)
            self.create_graph()
        except Exception as e:
            logging.error(f"ENV :: Exception when calling CoreV1Api->create_namespaced_binding: {e}")
            

    def create_graph(self):
        '''
        The graph will be the state of the system after adding the node
        the graph will be created and then added to the dataframe for storage
        '''
        G = nx.Graph()

        # Add the controller as the central node
        #G.add_node("controller", type="controller")

        # Fetch all nodes data
        nodes_data = self.kube_info.get_nodes_data()
        self.kube_info.update_node_index_mapping()

        # Add nodes to the graph
        for node_data in nodes_data:
            node_name = node_data['name']
            G.add_node(node_name)
            # Add or update node attributes
            for key, value in node_data.items():
                if key in ['roles','cpu_capacity','memory_capacity','total_cpu_used','total_memory_used','pod_count','pod_limit']:
                    G.nodes[node_name][key] = value
        control_node = None
        for node in G.nodes:
            if G.nodes[node]['roles'] == 'control-plane':
                control_node = node
                G.nodes[node]['roles'] = 0

        for node in G.nodes:
            if G.nodes[node]['roles'] != 0:
                G.add_edge(control_node,node)
                G.nodes[node]['roles'] = 1

        return G

    def graph_to_torch_data(self,G):
        '''
        Transform a networkx graph into a pytorch data thing
        this is needed as input to a GNN
        '''
        # Convert to integers 
        node_to_int = {node: i for i, node in enumerate(G.nodes())}

        # Node features
        x = [list(G.nodes[node].values()) for node in G.nodes]
        x = torch.tensor(x, dtype=torch.float)
        # Edge indices
        edge_index = [[node_to_int[edge[0]], node_to_int[edge[1]]] for edge in G.edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # Create Data object
        data = Data(x=x, edge_index=edge_index)

        return data

    def calculate_balance_reward(self,node_values, max_value_per_node=0.80):
        values = list(node_values.values())
        num_nodes = len(values)
        
        # Calculate all pairwise absolute differences
        pairwise_differences = sum(abs(values[i] - values[j]) for i in range(num_nodes) for j in range(i + 1, num_nodes))
        
        # The maximum possible sum of differences occurs when one node is at max value and all others are at zero
        max_possible_difference = (num_nodes - 1) * max_value_per_node
        
        # Normalize the sum of differences
        normalized_difference = pairwise_differences / max_possible_difference
        
        # Invert the result so that a higher value means more balance
        balance_score = 1 - normalized_difference
        
        # Scale the balance score to determine the final reward
        base_reward = 1.0  # Assume a base reward of 1.0 for maximum balance
        reward = balance_score * base_reward
        
        return reward


    def calc_reward(self):
        '''
        # TODO Return a reward for CPU, Memory, and Pod distribution.  All similar percentages. But for now just return CPU.  :-) So Cool
        total_reward = (cpu_weight * cpu_reward + memory_weight * memory_reward + pod_count_weight * pod_count_reward) / (cpu_weight + memory_weight + pod_count_weight)
        total_reward = min(cpu_reward, memory_reward, pod_count_reward)
        total_reward = (cpu_weight * cpu_reward + memory_weight * memory_reward + pod_count_weight * pod_count_reward) / (cpu_weight + memory_weight + pod_count_weight)


        '''
        # 1. Get the state of the cluster
        node_info = self.kube_info.get_nodes_data()
        #2. Get the CpuInfo for each node
        cpu_info = {}
        for node in node_info:
            if node['roles'] != 'control-plane':
                cpu_info[node['name']] = node['total_cpu_used']/node['cpu_capacity']
        #3. Now calculate reward -- note that defference functions can be tried here. 
        reward = self.calculate_balance_reward(cpu_info)
            
        return reward


    def done(self):
        '''
            TODO Need to sort out the done method.  
            Determines if the eposide is done or not
        '''
        return False


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
        selected_node = random.choice(nodes)
        return selected_node

    def update_node_index_mapping(self):
        self.node_index_to_name_mapping = {}
        self.node_name_to_index_mapping = {}
        for index, node in enumerate(self.nodes):
            node_name = node.metadata.name
            self.node_index_to_name_mapping[index] = node_name
            self.node_name_to_index_mapping[node_name] = index

    def select_action(self, state,pod):
        sample = random.random()
        eps_threshold = self.epsilon
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
        
        It receives node selectio to bind the pod to.
        It then takes the action and calulates the reward based on that action.
        
        '''
        while True:
            try:
                for event in self.watcher.stream(self.api.list_pod_for_all_namespaces):
                    pod = event['object']
                    if self.needs_scheduling(pod):
                        current_state= self.env.get_state()
                        action = self.select_action(current_state,pod)
                        logging.info(f"AGENT :: Action selected: move pod to  {action.node}")
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
                else:
                    logging.error(f"AGENT :: Unexpected API exception: {e}")
            except Exception as e:
                logging.error(f"AGENT :: Unexpected error: {e}")

if __name__ == "__main__":
    scheduler = CustomScheduler()
    scheduler.run()
