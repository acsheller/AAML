from kubernetes import client, config, watch
from kinfo import KubeInfo
import redis
import json
import random
import networkx as nx

import torch
from torch_geometric.data import Data
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import datetime
from cluster_env import ClusterEnvironment
from gnn_sched import GNNPolicyNetwork, ReplayBuffer
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')





class CustomScheduler:
    '''
    Implementation of RL agent using a GNN as the Deep protion of Deep Reinforcement Learning.
    '''

    def __init__(self,scheduler_name ="custom-scheduler",replay_buffer_size=100,learning_rate=1e-4,gamma=0.99,init_epsi=1.0, min_epsi=0.01,epsi_decay =0.9954,batch_size=16):
        self.scheduler_name = scheduler_name
        
        # Do K8s stuff
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.watcher = watch.Watch()
        self.kube_info = KubeInfo()
        self.env = ClusterEnvironment()

        #self.gnn_input = [] TODO Delete if its not useful
        self.BATCH_SIZE = batch_size
        self.replay_buffer_size = replay_buffer_size
        # Need to get the input size of the GNN.
        self.input_size = self.env.graph_to_torch_data(self.env.create_graph()).x.size(1)
        # Do the same for output size
        self.output_size = len(self.api.list_node().items) -1 # -1 because of 1 controller TODO Consider making this dynamic or an argument
        self.gnn_model = GNNPolicyNetwork(input_dim=self.input_size,hidden_dim=64,output_dim=self.output_size)

        logging.info("AGENT :: GNN Model Created")

        # Set up the optimizer
        self.optimizer = optim.Adam(self.gnn_model.parameters(), lr=learning_rate)
        
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        logging.info("AGENT :: Replay Buffer Created")

        # Set hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = init_epsi  # Starting value for epsilon in epsilon-greedy action selection -- Total Exploration
        self.min_epsi = min_epsi
        self.epsi_decay = epsi_decay
        self.action_list = []

        logging.info("AGENT :: scheduling Agent constructed.")


    def needs_scheduling(self, pod):
        '''
        Does this pod need scheduling?
        '''
        return (
            pod.status.phase == "Pending" and
            not pod.spec.node_name and
            pod.spec.scheduler_name == self.scheduler_name
        )


    def select_best_random_node(self,nodes,pod):

        '''
        will need this for the exploration, exploitation stuff.
        '''
        selected_node = random.choice(nodes)
        return selected_node

    def decay_epsilon(self):
        """
        Decays epsilon by the decay rate until it reaches the minimum value.
        """
        if self.epsilon > self.min_epsi:
            self.epsilon *= self.epsi_decay
            self.epsilon = max(self.epsilon, self.min_epsi)

    
    

    def select_action(self, state,pod):
        '''
        Policy networks selects action. 
        '''
        
        while True:
            available_nodes = [nd['name'] for nd in self.kube_info.get_valid_nodes()]
            rand = random.random()
            if rand > self.epsilon:
                with torch.no_grad():
                    # Use the model to choose the best action
                    action_index = self.gnn_model(state).max(1)[1].view(1, -1).item()
                    logging.info(f"AGENT :: Selected ACTION {action_index} selected")
                    self.action_list.append(action_index)
            else:
                # Choose a random action
                action_index = random.randrange(0,self.output_size)
                logging.info(f"AGENT :: Random ACTION {action_index} selected")
            # Map the action index to a node name using the environment's mapping

            node_name = self.env.kube_info.node_index_to_name_mapping[action_index]
            if node_name in available_nodes:
               return action_index,node_name
            else:
                logging.info(f"{node_name} is at capacity, Selecting Again")



    def add_to_buffer(self, state, action, reward, next_state,done):
        '''
        Add a new experience to the buffer. If the buffer is full, remove the oldest experience.
        '''
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop()
        self.replay_buffer.push(state, action, reward, next_state,done)


    def update_policy_network(self, experiences):
        # Unpack experiences
        states_batch = Batch.from_data_list([e[0] for e in experiences])
        actions = torch.tensor([e[1] for e in experiences], dtype=torch.int64)  # Assuming actions are indices
        rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float)
        next_states_batch = Batch.from_data_list([e[3] for e in experiences])
        dones = torch.tensor([e[4] for e in experiences], dtype=torch.float)
        
        # Compute predicted Q-values (Q_expected) from policy network
        Q_expected = self.gnn_model(states_batch).gather(1, actions.unsqueeze(-1))
        
        # Compute target Q-values (Q_targets) from next states
        # If using a target network, it would be involved here
        Q_targets_next = self.gnn_model(next_states_batch).detach().max(1)[0].unsqueeze(-1)
        Q_targets_next = Q_targets_next * (1-dones.unsqueeze(-1)) # Zero out the QValues for ones that are done
        
        Q_targets = rewards.unsqueeze(-1) + (self.gamma * Q_targets_next)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logging.info(f"AGENT :: Updated the Policy Network. Loss: {loss}")

        # Update target network, if necessary
        #if self.use_target_network and self.step_count % self.target_update_frequency == 0:
        #    self.target_network.load_state_dict(self.gnn_model.state_dict())


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
                        action,node_name = self.select_action(current_state,pod)
                        #logging.info(f"AGENT :: Action selected: assign pod to  {node_name}")
                        new_state, reward, done = self.env.step(pod,action)
                        logging.info(f"AGENT :: Reward for binding to {node_name} is {np.round(reward,3)}")
                        # Store the experience in the replay buffer
                        self.add_to_buffer(current_state, action, reward, new_state,done)
                        
                        # Deay the epsilon value
                        self.decay_epsilon()
                        logging.info(f"AGENT :: Decay Rate at {np.round(self.epsilon,3)}")
                        # Periodically update the policy network
                        if len(self.replay_buffer) > self.BATCH_SIZE:
                            experiences = self.replay_buffer.sample(self.BATCH_SIZE)
                            self.update_policy_network(experiences)
                            #logging.info("AGENT :: Policy Network Updated")
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
        filename = f"GNN_Model_{self.scheduler_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(self.gnn_model.state_dict(),)

    def load_model(self,f_name):
        model = GNNPolicyNetwork()
        model.load_state_dict(torch.load(f_name))
        model.eval()

if __name__ == "__main__":

    # Possible Values for the CustomerScheduler Constructor
    # scheduler_name ="custom-scheduler",replay_buffer_size=100,learning_rate=1e-4,gamma=0.99,init_epsi=1.0, min_epsi=0.01,epsi_decay =0.9954,batch_size=16
    scheduler = CustomScheduler(init_epsi=1.0,gamma=0.99,epsi_decay=0.99,replay_buffer_size=200,batch_size=32)
    scheduler.run()
