from kubernetes import client, config, watch
from kinfo import KubeInfo
import redis
import json
import random
import networkx as nx
import os

import torch

import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import datetime
from datetime import datetime, timezone
import time
from cluster_env import ClusterEnvironment
from gnn_sched import ReplayBuffer,DQN
import numpy as np
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter

import logging
# Create a unique filename based on the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"logs/dqn_log_{current_time}.log"

# Create a named logger
logger = logging.getLogger('MyDQNLogger')
logger.setLevel(logging.INFO)

# Create file handler which logs even debug messages
fh = logging.FileHandler(filename)
fh.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)

# Prevent the log messages from being propagated to the Jupyter notebook
logger.propagate = False

class CustomSchedulerDQN:
    '''
    Implementation of RL agent using a NN as the Deep protion of Deep Reinforcement Learning.
    '''

    def __init__(self,scheduler_name ="custom-scheduler",hidden_layers=64,replay_buffer_size=100,learning_rate=1e-4,gamma=0.99,init_epsi=1.0, min_epsi=0.01,epsi_decay =0.9954,batch_size=25,target_update_frequency=50,progress_indication=True,tensorboard_name=None):
        
        self.scheduler_name = scheduler_name
        self.progress_indication = progress_indication

        if tensorboard_name != None:
            self.tboard_name = tensorboard_name


        # Do K8s stuff
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.watcher = watch.Watch()
        self.kube_info = KubeInfo()

        # Create the environment
        self.env = ClusterEnvironment()

        # These are used for Tensorboard
        if self.tboard_name:
            self.writer = SummaryWriter(f'tlogs/{self.tboard_name}')
            logger.info(f"AGENT: Created TensorBoard writer {self.tboard_name}")
        else:
            self.writer = SummaryWriter(f'tlogs/')
        self.train_iter =0

        # use a target network
        self.use_target_network = True
        self.target_update_frequency = target_update_frequency

        #BATCH_SIZE is used in the replay_buffer
        self.BATCH_SIZE = batch_size
        self.replay_buffer_size = replay_buffer_size

        # Need to get the input size for the network.
        #self.input_size = self.env.graph_to_torch_data(self.env.create_graph(self.kube_info.get_nodes_data())).x.size(1)
        # Do the same for output size
        self.output_size = len(self.api.list_node().items) -1 # -1 because of 1 controller TODO Consider making this dynamic or an argument
        #self.gnn_model = GNNPolicyNetwork2(input_dim=self.input_size,hidden_dim=64,output_dim=self.output_size)
        # Hardcoding num_inputs to 33 as that's the valeus being returned for a 10 node cluster or 11*3 which is 
        self.dqn = DQN(num_inputs=30, num_outputs=self.output_size,num_hidden=hidden_layers)

        self.target_network = DQN(num_inputs=30,num_outputs=self.output_size,num_hidden=hidden_layers)
        self.target_network.load_state_dict(self.dqn.state_dict())
        self.target_network.eval()
        self.increase_decay = True
        logger.info("AGENT :: DQN Models Created")

        if torch.cuda.is_available():
            logger.info("AGENT :: GPU Available")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dqn.to(self.device)
        self.target_network.to(self.device)
        
        # Set up the optimizer
        #elf.optimizer = optim.Adam(self.gnn_model.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        logger.info("AGENT :: Replay Buffer Created")

        # Set hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = init_epsi  # Starting value for epsilon in epsilon-greedy action selection -- Total Exploration
        self.min_epsi = min_epsi
        self.epsi_decay = epsi_decay
        self.action_list = []
        self.action_select_count = 0
        self.step_count = 0
        logger.info("AGENT :: scheduling Agent constructed.")


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
        if self.increase_decay and self.epsilon < 0.9:
            self.increase_decay = False
            logger.info("AGENT :: Setting epsilon decay rate to 0.995")
            self.epsi_decay=0.995
        if self.epsilon > self.min_epsi:
            self.epsilon *= self.epsi_decay
            self.epsilon = max(self.epsilon, self.min_epsi)


    def select_action(self, state,pod):
        '''
        Policy networks selects action. 
        '''
        agent_type = None
        while True:
            available_nodes = [nd['name'] for nd in self.kube_info.get_valid_nodes()]
            randval = random.random()
            if randval > self.epsilon:
                with torch.no_grad():
                    # Use the model to choose the best action
                    
                    action_index = self.dqn(torch.tensor([state], dtype=torch.float32).to(self.device)).max(1)[1].view(1, -1).item()
                    node_name = self.env.kube_info.node_index_to_name_mapping[action_index]
                    logger.info(f"  DQN :: DQN-Selected {np.round(randval,3)} ACTION: Assign {pod.metadata.name} to {node_name}")
                    agent_type='DQN'
            else:
                if random.randrange(1,2):
                    # Choose a random action
                    nodes = self.kube_info.get_nodes_data(include_controller=False)
                    selected_node = random.choice(nodes)
                    action_index = self.env.kube_info.node_name_to_index_mapping[selected_node['name']]
                    node_name = selected_node['name']
                    logger.info(f"AGENT :: Random {np.round(randval,3)} Selection: Assign {pod.metadata.name} to {node_name}")
                    agent_type='Random'
                else:
                    sorted_nodes = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=False)
                    lowest_cpu = np.round(sorted_nodes[0]['total_cpu_used'] / sorted_nodes[0]['cpu_capacity'],4)
                    lowest_nodes = [node for node in sorted_nodes if np.round(node['total_cpu_used'] / node['cpu_capacity'],4) == lowest_cpu]
                    selected_node = random.choice(lowest_nodes)
                    action_index = self.env.kube_info.node_name_to_index_mapping[selected_node['name']]
                    node_name = selected_node['name']
                    logger.info(f"AGENT :: Heuristic {np.round(randval,3)} Selection: Assign {pod.metadata.name} to {node_name}")
                    agent_type='Heuristic'
            if node_name in available_nodes:
               return action_index,agent_type
            else:
                self.action_select_count += 1
                
                time.sleep(2)
                self.action_select_count =0
                bad_reward = -1
                self.replay_buffer.push(state,action_index,bad_reward,state,False)
                logger.info(f"AGENT :: {node_name} is at capacity, Selecting Again")


    def add_to_buffer(self, state, action, reward, next_state,done):
        '''
        Add a new experience to the buffer. If the buffer is full, remove the oldest experience.
        '''
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop()
        self.replay_buffer.push(state, action, reward, next_state,done)


    def train_policy_network(self, experiences, epochs=3):
        for epoch in range(epochs):
            #logger.info(f"Starting epoch {epoch + 1}/{epochs}")

            # Unpack experiences
            # Use the appropriate format for states and next_states (graph or list)
            states_batch = torch.tensor([e[0] for e in experiences], dtype=torch.float32).to(self.device)
            actions = torch.tensor([e[1] for e in experiences], dtype=torch.int64).to(self.device)
            rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float).to(self.device)
            next_states_batch = torch.tensor([e[3] for e in experiences], dtype=torch.float32).to(self.device)
            dones = torch.tensor([e[4] for e in experiences], dtype=torch.float).to(self.device)

            # Compute predicted Q-values (Q_expected) from policy network
            Q_expected = self.dqn(states_batch).gather(1, actions.unsqueeze(-1))

            # Compute target Q-values (Q_targets) from next states using a target Network
            Q_targets_next = self.target_network(next_states_batch).detach().max(1)[0].unsqueeze(-1)
            Q_targets_next = Q_targets_next * (1 - dones.unsqueeze(-1))

            Q_targets = rewards.unsqueeze(-1) + (self.gamma * Q_targets_next)

            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)
            
            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {np.round(loss.cpu().detach().numpy().item(), 5)}")
            #logger.info(f"1 AGENT :: Updated the Policy Network. Loss: {np.round(loss.cpu().detach().numpy().item(),5)}")
            # Update step count and potentially update target network
            #self.step_count += 1
            if self.use_target_network and self.step_count % self.target_update_frequency == 0:
                logger.info("AGENT :: *** Updating the target Network")
                self.target_network.load_state_dict(self.dqn.state_dict())
        logger.info(f"AGENT :: Updated the Policy Network. Loss: {np.round(loss.cpu().detach().numpy().item(),5)}")
        self.train_iter += 1
        self.writer.add_scalar('Loss/Train',np.round(loss.cpu().detach().numpy().item()),self.train_iter)


    def update_policy_network(self, experiences):
        # Unpack experiences
        ### This one for a graph
        #states_batch = Batch.from_data_list([e[0] for e in experiences])
        ### This one for a list  -- for NN
        states_batch = torch.tensor([e[0] for e in experiences],dtype =torch.float32).to(self.device)
        actions = torch.tensor([e[1] for e in experiences], dtype=torch.int64).to(self.device)  # Assuming actions are indices
        rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float).to(self.device)
        # for GRaph
        # next_states_batch = Batch.from_data_list([e[3] for e in experiences])
        ## For NN
        next_states_batch = torch.tensor([e[3] for e in experiences],dtype =torch.float32).to(self.device)
        dones = torch.tensor([e[4] for e in experiences], dtype=torch.float).to(self.device)
        
        # Compute predicted Q-values (Q_expected) from policy network
        Q_expected = self.dqn(states_batch).gather(1, actions.unsqueeze(-1))
        # Compute target Q-values (Q_targets) from next states
        # Using a target Network
        Q_targets_next = self.target_network(next_states_batch).detach().max(1)[0].unsqueeze(-1)
        
        Q_targets_next = Q_targets_next * (1-dones.unsqueeze(-1)) # Zero out the QValues for ones that are done
        
        
        Q_targets = rewards.unsqueeze(-1) + (self.gamma * Q_targets_next)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.info(f"AGENT :: Updated the Policy Network. Loss: {np.round(loss.cpu().detach().numpy().item(),5)}")
        
        #self.step_count += 1
        # Update target network, if necessary
        if self.use_target_network and self.step_count % self.target_update_frequency == 0:
            logger.info("*** AGENT :: Updating the target Network")
            self.target_network.load_state_dict(self.dqn.state_dict())


    def should_shutdown(self):
        '''
        So everything can shutdown about the same time
        '''
        logger.info("AGENT :: checking shutdown status")
        return os.path.exists("shutdown_signal.txt")

    def should_reset(self):
        '''
        Checking for epoch reset of parameters
        '''
        if os.path.exists('epoch_complete.txt'):
            logger.info("AGENT :: Epoch Complete")
            os.remove('epoch_complete.txt')
            return True
        return False

    def run(self,epochs=1):
        '''
        
        It receives node selection to bind the pod to.
        It then takes the action and calulates the reward based on that action.
        
        '''
        c_sum_reward = 0
        if self.progress_indication:
            print(f"\rDQN Model Ready",end='', flush=True)
        while not self.should_shutdown():

            try:
                for event in self.watcher.stream(self.api.list_pod_for_all_namespaces,timeout_seconds=120):
                    pod = event['object']
                    if self.needs_scheduling(pod):
                        current_state= self.env.get_state(dqn=True)
                        action,agent_type = self.select_action(current_state,pod,)
                        #logger.info(f"AGENT :: Action selected: assign pod to  {node_name}")
                        new_state, reward, done = self.env.step(pod,action,dqn=True)
                        c_sum_reward += reward
                        logger.info(f"AGENT :: Reward for binding {pod.metadata.name} is {np.round(reward,6)}")
                        # Store the experience in the replay buffer
                        self.add_to_buffer(current_state, action, reward, new_state,done)
                        if self.progress_indication:
                            print(f"\rDecay Rate at {np.round(self.epsilon,4)}. {agent_type} Agent Receieved Reward {np.round(reward,3)}",end='', flush=True)
                        # Deay the epsilon value
                        self.decay_epsilon()
                        logger.info(f"AGENT :: Decay Rate at {np.round(self.epsilon,3)}")
                        # Periodically update the policy network

                        if len(self.replay_buffer) >= self.BATCH_SIZE and not self.step_count % (self.BATCH_SIZE // 2):
                            experiences = self.replay_buffer.sample(self.BATCH_SIZE)
                            self.train_policy_network(experiences,epochs=epochs)

                        self.step_count += 1
                        self.writer.add_scalar('CSR',c_sum_reward,self.step_count)
            except client.exceptions.ApiException as e:
                if e.status == 410:
                    logger.warning("AGENT :: Watch timed out")
                    if self.should_shutdown():
                        logger.info("AGENT :: Received shutdown signal")
                        break
                    else:
                        logger.info("AGENT :: Restarting Watch")
                        self.watcher = watch.Watch()
                else:
                    logger.error(f"AGENT :: Unexpected API exception: {e}")
            except Exception as e:
                logger.error(f"AGENT :: Unexpected error: {e}")

        logger.info("AGENT :: Saving DQN Model for Reuse")

        filename = f"DQN_Model_{self.tboard_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(self.dqn.state_dict(),filename)

    def load_model(self,f_name):
        '''
        Got a model to load then load it
        TODO needs more work
        '''
        model = DQN()
        model.load_state_dict(torch.load(f_name))
        model.eval()

if __name__ == "__main__":

    # Possible Values for the CustomerScheduler Constructor
    # scheduler_name ="custom-scheduler",replay_buffer_size=100,learning_rate=1e-4,gamma=0.99,init_epsi=1.0, min_epsi=0.01,epsi_decay =0.9954,batch_size=16
    #scheduler = CustomSchedulerDQN(init_epsi=1.0,gamma=0.9,learning_rate=1e-3,epsi_decay=0.9995,replay_buffer_size=500,batch_size=50,target_update_frequency=50)
    scheduler = CustomSchedulerDQN(hidden_layer_size=64,init_epsi=1.0,gamma=0.9,learning_rate=1e-4,epsi_decay=0.995,replay_buffer_size=100,batch_size=25,target_update_frequency=50)
    scheduler.run()