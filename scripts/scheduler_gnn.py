from kubernetes import client, config, watch
from kinfo import KubeInfo
import json
import random
import networkx as nx
import os
import time
import pandas as pd
# For progress display
start_time = time.time()


import torch
from torch_geometric.data import Data
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import datetime
from datetime import datetime, timezone
import time
from cluster_env import ClusterEnvironment
from drl_models import GNNPolicyNetwork,GNNPolicyNetwork2, ReplayBuffer
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a unique filename based on the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"logs/gnn_log_{current_time}.log"

# Create a named logger
logger = logging.getLogger('MyGNNLogger')
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

INPUTS = 10

class CustomSchedulerGNN:
    '''
    Implementation of RL agent using a GNN as the Deep protion of Deep Reinforcement Learning.
    '''

    def __init__(self,scheduler_name ="custom-scheduler",replay_buffer_size=100,learning_rate=1e-4,gamma=0.99,init_epsi=1.0, min_epsi=0.01,epsi_decay =0.9954,batch_size=25,target_update_frequency=50,progress_indication=True,tensorboard_name=None):

        if tensorboard_name != None:
            self.tboard_name = tensorboard_name
        else:
            self.tboard_name = self.generate_funny_name()

        
        self.scheduler_name = scheduler_name
        self.progress_indication = progress_indication

        # Do K8s stuff
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.v1 = client.AppsV1Api()
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

        # Should we use a target network
        self.use_target_network = True
        self.target_update_frequency = target_update_frequency

        # BATCH_SIZE is used in the replay buffer
        self.BATCH_SIZE = batch_size
        self.replay_buffer_size = replay_buffer_size

        # Need to get the input size for the network.
        self.input_size = self.env.graph_to_torch_data(self.env.create_graph(self.kube_info.get_nodes_data())).x.size(1)
        #self.input_size = self.env.graph_to_torch_data(self.env.create_full_connected_worker_graph(self.kube_info.get_nodes_data(include_controller=False))).x.size(1)
        # Do the same for output size
        self.output_size = len(self.api.list_node().items) -1 # -1 because of 1 controller TODO Consider making this dynamic or an argument
        self.gnn = GNNPolicyNetwork2(input_dim=self.input_size,hidden_dim=64,output_dim=self.output_size)
        
        self.target_network = GNNPolicyNetwork2(input_dim=self.input_size,hidden_dim=64,output_dim=self.output_size)
        self.target_network.load_state_dict(self.gnn.state_dict())
        self.target_network.eval()
        self.increase_decay = False
        logger.info("AGENT :: GNN Model Created")

        if torch.cuda.is_available():
            logger.info("AGENT :: GPU Available")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn.to(self.device)
        self.target_network.to(self.device)

        # Set up the optimizer
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=learning_rate)
        
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        logger.info("AGENT :: Replay Buffer Created")

        # Set hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = init_epsi  # Starting value for epsilon in epsilon-greedy action selection -- Total Exploration
        self.min_epsi = min_epsi
        self.epsi_decay = epsi_decay
        self.action_list = []
        self.action_select_count = 0
        self.step_count = 1
        logger.info("AGENT :: scheduling Agent constructed.")


    def contains_non_alpha_or_dash(self,s):
        '''
        This is necessary because some of the names have funny characters in them. 
        '''
        for char in s:
            if not char.isalpha() and char != '-':
                return True
        return False


    def generate_funny_name(self):
        '''
        I cannot recall where I saw this done but I like it so crafted this version of it.
        '''
        import randomname
        while True:
            name = randomname.get_name().lower()
            if not self.contains_non_alpha_or_dash(name):
                return name

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
        if self.epsilon > self.min_epsi:
            self.epsilon *= self.epsi_decay
            self.epsilon = max(self.epsilon, self.min_epsi)


    def select_action(self, state,pod):
        '''
        Policy networks selects action. 
        '''
        
        while True:
            available_nodes = [nd['name'] for nd in self.kube_info.get_valid_nodes()]
            randval = random.random()
            if randval > self.epsilon:
                # use the network
                with torch.no_grad():
                    # Use the model to choose the best action
                    
                    action_index = self.gnn(state.to(self.device)).max(1)[1].view(1, -1).item()
                    node_name = self.env.kube_info.node_index_to_name_mapping[action_index]
                    logger.info(f"  GNN :: {np.round(randval,3)} assign {pod.metadata.name} to {node_name}")
                    agent_type="GNN"
            else:
                ## Will be random -- always 1 -- set equal to 0 or put a not in front of it to be heuristic
                if random.randrange(0,2): 
                    # Choose an action Randomly
                    nodes = self.kube_info.get_nodes_data(include_controller=False)
                    selected_node = random.choice(nodes)
                    action_index = self.env.kube_info.node_name_to_index_mapping[selected_node['name']]
                    node_name = selected_node['name']
                    logger.info(f' RAND :: {np.round(randval,3)} Selection: Assign {pod.metadata.name} to {node_name}')
                else:
                    # Heuristic Selection
                    sorted_nodes = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=False)
                    lowest_cpu = np.round(sorted_nodes[0]['total_cpu_used'] / sorted_nodes[0]['cpu_capacity'],4)
                    lowest_nodes = [node for node in sorted_nodes if np.round(node['total_cpu_used'] / node['cpu_capacity'],4) == lowest_cpu]
                    selected_node = random.choice(lowest_nodes)
                    action_index = self.env.kube_info.node_name_to_index_mapping[selected_node['name']]
                    node_name = selected_node['name']
                    logger.info(f" HEUR :: Heuristic {np.round(randval,3)} Selection: Assign {pod.metadata.name} to {node_name}")
                    agent_type="Heuristic"
            if node_name in available_nodes:
               return action_index,agent_type
            else:
                self.action_select_count += 1

                if self.action_select_count > 4:
                    self.env.reset()
                    time.sleep(2)
                self.replay_buffer.push(state,action_index,0,state,False)
                logger.info(f"agent :: {node_name} is at capacity, Selecting Again")


    def add_to_buffer(self, state, action, reward, next_state,done):
        '''
        Add a new experience to the buffer. If the buffer is full, remove the oldest experience.
        '''
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop()
        self.replay_buffer.push(state, action, reward, next_state,done)



    def train_policy_network(self, experiences, epochs=1):
        '''
        Trains the policy network.  Only needs one epoch because loss remains the same. 
        '''
        
        for epoch in range(epochs):
            

            # Unpack experiences
            # Use the appropriate format for states and next_states (graph or list)
            
            states_batch = Batch.from_data_list([e[0] for e in experiences]).to(self.device)
            actions = torch.tensor([int(e[1]) for e in experiences], dtype=torch.int64).to(self.device)
            rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float).to(self.device)
            next_states_batch = Batch.from_data_list([e[0].to(self.device) for e in experiences])
            dones = torch.tensor([e[4] for e in experiences], dtype=torch.float).to(self.device)

            # Compute predicted Q-values (Q_expected) from policy network
            Q_expected = self.gnn(states_batch).gather(1, actions.unsqueeze(-1))

            with torch.no_grad():
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

            if self.use_target_network and self.step_count % self.target_update_frequency == 0:
                logger.info("AGENT :: *** Updating the target Network")
                self.target_network.load_state_dict(self.dqn.state_dict())
        logger.info(f"AGENT :: Updated the Policy Network. Loss: {np.round(loss.cpu().detach().numpy().item(),5)}")
        self.train_iter += 1
        self.writer.add_scalar('Loss/Train',np.round(loss.cpu().detach().numpy().item(),7),self.train_iter)


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

    def get_num_pods_for_deployment(self,deployment_name):
        try:
            deployment = self.v1.read_namespaced_deployment(name=deployment_name,namespace = 'default')
            desired_replicas = deployment.spec.replicas
        except client.ApiException as e:
            logger.info("AGENT :: Exception when calling AppsV1Api-read_namespace_deployment from get_num_pods_for_deployment %s\n" % e)
            return False
        return desired_replicas

    def new_epoch(self):
        '''
        Checking for epoch reset of parameters
        '''
        if os.path.exists('epoch_complete.txt'):
            return True
        return False


    def get_deployment_name(self,pod):
        if pod.metadata.labels:
            label_key = 'app'
            deployment_name = pod.metadata.labels.get(label_key)
            if deployment_name:
                return deployment_name
        return None


    def run(self,epochs=1):
        '''
        
        It receives node selection to bind the pod to.
        It then takes the action and calulates the reward based on that action.
        
        '''

        deployed_pods = []
        deployment_counts = {}
        deployments_pods = {}

        c_sum_reward =0
        itercount = 0
        current_state = None
        first_two = [0.0,0.0]
        if self.progress_indication:
            print("\rGNN Ready",end='',flush=True)

        while not self.should_shutdown():
            # if sum(1 for pod in self.api.list_namespaced_pod(namespace = 'default').items if pod.status.phase == 'Pending') == 0:
            #     if self.new_epoch():
            #         logger.info("AGENT :: Acknowledge Epoch Complete")
            #         os.remove('epoch_complete.txt')
            #         deployed_pods = []
            #         deployment_counts = {}
            #         deployments_pods = {}
            #         time.sleep(10)

            try:
                for event in self.watcher.stream(self.api.list_pod_for_all_namespaces,timeout_seconds=120):
                    pod = event['object']
                    if self.needs_scheduling(pod):
                        try:
                           
                            # Get the deployment Information
                            deployment_name = self.get_deployment_name(pod)
                            if deployment_name not in deployment_counts.keys():
                                deployment_counts[deployment_name] = self.get_num_pods_for_deployment(deployment_name)
                                deployments_pods[deployment_name] = []
                                logger.info(f"AGENT :: Processing Deployment {deployment_name} with {deployment_counts[deployment_name]}")
                            # Process the pod so we can can count and add done = True later
                            if pod.metadata.name not in deployed_pods:
                                    deployed_pods.append(pod.metadata.name)
                                    deployments_pods[deployment_name].append(pod.metadata.name)


                            current_state= self.env.get_state()
                            action,agent_type = self.select_action(current_state,pod)
                            new_state, reward, done = self.env.step(pod,action)
                            c_sum_reward += reward
                            self.action_list.append(action)
                            logger.info(f"AGENT :: Reward {np.round(reward,6)} {self.step_count}")
                            if len(deployments_pods[deployment_name]) == deployment_counts[deployment_name]:
                                done = 1

                            # Store the experience in the replay buffer
                            self.add_to_buffer(current_state, action, reward, new_state,done)
                            if self.progress_indication:                   
                                print(f"\rDecay Rate at {np.round(self.epsilon,4)}. Reward Received {np.round(reward,3)}",end='', flush=True)

                        except Exception as e:
                            logger.error(f"1. AGENT :: Unexpected error: {e}")

                        try:
                            # Deay the epsilon value
                            self.decay_epsilon()
                            logger.info(f"AGENT :: Decay Rate at {np.round(self.epsilon,3)}")
                        
                            if len(self.replay_buffer) >= self.BATCH_SIZE and not self.step_count % self.BATCH_SIZE // 2:
                                experiences = self.replay_buffer.sample(self.BATCH_SIZE)
                                self.train_policy_network(experiences,epochs=epochs)                     
                                self.step_count += 1
                                self.writer.add_scalar('CSR',c_sum_reward,self.step_count)
                        except Exception as e:
                            logger.error(f"2. AGENT :: Unexpected error: {e}")

                        try:
                            if not self.step_count %10:
                                self.writer.add_histogram('actions',torch.tensor(self.action_list),self.step_count)
                                temp_state = self.env.kube_info.get_nodes_data(sort_by_cpu=False,include_controller=False)
                                cpu_info = []
                                for node in temp_state:
                                    cpu_info.append(np.round(node['total_cpu_used']/node['cpu_capacity'],4))
                                self.writer.add_scalar('Cluster Variance',np.var(cpu_info),self.step_count)
                            
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

                ## Will wrap back around resetting a few things but it will not exit.
                if sum(1 for pod in self.api.list_namespaced_pod(namespace = 'default').items if pod.status.phase == 'Pending') == 0:
                    if self.new_epoch():
                        logger.info("AGENT :: Acknowledge Epoch Complete")
                        os.remove('epoch_complete.txt')
                        deployed_pods = []
                        deployment_counts = {}
                        deployments_pods = {}
                        time.sleep(10)


            except Exception as e:
                logger.error(f"AGENT :: Unexpected error: {e}")

        logger.info("AGENT :: Saving GNN Model for Reuse")

        filename = f"GNN_Model_{self.tboard_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(self.gnn.state_dict(),filename)
        os.remove("shutdown_signal.txt")



    def load_model(self,f_name):
        '''
        Got a model to load then load it
        TODO needs more work
        '''
        model = ()
        model.load_state_dict(torch.load(f_name))
        model.eval()


    def populate_replay(self,df,epochs = 10):
        '''
        Populate the replay buffer
        '''

        len_data = len(df)
        for i in range(len_data):
            
            #state = [float(item) for item in df.iloc[i][['cpu_request', 'memory_request'] + [f'cpu_usage_node_{j}' for j in range(10)]].to_list()]
            state = [float(item) for item in df.iloc[i][[f'cpu_usage_node_{j}' for j in range(10)]].to_list()]
            action = np.float32(df.iloc[i]['action'])
            reward = np.float32(df.iloc[i]['reward'])
            done = df.iloc[i]['done']
            
            # Check if it's the last row
            if i == len_data - 1:
                # For the last row, there is no 'next state'
                next_state = state  # or a terminal state representation if you have one

            else:
                next_state = [float(item) for item in df.iloc[i+1][[f'cpu_usage_node_{j}' for j in range(10)]].to_list()]

            
            self.add_to_buffer(state, action, reward, next_state, done)



if __name__ == "__main__":

    file_names = ["data/sched_20231204_084827.csv","data/sched_20231204_095136.csv",'data/sched_20231204_105201.csv']
    dfs = []
    for data_file in file_names:
        dfs.append(pd.read_csv(data_file))
    main_df = pd.concat(dfs,ignore_index=True)

    agent = CustomSchedulerGNN(init_epsi=1.0,gamma=0.9,learning_rate=1e-3,epsi_decay=0.9985,replay_buffer_size=100,batch_size=20,target_update_frequency=40)
    agent.populate_replay(main_df)
    agent.run(epochs=1)

