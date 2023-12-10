from kubernetes import client, config, watch
from kinfo import KubeInfo
import json
import random
import networkx as nx
import os
import time
import pandas as pd
import argparse
import torch
from torch_geometric.data import Data
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import datetime
from datetime import datetime, timezone
import time
from cluster_env import ClusterEnvironment
from drl_models import GNNPolicyNetwork,GNNPolicyNetwork2,GNNPolicyNetwork3 as GNN, ReplayBuffer
import numpy as np

import concurrent.futures
from tqdm import tqdm

import concurrent.futures

from torch.utils.tensorboard import SummaryWriter

import logging

def def_logging(log_propogate=False):
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
    logger.propagate = log_propogate
    return logger

INPUTS = 10

class CustomSchedulerGNN:
    '''
    Implementation of RL agent using a GNN as the Deep  of Reinforcement Learning.
    '''

    def __init__(self,scheduler_name ="custom-scheduler",replay_buffer_size=100,hidden_layers=64,learning_rate=1e-4,gamma=0.99, \
                init_epsi=1.0, min_epsi=0.01,epsi_decay =0.9954,update_frequency=25,batch_size=25,target_update_frequency=50, \
                progress_indication=False,use_heuristic=False,tensorboard_name=None,log_propogate=False,num_inputs=10):

        self.agent_mode = 'train'
        self.use_heuristic= use_heuristic
        self.log_propogate = log_propogate
        self.logger = def_logging(log_propogate=log_propogate)
        self.num_inputs=num_inputs
        self.scheduler_name = scheduler_name
        self.progress_indication = progress_indication
        
        if tensorboard_name != None:
            self.tboard_name = tensorboard_name
        else:
            self.tboard_name = self.generate_funny_name()

        

        # For connecting to Kubernetes cluster
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.v1 = client.AppsV1Api()
        self.watcher = watch.Watch()
        self.kube_info = KubeInfo()

        # Create the environment
        self.env = ClusterEnvironment()

        # These are used for Tensorboard
        if self.tboard_name:
            self.writer = SummaryWriter(f'tlogs/gnn_{self.tboard_name}')
            self.logger.info(f"AGENT: Created TensorBoard writer {self.tboard_name}")
        else:
            self.writer = SummaryWriter(f'tlogs/')


        # Should we use a target network
        self.train_iter =0
        self.update_frequency = update_frequency
        self.use_target_network = True
        self.target_update_frequency = target_update_frequency

        # BATCH_SIZE is used in the replay buffer
        self.BATCH_SIZE = batch_size
        self.replay_buffer_size = replay_buffer_size

        # Need to get the input size for the network -- can be sought out with a graph.
        self.input_size = self.env.graph_to_torch_data(self.env.create_graph(self.kube_info.get_nodes_data())).x.size(1)
        # Do the same for output size
        self.output_size = len(self.api.list_node().items) -1 # -1 because of 1 controller TODO Consider making this dynamic or an argument
        self.gnn = GNN(input_dim=self.input_size,hidden_dim=hidden_layers,output_dim=self.output_size)
        self.target_network = GNN(input_dim=self.input_size,hidden_dim=hidden_layers,output_dim=self.output_size)
        self.target_network.load_state_dict(self.gnn.state_dict())
        self.target_network.eval()

        self.increase_decay = False
        self.logger.info("AGENT :: GNN Model Created")

        if torch.cuda.is_available():
            self.logger.info("AGENT :: GPU Available")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn.to(self.device)
        self.target_network.to(self.device)

        # Set up the optimizer
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=learning_rate)
        
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.logger.info("AGENT :: Replay Buffer Created")

        # Set hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = init_epsi  # Starting value for epsilon in epsilon-greedy action selection -- Total Exploration
        self.min_epsi = min_epsi
        self.epsi_decay = epsi_decay
        self.action_list = []
        self.action_select_count = 0
        self.step_count = 1
        self.logger.info("AGENT :: Agent Ready.")


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
        Used in early development kept for postarity
        '''
        selected_node = random.choice(nodes)
        return selected_node

    def decay_epsilon(self):
        """
        Decays epsilon by the decay rate until it reaches the minimum value.
        """
        if self.increase_decay and self.epsilon < 0.9:
            self.increase_decay = False
            #self.epsi_decay=0.995
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
            if self.agent_mode != 'train':
                self.epsilon = -1
            rand_start = 1
            if self.use_heuristic:
                rand_start = 0
            if randval > self.epsilon:
                with torch.no_grad():
                    # Use the model to choose the best action
                    action_index = self.gnn(state.to(self.device)).max(1)[1].view(1, -1).item()
                    node_name = self.env.kube_info.node_index_to_name_mapping[action_index]
                    self.logger.info(f"  GNN :: {np.round(randval,3)} assign {pod.metadata.name} to {node_name}")
                    agent_type="GNN"
            else:
                # Choose Random or Heuristic -- the latter keeps things on track
                if random.randrange(rand_start,2): 
                    # Choose an action Randomly
                    nodes = self.kube_info.get_nodes_data(include_controller=False)
                    selected_node = random.choice(nodes)
                    action_index = self.env.kube_info.node_name_to_index_mapping[selected_node['name']]
                    node_name = selected_node['name']
                    self.logger.info(f' RAND :: {np.round(randval,3)} Selection: Assign {pod.metadata.name} to {node_name}')
                else:
                    # Heuristic Selection
                    sorted_nodes = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=False)
                    lowest_cpu = np.round(sorted_nodes[0]['total_cpu_used'] / sorted_nodes[0]['cpu_capacity'],4)
                    lowest_nodes = [node for node in sorted_nodes if np.round(node['total_cpu_used'] / node['cpu_capacity'],4) == lowest_cpu]
                    selected_node = random.choice(lowest_nodes)
                    action_index = self.env.kube_info.node_name_to_index_mapping[selected_node['name']]
                    node_name = selected_node['name']
                    self.logger.info(f" HEUR :: Heuristic {np.round(randval,3)} Selection: Assign {pod.metadata.name} to {node_name}")
                    agent_type="Heuristic"
            return action_index,agent_type



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


            try:
                states_batch = Batch.from_data_list([e[0].to(self.device) for e in experiences]).to(self.device)
                actions = torch.tensor([int(e[1]) for e in experiences], dtype=torch.int64).to(self.device)
                rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float).to(self.device)
                next_states_batch = Batch.from_data_list([e[3].to(self.device) for e in experiences])
                dones = torch.tensor([e[4] for e in experiences], dtype=torch.float).to(self.device)
            except Exception as e:
                self.logger.error("1 AGENT :: ERROR in section 1 of train_policy_network".format(e))

            try:
                # Compute predicted Q-values (Q_expected) from policy network
                Q_expected = self.gnn(states_batch).gather(1, actions.unsqueeze(-1))

                with torch.no_grad():
                    # Compute target Q-values (Q_targets) from next states using a target Network
                    Q_targets_next = self.target_network(next_states_batch).detach().max(1)[0].unsqueeze(-1)
                    Q_targets_next = Q_targets_next * (1 - dones.unsqueeze(-1))
                    Q_targets = rewards.unsqueeze(-1) + (self.gamma * Q_targets_next)

                    # Compute loss
                loss = F.mse_loss(Q_expected, Q_targets)
            except Exception as e:
                self.logger.error("2 AGENT :: ERROR in section 2 of train_policy_network".format(e))
            try:
                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.logger.info(f"AGENT :: Updated the Policy Network. Loss: {np.round(loss.cpu().detach().numpy().item(),5)}")
                self.train_iter += 1
                self.writer.add_scalar('2. Loss/Train',np.round(loss.cpu().detach().numpy().item(),7),self.train_iter)
            except Exception as e:
                self.logger.error("3 AGENT :: ERROR in section 3 of train_policy_network".format(e))


    def should_shutdown(self):
        '''
        So everything can shutdown about the same time
        '''
        self.logger.info("AGENT :: checking shutdown status")
        return os.path.exists("shutdown_signal.txt")

    def should_reset(self):
        '''
        Checking for epoch reset of parameters
        '''
        if os.path.exists('epoch_complete.txt'):
            self.logger.info("AGENT :: Epoch Complete")
            os.remove('epoch_complete.txt')
            return True
        return False

    def get_num_pods_for_deployment(self,deployment_name):
        try:
            deployment = self.v1.read_namespaced_deployment(name=deployment_name,namespace = 'default')
            desired_replicas = deployment.spec.replicas
        except client.ApiException as e:
            self.logger.info("AGENT :: Exception when calling AppsV1Api-read_namespace_deployment from get_num_pods_for_deployment %s\n" % e)
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
                                self.logger.info(f"AGENT :: Processing Deployment {deployment_name} with {deployment_counts[deployment_name]}")
                            # Process the pod so we can can count and add done = True later
                            if pod.metadata.name not in deployed_pods:
                                    deployed_pods.append(pod.metadata.name)
                                    deployments_pods[deployment_name].append(pod.metadata.name)

                            # Get the current State and we are using GNN -- I started with it first
                            # Saving in same state as dqn because it is too costly to covert all of 
                            # data in replay buffer to graphs for state and next state
                            current_state= self.env.get_state()
                            
                            # Now get the action
                            action,agent_type = self.select_action(current_state,pod)
                            # Now take a step and get back new_state, reward and status
                            # dqn to True so replay buffer has all the same format
                            new_state, reward, done = self.env.step(pod,action)
                            # Keep track of rewards
                            c_sum_reward += reward

                            # Used for Histogram in tensorboard
                            self.action_list.append(action)
                            self.logger.info(f"AGENT :: Reward {np.round(reward,6)} {self.step_count}")
                            
                            # Each Deployment is an "episode"
                            if len(deployments_pods[deployment_name]) == deployment_counts[deployment_name]:
                                done = 1

                            # Store the experience in the replay buffer
                            self.add_to_buffer(current_state, action, reward, new_state,done)
                            if self.progress_indication and not self.log_propogate:                   
                                print(f"\rDecay Rate at {np.round(self.epsilon,4)}. Reward Received {np.round(reward,3)}",end='', flush=True)

                        except Exception as e:
                            self.logger.error(f"1. AGENT :: Unexpected error in section 1: {e}")

                        try:
                            self.decay_epsilon()
                            self.logger.info(f"AGENT :: Decay Rate at {np.round(self.epsilon,3)}")
                        
                            if self.step_count != 0 and not self.step_count % self.update_frequency:
                                experiences = self.replay_buffer.sample(self.BATCH_SIZE)
                                self.train_policy_network(experiences,epochs=epochs)      

                            if self.use_target_network and self.step_count % self.target_update_frequency == 0:
                                self.logger.info("AGENT :: *** Updating the target Network")
                                self.target_network.load_state_dict(self.gnn.state_dict())
            
                            self.step_count += 1
                            self.writer.add_scalar('1. CSR',c_sum_reward,self.step_count)
                        except Exception as e:
                            self.logger.error(f"2. AGENT :: Unexpected error in section 2: {e}")

                        try:
                            if not self.step_count %5:
                                self.writer.add_histogram('4. Actions',torch.tensor(self.action_list),self.step_count)
                                temp_state = self.env.kube_info.get_nodes_data(sort_by_cpu=False,include_controller=False)
                                cpu_info = []
                                for node in temp_state:
                                    cpu_info.append(np.round(node['total_cpu_used']/node['cpu_capacity'],4))
                                self.writer.add_scalar('3. Cluster Variance',np.var(cpu_info),self.step_count)
                                self.logger.info(f"AGENT :: Cluster Variance at {np.round(np.var(cpu_info),4)}")
                            
                        except client.exceptions.ApiException as e:
                            if e.status == 410:
                                self.logger.warning("AGENT :: Watch timed out")
                                if self.should_shutdown():
                                    self.logger.info("AGENT :: Received shutdown signal")
                                    break
                                else:
                                    self.logger.info("AGENT :: Restarting Watch")
                                    self.watcher = watch.Watch()
                            else:
                                self.logger.error(f"AGENT :: Unexpected API exception: {e}")
                        except Exception as e:
                            self.logger.error(f"AGENT :: Unexpected error: {e}")

                ## Will wrap back around resetting a few things but it will not exit.
                if sum(1 for pod in self.api.list_namespaced_pod(namespace = 'default').items if pod.status.phase == 'Pending') == 0:
                    if self.new_epoch():
                        self.logger.info("AGENT :: Acknowledge Epoch Complete")
                        os.remove('epoch_complete.txt')
                        deployed_pods = []
                        deployment_counts = {}
                        deployments_pods = {}
                        time.sleep(15)


            except Exception as e:
                self.logger.error(f"AGENT :: Unexpected error: {e}")

        self.logger.info("AGENT :: Saving GNN Model for Reuse")

        filename = f"models/GNN_Model_{self.tboard_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(self.gnn.state_dict(),filename)
        self.logger.info("AGENT :: Removing shutdown signal")
        os.remove("shutdown_signal.txt")
        return


    def load_model(self,f_name):
        '''
        Got a model to load then load it
        TODO needs more work
        '''
        model = ()
        model.load_state_dict(torch.load(f_name))
        model.eval()


    def populate_replay(self,df):
        '''
        Populate the replay buffer
        '''
        self.logger.info("AGENT :: Populating Replay Buffer")
        len_data = len(df)
        for i in range(len_data):
            if not i % int(0.1*len_data):
                self.logger.info(f"AGENT :: {np.round(np.round(i/len_data,2)*100,2)}% Complete ")
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


            state = self.env.graph_to_torch_data(self.env.convert_replay_to_graph(state))
            next_state = self.env.graph_to_torch_data(self.env.convert_replay_to_graph(next_state))
            self.add_to_buffer(state, action, reward, next_state, done)
        self.logger.info("AGENT :: Populating Replay Buffer Complete")


    def process_chunk(self,df_chunk, next_chunk_start):
        buffer_entries = []
        len_chunk = len(df_chunk)
        for i in range(len_chunk):
            state = [float(item) for item in df_chunk.iloc[i][[f'cpu_usage_node_{j}' for j in range(10)]].to_list()]
            action = np.float32(df_chunk.iloc[i]['action'])
            reward = np.float32(df_chunk.iloc[i]['reward'])
            done = df_chunk.iloc[i]['done']
            
            if i < len_chunk - 1:
                next_state = [float(item) for item in df_chunk.iloc[i+1][[f'cpu_usage_node_{j}' for j in range(10)]].to_list()]
            else:
                next_state = next_chunk_start
            state = self.env.graph_to_torch_data(self.env.convert_replay_to_graph(state))
            next_state = self.env.graph_to_torch_data(self.env.convert_replay_to_graph(next_state))
            buffer_entries.append((state, action, reward, next_state, done))
        return buffer_entries

    def populate_replay_parallel(self,df):
        self.logger.info("AGENT :: Populating Replay Buffer")

        chunk_size = 20 
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for i in range(0, len(df), chunk_size):
                df_chunk = df.iloc[i:i+chunk_size]
                if i + chunk_size < len(df):
                    next_chunk_start = [float(item) for item in df.iloc[i+chunk_size][[f'cpu_usage_node_{j}' for j in range(10)]].to_list()]
                else:
                    next_chunk_start = None  # Handle the last chunk separately
                futures.append(executor.submit(self.process_chunk, df_chunk, next_chunk_start))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Populating Replay Buffer"):
                buffer_entries = future.result()
                for entry in buffer_entries:
                    self.add_to_buffer(*entry)  # Add each entry to the buffer

        self.logger.info("AGENT :: Populating Replay Buffer Complete")


if __name__ == "__main__":

    file_names = ['data/sched_20231204_084827.csv','data/sched_20231204_095136.csv','data/sched_20231204_105201.csv']
    dfs = []
    for data_file in file_names:
        dfs.append(pd.read_csv(data_file))
    main_df = pd.concat(dfs,ignore_index=True)

    # parse arguments if passed in. use --help for help on this
    parser = argparse.ArgumentParser(description="rungnn is an alias to scheduler_gnn.py. It is used for running the GNN Scheduler with various configurations.")
    parser.add_argument('--hidden_layers',type=int, default=32,help='Number of Hidden Layers  (default: %(default)s)')
    parser.add_argument('--init_epsi',type=float, default=1.0,help='Initial Epsilon Starting Value  (default: %(default)s)')
    parser.add_argument('--epsi_decay',type=float, default=0.995,help='Epsilon Decay Rate  (default: %(default)s)')
    parser.add_argument('--gamma',type=float, default=0.99,help='Discount Factor  (default: %(default)s)')
    parser.add_argument('--learning_rate',type=float, default=0.001,help='Learning Rate  (default: %(default)s)')
    parser.add_argument('--replay_buffer_size',type=int, default=1000,help='Length of the Replay Buffer  (default: %(default)s)')
    parser.add_argument('--update_frequency',type=int, default=20,help='Network Update Frequency  (default: %(default)s)')
    parser.add_argument('--target_update_frequency',type=int, default=40,help='Target Network Update Frequency  (default: %(default)s)')
    parser.add_argument('--batch_size',type=int, default=200,help='Batch Size of replay Buffer sample to pass during training  (default: %(default)s)')
    parser.add_argument('--progress', action='store_true', help='Enable progress indication. Only when logs are not scrolling  (default: %(default)s)')
    parser.add_argument('--log_scrolling', action='store_true', help='Enable Log Scrolling to Screen. Disables progress Indication  (default: %(default)s)')
    parser.add_argument('--use_heuristic', action='store_true', help='Enable use of Heuristic half of random time (default: %(default)s)')
    args = parser.parse_args()

    agent = CustomSchedulerGNN(hidden_layers=args.hidden_layers,init_epsi=args.init_epsi,gamma=args.gamma, \
                               learning_rate=args.learning_rate,epsi_decay=args.epsi_decay, \
                               replay_buffer_size=args.replay_buffer_size,update_frequency=args.update_frequency, \
                               target_update_frequency=args.target_update_frequency,batch_size=args.batch_size, \
                               progress_indication=args.progress,log_propogate=args.log_scrolling,use_heuristic=args.use_heuristic)

    agent.populate_replay(main_df)
    agent.run(epochs=1)
