
'''
Advanced Applied Machine Learning
EN.705.742
Anthony Sheller

This python module and source code is provided 


'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch

from torch.utils.tensorboard import SummaryWriter

from drl_models import ActorGNN, CriticGNN

from kubernetes import client, config, watch
from kinfo import KubeInfo
import redis
import json
import random
import networkx as nx
import os
import numpy as np
import argparse
import datetime
from datetime import datetime, timezone
import time
from cluster_env import ClusterEnvironment

### ---- Preparing the logging ---- ###
import logging
def def_logging(log_propogate=False):
    # Create a unique filename based on the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"logs/acgnn_log_{current_time}.log"

    # Create a named logger
    logger = logging.getLogger('MyACGNNLogger')
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
    # Set to true for logging to appear in log file as well as on screen.
    # Set to False to only appear in file.
    logger.propagate = log_propogate
    return logger

    ### ---- End of Logging Prep ---- ###


class ActorCriticGNN:
    '''
    an Actor Critic Deep Reinforcement Learning Agent that uses an "greph neural network.

    '''


    def __init__(self,namespace='default',scheduler_name ="custom-scheduler",hidden_layers=64,actor_learning_rate=1e-4, \
                critic_learning_rate=1e-4,gamma=0.99,progress_indication=True,tensorboard_name=None,optimizer=0, \
                log_propogate=False):
        '''
        Constructor 
        '''
        self.log_propogate = log_propogate
        self.logger = def_logging(log_propogate=log_propogate)
        self.scheduler_name = scheduler_name
        self.progress_indication = progress_indication
        self.namespace = namespace
        if tensorboard_name != None:
            self.tboard_name = tensorboard_name
        else:
            self.tboard_name = "acgnn_" + self.generate_funny_name()


        # Do K8s stuff
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.v1 = client.AppsV1Api()
        self.watcher = watch.Watch()
        self.kube_info = KubeInfo()

        # Create the environment
        self.env = ClusterEnvironment()

        # These are used for Tensorboard
        if self.tboard_name != None:
            self.writer = SummaryWriter(f'tlogs/{self.tboard_name}')
            self.logger.info(f"AGENT: ACGNN Created TensorBoard writer {self.tboard_name}")
        else:
            # Generate a name if one is not provided. 
            self.writer = SummaryWriter(f'tlogs/{self.generate_funny_name()}')
        self.train_iter = 0

        self.input_size = self.env.graph_to_torch_data(self.env.create_graph(self.kube_info.get_nodes_data())).x.size(1)
        # How many actions is the number of nodes in teh list.
        self.action_size = len(self.api.list_node().items) -1 # -1 because of 1 controller TODO Consider making this dynamic or an argument

        # Hardcoding num_inputs to 30 as that's the valeus being returned for a 10 node cluster or 10*3 which is 
        # cpu, memory, pod count
        # Create the Actor
        self.actor = ActorGNN(num_inputs=self.input_size, num_outputs=self.action_size,num_hidden=hidden_layers)

        # Set in train mode
        self.actor.train()

        # Create the Critic
        self.critic = CriticGNN(num_inputs=self.input_size, num_hidden=hidden_layers)

        # If cuda is available then use it.
        if torch.cuda.is_available():
            self.logger.info("AGENT :: GPU Available")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Put the models to the device either gpu or cpu
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        # Set up separate optimizers for Actor and Critic with different learning rates
        if optimizer == 1:
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=critic_learning_rate)
        elif optimizer == 2:
            self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_learning_rate)
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_learning_rate)
        elif optimizer == 3:
            self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=actor_learning_rate, momentum=0.9)
            self.critic_optimizer = optim.SGD(self.critic.parameters(), lr=critic_learning_rate, momentum=0.9)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # Set hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.action_list = []
        self.step_count = 0
        self.logger.info("AGENT :: Actor Critic GNN Agent constructed.")


    def select_action(self, state,pod):
        '''
        The Actor selects the action.  Because it is using multinomial, it is random at first.
        Each action as a probability of being selected. 
        '''
        
        while True:
            
            action_probs = self.actor(state.to(self.device))
            action_index = torch.multinomial(action_probs, 1).item()
            node_name = self.env.kube_info.node_index_to_name_mapping[action_index]

            #if node_name in available_nodes:
            return action_index, action_probs, "Actor"
            #else:
            #    choices = list(range(0,10))
            #    choices.remove(action_index)
            #    return random.choice(choices), action_probs, "Actor"
                

    def needs_scheduling(self, pod):
        '''
        Does this pod need scheduling? Just check if its in a pending state.
        '''
        return (
            pod.status.phase == "Pending" and
            not pod.spec.node_name and
            pod.spec.scheduler_name == self.scheduler_name and 
            self.env.pod_exists(pod.metadata.name)
        )


    def should_shutdown(self):
        '''
        So everything can shutdown about the same time
        '''
        self.logger.info("AGENT :: checking shutdown status")
        return os.path.exists("shutdown_signal.txt")

    def new_epoch(self):
        '''
        Checking for epoch reset of parameters
        '''
        if os.path.exists('epoch_complete.txt'):
            return True
        return False

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


    def run(self):
        '''
        
        runs until the shutdown signal is recieved.  This is just a file written to disk by the 
        deployment simulator. 

        Most of the work happens here. 
        
        '''
        c_sum_reward = 0
        deployed_pods = []
        deployment_counts = {}
        deployments_pods = {}

        if self.progress_indication:
            print(f"\rAC Model Ready",end='', flush=True)
        while not self.should_shutdown():
            
            try:
                # Listen for pods to be deployed to the cluster
                for event in self.watcher.stream(self.api.list_pod_for_all_namespaces,timeout_seconds=60):
                    
                    pod = event['object']
                    if self.needs_scheduling(pod):
                        try:

                            # Get the deployment Information
                            deployment_name = self.get_deployment_name(pod)
                            if deployment_name not in deployment_counts.keys():
                                deployment_counts[deployment_name] = self.get_num_pods_for_deployment(deployment_name)
                                deployments_pods[deployment_name] = []
                                self.logger.info(f"AGENT :: Processing Deployment {deployment_name} with {deployment_counts[deployment_name]} pods.")
                            # Process the pod so we can can count and add done = True later
                            if pod.metadata.name not in deployed_pods:
                                    deployed_pods.append(pod.metadata.name)
                                    deployments_pods[deployment_name].append(pod.metadata.name)

                            current_state= self.env.get_state()
                            # The actor selects the action in the select_action method
                            # Action probs are needed to calculate loss
                            action,action_probs, agent_type = self.select_action(current_state,pod,) 
                            new_state, reward, done = self.env.step(pod,action)
                            c_sum_reward += reward
                            self.action_list.append(action)

                            # Each Deployment is an "episode" 
                            if len(deployments_pods[deployment_name]) == deployment_counts[deployment_name]:
                                done = 1

                        except Exception as e:
                            self.logger.error(f"1. AGENT :: Unexpected error in section 1: {e}")                                   
                        
                        try:
                            # Critics Evaluation
                            value = self.critic(current_state.to(self.device))
                            next_value = self.critic(new_state.to(self.device))

                            # Calculate advantage
                            td_error = reward + self.gamma * next_value * (1 - int(done)) - value
                            advantage = td_error.detach()

                        except Exception as e:
                            self.logger.error(f"2. AGENT :: Unexpected error in section 2: {e}")       
                        try:
                            # Update Critic
                            critic_loss = td_error.pow(2)[0][0]
                            self.critic_optimizer.zero_grad()
                            critic_loss.backward()
                            self.critic_optimizer.step()
                        except Exception as e:
                            self.logger.error(f"3. AGENT :: Unexpected error in section 3: {e}")       

                        try:
                            # Update Actor using the advantage
                            actor_loss = -torch.log(action_probs[0,action]) * advantage
                            self.actor_optimizer.zero_grad()
                            actor_loss.backward()
                            self.actor_optimizer.step()

                        except Exception as e:
                            self.logger.error(f"4. AGENT :: Unexpected error in section 4: {e}")       

                        try:

                            if self.progress_indication and not self.log_propogate:
                                print(f"Reward for binding {pod.metadata.name} to node {self.env.kube_info.node_index_to_name_mapping[action]} is {np.round(reward,3)} ",end='', flush=True)
                            self.logger.info(f"AGENT :: Reward for binding {pod.metadata.name} to node {self.env.kube_info.node_index_to_name_mapping[action]} is {np.round(reward,3)} ")

                            # Save off some metrics for analysis
                            self.step_count += 1
                            self.writer.add_scalar('5. Actor Loss',actor_loss.item(),self.step_count)
                            self.writer.add_scalar('6. Critic Loss',critic_loss.item(),self.step_count)
                            self.writer.add_scalar('1. CSR',c_sum_reward,self.step_count)
                            self.writer.add_scalar('2. Advantage', advantage.item(),self.step_count)
                            if not self.step_count %10:
                                self.writer.add_histogram('4. Actions',torch.tensor(self.action_list),self.step_count)
                                temp_state = self.env.kube_info.get_nodes_data(sort_by_cpu=False,include_controller=False)
                                cpu_info = []
                                for node in temp_state:
                                    cpu_info.append(np.round(node['total_cpu_used']/node['cpu_capacity'],4))
                                self.writer.add_scalar('3. Cluster Variance',np.var(cpu_info),self.step_count)
                                self.logger.info(f"AGENT :: Cluster Variance at {np.round(np.var(cpu_info),4)}")
                            if sum(1 for pod in self.api.list_namespaced_pod(namespace = self.namespace).items if pod.status.phase == 'Pending') == 0:
                                if self.new_epoch():
                                    self.logger.info("AGENT :: Acknowledge Epoch Complete")
                                    os.remove('epoch_complete.txt')
                        except Exception as e:
                            self.logger.error(f"5. AGENT :: Unexpected error in section 5: {e}")       

            # Catch exceptions as things can happen. 
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

        self.logger.info("AGENT :: Saving AC Model for Reuse")
        # Save off actor critic models in case they want to be deployed.
        filename = f"models/Actor_{self.tboard_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        filename2 = f"models/Critic_{self.tboard_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(self.actor.state_dict(),filename)
        torch.save(self.critic.state_dict(),filename2)
        os.remove('shutdown_signal.txt')
        return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="runacgnn is an alias to actorcritic_gnn.py. It is used for running the Actor Critic GNN Scheduler with various configurations.")
    parser.add_argument('--hidden_layers',type=int, default=32,help='Number of Hidden Layers  (default: %(default)s)')
    parser.add_argument('--gamma',type=float, default=0.95,help='Discount Factor  (default: %(default)s)')
    parser.add_argument('--actor_learning_rate',type=float, default=0.001,help='Actor Learning Rate  (default: %(default)s)')
    parser.add_argument('--critic_learning_rate',type=float, default=0.001,help='Critic Learning Rate  (default: %(default)s)')
    parser.add_argument('--progress', action='store_true', help='Enable progress indication. Only when logs are not scrolling  (default: %(default)s)')
    parser.add_argument('--log_scrolling', action='store_true', help='Enable Log Scrolling to Screen. Disables progress Indication  (default: %(default)s)')
    args = parser.parse_args()

    scheduler = ActorCriticGNN(hidden_layers=args.hidden_layers,gamma=args.gamma,actor_learning_rate=args.actor_learning_rate, \
                              critic_learning_rate=args.critic_learning_rate, progress_indication=args.progress,log_propogate=args.log_scrolling)  
    scheduler.run()