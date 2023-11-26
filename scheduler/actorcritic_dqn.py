
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

from drl_models import Actor, Critic

from kubernetes import client, config, watch
from kinfo import KubeInfo
import redis
import json
import random
import networkx as nx
import os
import numpy as np
import datetime
from datetime import datetime, timezone
import time
from cluster_env import ClusterEnvironment

### ---- Preparing the logging ---- ###
import logging
# Create a unique filename based on the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"logs/ac_log_{current_time}.log"

# Create a named logger
logger = logging.getLogger('MyACLogger')
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
logger.propagate = True

### ---- End of Logging Prep ---- ###


class ActorCriticDQN:
    '''
    an Actor Critic Deep Reinforcement Learning Agent that uses an "ordinairy neural network.

    '''


    def __init__(self,scheduler_name ="custom-scheduler",hidden_layers=64,actor_learning_rate=1e-4,critic_learning_rate=1e-4,gamma=0.99,progress_indication=True,tensorboard_name=None):
        '''
        Constructor 
        '''
        
        self.scheduler_name = scheduler_name
        self.progress_indication = progress_indication

        self.tboard_name = tensorboard_name


        # Do K8s stuff
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.watcher = watch.Watch()
        self.kube_info = KubeInfo()

        # Create the environment
        self.env = ClusterEnvironment()

        # These are used for Tensorboard
        if self.tboard_name != None:
            self.writer = SummaryWriter(f'tlogs/{self.tboard_name}')
            logger.info(f"AGENT: Created TensorBoard writer {self.tboard_name}")
        else:
            # Generate a name if one is not provided. 
            self.writer = SummaryWriter(f'tlogs/{self.generate_funny_name()}')
        self.train_iter = 0

        # How many actions is the number of nodes in teh list.
        self.action_size = len(self.api.list_node().items) -1 # -1 because of 1 controller TODO Consider making this dynamic or an argument

        # Hardcoding num_inputs to 30 as that's the valeus being returned for a 10 node cluster or 10*3 which is 
        # cpu, memory, pod count
        # Create the Actor
        self.actor = Actor(num_inputs=30, num_outputs=self.action_size,num_hidden=hidden_layers)

        # Set in train mode
        self.actor.train()

        # Create the Critic
        self.critic = Critic(num_inputs=30, num_outputs=self.action_size,num_hidden=hidden_layers)

        # If cuda is available then use it.
        if torch.cuda.is_available():
            logger.info("AGENT :: GPU Available")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Put the models to the device either gpu or cpu
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        # Set up separate optimizers for Actor and Critic with different learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # Set hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.action_list = []
        self.step_count = 0
        logger.info("AGENT :: scheduling Agent constructed.")


    def select_action(self, state,pod):
        '''
        The Actor selects the action.  Because it is using multinomial, it is random at first.
        Each action as a probability of being selected. 
        '''
        agent_type = None
        while True:
            available_nodes = [nd['name'] for nd in self.kube_info.get_valid_nodes()]
            state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
            #state_tensor.requires_grad = True  

            action_probs = self.actor(state_tensor)
            action_index = torch.multinomial(action_probs, 1).item()
            node_name = self.env.kube_info.node_index_to_name_mapping[action_index]

            if node_name in available_nodes:
                return action_index, action_probs, "Actor"

    def needs_scheduling(self, pod):
        '''
        Does this pod need scheduling? Just check if its in a pending state.
        '''
        return (
            pod.status.phase == "Pending" and
            not pod.spec.node_name and
            pod.spec.scheduler_name == self.scheduler_name
        )


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


    def run(self,epochs=1):
        '''
        
        runs until the shutdown signal is recieved.  This is just a file written to disk by the 
        deployment simulator. 

        Most of the work happens here. 
        
        '''
        c_sum_reward = 0
        
        if self.progress_indication:
            print(f"\rAC Model Ready",end='', flush=True)
        while not self.should_shutdown():

            try:
                # Listen for pods to be deployed to the cluster
                for event in self.watcher.stream(self.api.list_pod_for_all_namespaces,timeout_seconds=120):
                    pod = event['object']
                    if self.needs_scheduling(pod):

                        current_state= self.env.get_state(dqn=True)
                        # The actor selects the action in the select_action method
                        # Action probs are needed to calculate loss
                        action,action_probs, agent_type = self.select_action(current_state,pod,)
                        new_state, reward, done = self.env.step(pod,action,dqn=True)
                        c_sum_reward += reward
                        self.action_list.append(action)
                        
                        # Critics Evaluation
                        value = self.critic(torch.tensor([current_state], dtype=torch.float32).to(self.device))
                        next_value = self.critic(torch.tensor([new_state], dtype=torch.float32).to(self.device))                        

                        # Calculate advantage
                        td_error = reward + self.gamma * next_value * (1 - int(done)) - value
                        advantage = td_error.detach()
                        #print("action_probs requires_grad:", action_probs.requires_grad)
                        #print("advantage requires_grad:", advantage.requires_grad)

                        # Update Critic
                        critic_loss = td_error.pow(2)
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        self.critic_optimizer.step()


                        # Update Actor using the advantage
                        actor_loss = -torch.log(action_probs[0,action]) * advantage
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        logger.info(f"AGENT :: Reward for binding {pod.metadata.name} to node {self.env.kube_info.node_index_to_name_mapping[action]} is {np.round(reward,3)} ")

                        # Save off some metrics for analysis
                        self.step_count += 1
                        self.writer.add_scalar('Actor Loss',actor_loss.item(),self.step_count)
                        self.writer.add_scalar('Critic Loss',critic_loss.item(),self.step_count)
                        self.writer.add_scalar('CSR',c_sum_reward,self.step_count)
                        self.writer.add_scalar('Advantage', advantage.item(),self.step_count)
                        if not self.step_count %10:
                            self.writer.add_histogram('actions',torch.tensor(self.action_list),self.step_count)
            # Catch exceptions as things can happen. 
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

        logger.info("AGENT :: Saving AC Model for Reuse")
        # Save off actor critic models in case they want to be deployed.
        filename = f"AC_ModelActor_{self.tboard_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        filename2 = f"AC_ModelCritic_{self.tboard_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(self.actor.state_dict(),filename)
        torch.save(self.critic.state_dict(),filename2)


if __name__ == "__main__":
    # This is how it can be run if a cluster is up and running.  this can be run external of the cluster as the docker compose
    # provides a cluster to work with.  
    scheduler = ActorCriticDQN(hidden_layers=64,gamma=0.9,actor_learning_rate=1e-4,critic_learning_rate=1e-4,progress_indication=True,tensorboard_name=None)    
    scheduler.run()