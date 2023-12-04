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
from drl_models import ReplayBuffer,DQN
import numpy as np
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter

import logging
# Create a unique filename based on the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"logs/scheduler_log_{current_time}.log"

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
logger.propagate = True


class SchedulerRecord:
    '''
    Record data from the simulator so it can be played back
    '''
    def __init__(self,scheduler_name='default_scheduler'):

        self.scheduler_name = scheduler_name

        # Do K8s stuff
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.v1 = client.AppsV1Api()
        self.watcher = watch.Watch()
        self.kube_info = KubeInfo()

        self.env = ClusterEnvironment()
        

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

        # Function to get resource requests
    def get_resource_requests(self, pod):
        cpu_request = memory_request = 0
        for container in pod.spec.containers:
            resources = container.resources.requests if container.resources and container.resources.requests else {}
            cpu_request += int(resources.get('cpu', '0').replace('m', '').replace('Gi','')) if 'cpu' in resources else 0
            memory_request += int(resources.get('memory', '0').replace('Mi', '').replace('Gi','')) if 'memory' in resources else 0
        return cpu_request, memory_request

    def record_scheduling_action(self, pod):
        cpu_request, memory_request = self.get_resource_requests(pod)
        action = self.env.kube_info.node_name_to_index_mapping[pod.spec.node_name]
        cpu_usages = self.env.kube_info.get_node_data_single_inputCPU()

        # Ensure that cpu_usages is an array of ten floats as described
        assert len(cpu_usages) == 10 and all(isinstance(x, float) for x in cpu_usages)
        nodes_state = self.env.kube_info.get_nodes_data(include_controller=False)
        if action not in self.env.assignment_count:
            self.env.assignment_count[action] =0
        self.env.assignment_count[action] +=1
        new_row = {
            'pod_name': pod.metadata.name,
            'action': action,
            'cpu_request': cpu_request,
            'memory_request': memory_request,
            'reward': self.env.calc_scaled_reward4(nodes_state,action),
            'done': 0,
            'cpu_usages': cpu_usages  # Adding the entire array of CPU usages
        }
        return new_row


    def get_num_pods_for_deployment(self,deployment_name):
        try:
            deployment = self.v1.read_namespaced_deployment(name=deployment_name,namespace = 'default')
            desired_replicas = deployment.spec.replicas
        except client.ApiException as e:
            logger.info("Exception when calling AppsV1Api-read_namespace_deployment from get_num_pods_for_deployment %s\n" % e)
            return False
        return desired_replicas

        # Compare the current count with the desired replica count
        current_count = deployment_count.get(deployment_name, 0)
        return current_count >= desired_replicas

    def get_deployment_name(self,pod):
        if pod.metadata.labels:
            label_key = 'app'
            deployment_name = pod.metadata.labels.get(label_key)
            if deployment_name:
                return deployment_name
        return None




    def run(self):
        import pandas as pd
        deployed_pods = []
        deployment_counts = {}
        deployments_pods = {}

        columns = ['pod_name','action','reward', 'done', 'cpu_request', 'memory_request'] + [f'cpu_usage_node_{i}' for i in range(10)]
        c_sum_reward = 0
        df = pd.DataFrame(columns=columns)
        while not self.should_shutdown():
            #pod_count = sum(1 for pod in self.api.list_namespaced_pod(namespace = 'default').items)
            try:
                for event in self.watcher.stream(self.api.list_pod_for_all_namespaces,timeout_seconds=120):
                    pod = event['object']
                    #logger.info(f"pod {pod.metadata.name}")
                    if pod.status.phase == "Running" and pod.spec.node_name and pod.metadata.name not in deployed_pods:
                        deployment_name = self.get_deployment_name(pod)
                        if deployment_name not in deployment_counts.keys():
                            deployment_counts[deployment_name] = self.get_num_pods_for_deployment(deployment_name)
                            deployments_pods[deployment_name] = []
                            logger.info(f"Processing Deployment {deployment_name} with {deployment_counts[deployment_name]}")

                        if pod.metadata.name not in deployed_pods:
                            deployed_pods.append(pod.metadata.name)
                            deployments_pods[deployment_name].append(pod.metadata.name)
                            data = self.record_scheduling_action(pod)
                            c_sum_reward += data['reward']
                            if len(deployments_pods[deployment_name]) == deployment_counts[deployment_name]:
                                data['done'] = 1
                            row_data = [data['pod_name'], data['action'], data['reward'], data['done'], data['cpu_request'], data['memory_request']] + data['cpu_usages']
                            
                            df.loc[len(df)] = row_data
                    
                            logger.info(f'{data["pod_name"]}, {np.round(c_sum_reward,4)} cumulative sum of rewards df_size {len(df)}')
            except Exception as e:
                print(e)
        if not df.empty:
            df.iloc[-1, df.columns.get_loc('done')]= 1
        os.remove('shutdown_signal.txt')
        logger.info("Logger :: Saving data to csv file")
        file_path = f"data/sched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv" 
        df.to_csv(file_path, index=False)
        return

if __name__ == "__main__":
    # This is how it can be run if a cluster is up and running.  this can be run external of the cluster as the docker compose
    # provides a cluster to work with.  
    scheduler = SchedulerRecord() 
    scheduler.run()