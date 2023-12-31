'''
Module cluster_env

ClusterEnvironment - A Python class representing a K8s cluster.



'''

from kubernetes import client, config, watch
from kinfo import KubeInfo
import redis
import json
import random
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClusterEnvironment:
    def __init__(self,namespace='default'):
        
        self.namespace = namespace
        #TODO Add constructor argument so that this can be in or out of cluster.
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.app_api= client.AppsV1Api()
        self.kube_info = KubeInfo()
        self.nodes = self.api.list_node().items
        self.watcher = watch.Watch()
        self.last_node_assigned = None
        self.assignment_count = {}

    def reset(self):
        # Reset the environment to an initial state
        # and return the initial observation
        # TODO Need to reset the cluster from here.  wipe it clean -- need to check this works but it looks like it will.
        try:
            logging.info("  ENV :: Resetting Cluster for some reason")
            #self.app_api.delete_collection_namespaced_deployment(namespace=self.namespace)
            #logging.info(f"  ENV :: All deployments in the '{self.namespace}' namespace have been deleted.")
        
            # Delete all pods in the specified namespace
            #self.api.delete_collection_namespaced_pod(namespace=self.namespace)
            #logging.info(f"  ENV :: All pods in the '{self.namespace}' namespace have been deleted.")
        
        except client.rest.ApiException as e:
            logging.error(f"  ENV :: Exception when calling Kubernetes API: {e}")

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        
        G = self.create_graph(self.kube_info.get_nodes_data())
        initial_state = G # self.graph_to_torch_data(G)
        return initial_state



    def step(self,pod, action,dqn=False):
        # Apply the action to the environment (schedule a pod)
        # Update the state (create a new graph)
        # Calculate the reward
        # Check if the episode is done
        #beforeState = self.kube_info.get_nodes_data()
        #
        if not dqn: # It is GNN
            beforeStateReward = self.kube_info.get_node_data_single_input(sort_by_cpu=True,include_controller=False)
            beforeState = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=True)            
            node_name = self.apply_action(pod,action)


            # For a GNN
            afterState = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=True)
            afterState2 = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=False)
            new_state = self.graph_to_torch_data(self.create_graph(afterState))
            
            reward = self.calc_reward(beforeState,afterState2,action)
            #reward = self.calc_reward()
            done = self.done()
        else: # It is a DQN
            beforeState = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=True)
            node_name = self.apply_action(pod,action)
            
            new_state = self.kube_info.get_node_data_single_inputCPU(include_controller=False)
            afterState = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=False)
            reward = self.calc_reward(beforeState,afterState,action)

            done = 0
        return new_state, reward, done


    def get_state(self,cpu_limit=0.8,mem_limit=0.8,pod_limit=0.8,dqn=False):
        '''
        Will return nodes that can be scheduled - whose capacity is less than 0.8
        '''
        if not dqn:
            G = self.create_graph(self.kube_info.get_nodes_data(),cpu_limit,mem_limit,pod_limit)
            data = self.graph_to_torch_data(G)
            return data
        else:
            return self.kube_info.get_node_data_single_inputCPU()

    def apply_action(self, pod,action):
        node_name = self.kube_info.node_index_to_name_mapping[action]
        self.bind_pod_to_node(pod,node_name)


    def pod_exists(self, pod_name):
        '''
        Double check if pod exists
        '''
        try:
            v1 = client.CoreV1Api()
            v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            return True
        except client.exceptions.ApiException as e:
            if e.status == 404:
                # Pod not found
                return False
            else:
                raise e

    def bind_pod_to_node(self, pod, node_name):
        if node_name is None:
            logging.error("  ENV :: Node name is None. Cannot bind pod.")
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

        try:
            if self.pod_exists(pod.metadata.name):
                self.api.create_namespaced_binding(namespace=pod.metadata.namespace, body=binding,_preload_content=False)
                self.create_graph(self.kube_info.get_nodes_data())
            else:
                logging.info("  ENV :: Pod did not exists so not creating it.")
        except Exception as e:
            logging.error(f"  ENV :: Exception when calling CoreV1Api->create_namespaced_binding: {e}")

    def convert_replay_to_graph(self,replay_data):
        node_data = self.kube_info.get_nodes_data()
        for index,replay in enumerate(replay_data):
            for node in node_data:
                if node['roles'] != 'control-plane':
                    if node['name'].endswith(str(index)):
                        node['total_cpu_used'] = int(np.round(replay,2)*node['cpu_capacity'])
                        break

        return self.create_graph(node_data)



    def create_graph(self,nodeData,cpu_limit=0.8,mem_limit=0.8,pod_limit=0.8):
        '''
        The graph will be the state of the system after adding the node
        '''
        G = nx.Graph()

        # Fetch all nodes data
        #nodes_data = self.kube_info.get_nodes_data()
        nodes_data = nodeData
        #self.kube_info.update_node_index_mapping()
        # Filter Nodes that are at 80% capacity.

        # Add nodes to the graph
        for node_data in nodes_data:
            node_name = node_data['name']
            G.add_node(node_name)
            # Add or update node attributes
            for key, value in node_data.items():
                #if key in ['roles','cpu_capacity','memory_capacity','total_cpu_used','total_memory_used','pod_count','pod_limit']:
                if key in ['roles','cpu_capacity']:
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


    def create_full_connected_worker_graph(self,nodeData,cpu_limit=0.8,mem_limit=0.8,pod_limit=0.8):
        '''
        The graph will be the state of the system after adding the node. It represents the state
        as the worker nodes which are fully connected. 
        '''
        G = nx.Graph()

        # Fetch all nodes data
        #nodes_data = self.kube_info.get_nodes_data()
        nodes_data = nodeData
        #self.kube_info.update_node_index_mapping()
        # Filter Nodes that are at 80% capacity.

        # Add nodes to the graph
        for node_data in nodes_data:
            node_name = node_data['name']
            G.add_node(node_name)
            # Add or update node attributes
            for key, value in node_data.items():
                if key in ['roles','cpu_capacity','memory_capacity','total_cpu_used','total_memory_used','pod_count','pod_limit']:
                    if value == 'agent':
                        value = 1
                    G.nodes[node_name][key] = value

        for node in G.nodes:
            for node2 in G.nodes:
                if node != node2:
                    G.add_edge(node,node2)

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

    def calc_scaled_reward4(self,State, action,neg_value=False):
        # Extract CPU usages
        cpu_usages = [np.round(node['total_cpu_used']/node['cpu_capacity'], 4) for node in State]

        # Calculate variance and scale it non-linearly to increase sensitivity
        variance = np.var(cpu_usages)
        scaled_variance = np.sqrt(variance)  # Example of non-linear scaling

        # Define maximum and minimum rewards
        max_reward = 1.0
        min_reward = -1.0

        # Normalize the scaled variance to be between 0 and 1
        normalized_variance = scaled_variance / (scaled_variance + 1)


        scale_on_freq = 1/np.sqrt(self.assignment_count[action])
        # Calculate reward (higher reward for lower variance)
        reward = max_reward - normalized_variance * (max_reward - min_reward)
        if neg_value:
            reward = -reward
            reward = max(reward, -1.0)  # Cap the negative reward
        else:
            reward *= scale_on_freq

        reward = np.round(reward, 4)
        return reward


    def calc_reward(self,beforeState,afterState,action):
        '''

        '''
        # 1. Get the state of the cluster
        #node_info = self.kube_info.get_nodes_data()
        if action not in self.assignment_count: 
            self.assignment_count[action] =0
        self.assignment_count[action] +=1
        node_info_before = beforeState
        node_info_after = afterState
        # Deter assigning to same node back to back.
        if self.last_node_assigned == None:
            self.last_node_assigned = self.kube_info.node_index_to_name_mapping[action]
        elif self.last_node_assigned == self.kube_info.node_index_to_name_mapping[action]:
            return self.calc_scaled_reward4(beforeState,action,neg_value=True)
            #return -1.5
        else:
            self.last_node_assigned = self.kube_info.node_index_to_name_mapping[action]
        reward = self.calc_scaled_reward4(beforeState,action)
        return reward



    def done(self):
        '''
            TODO Need to sort out the done method.  
            Determines if the eposide is done or not
        '''
        import os
        if os.path.exists('epoch_complete.txt'):
            return True
        return False