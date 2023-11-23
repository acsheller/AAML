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
            
            new_state = self.kube_info.get_node_data_single_input(include_controller=False)
            afterState = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=False)
            reward = self.calc_reward(beforeState,afterState,action)

            done = self.done()

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
            return self.kube_info.get_node_data_single_input()

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




    def calc_scaled_reward(self,State,action):
        #1. Get the nodes in sorted order
        #2. 
        step  = 2 / (len(State)-1)
        values = [np.round(1 - i*step,4) for i in range(len(State))]
        for index, ii in enumerate(State):
            if ii['name'] == self.kube_info.node_index_to_name_mapping[action]:
                return values[index]

    def calc_scaled_reward2(self, State, action):
        # Extract CPU usages and get unique values
        cpu_usages = [np.round(node['total_cpu_used']/node['cpu_capacity'],4) for node in State]
        unique_cpu_usages = sorted(set(cpu_usages))

        # If all CPU usages are the same, return maximum reward
        if len(unique_cpu_usages) == 1:
            return 1

        # Calculate reward values based on unique CPU usages
        step = 1 / (len(unique_cpu_usages) - 1)
        values = {usage: np.round(1 - i * step,4) for i, usage in enumerate(unique_cpu_usages)}

        # Find the CPU usage of the node chosen by the action
        for index,ii in enumerate(State):
            if ii['name'] == self.kube_info.node_index_to_name_mapping[action]:
                return values[np.round(ii['total_cpu_used']/ii['cpu_capacity'],4)]

    def calc_scaled_reward3(self, State, action):
        # Extract CPU usages and get unique values
        cpu_usages = [np.round(node['total_cpu_used']/node['cpu_capacity'], 4) for node in State]
        unique_cpu_usages = sorted(set(cpu_usages))

        # If all CPU usages are the same, return maximum reward (which is 1 in this new scale)
        if len(unique_cpu_usages) == 1:
            return 1

        # Calculate the original values (scaled between 0 and 1)
        step = 1 / (len(unique_cpu_usages) - 1)
        values = {usage: np.round(1 - i * step, 4) for i, usage in enumerate(unique_cpu_usages)}

        # Transform values from [0, 1] to [-1, 1]
        values = {key: 2 * value - 1 for key, value in values.items()}


        # Find the CPU usage of the node chosen by the action
        for index, ii in enumerate(State):
            if ii['name'] == self.kube_info.node_index_to_name_mapping[action]:
                return values[np.round(ii['total_cpu_used']/ii['cpu_capacity'], 4)]

        # Fallback in case no matching node is found
        return -1  # You might want to handle this case based on your specific requirements


    def calc_reward(self,beforeState,afterState,action):
        '''

        '''
        # 1. Get the state of the cluster
        #node_info = self.kube_info.get_nodes_data()

        node_info_before = beforeState
        node_info_after = afterState

        reward = self.calc_scaled_reward3(beforeState,action)
        return reward
        #2. Get the CpuInfo for each node
        cpu_info_before = {}
        cpu_info_after = {}
        mem_info_before = {}
        mem_info_after = {}
        pod_info_before = {}
        pod_info_after = {}
        for index,node in enumerate(node_info_before):
            if node['roles'] != 'control-plane':
                cpu_info_before[node['name']] = 1 - np.round(node['total_cpu_used']/node['cpu_capacity'],4)
                node2 = node_info_after[index]
                cpu_info_after[node2['name']]= 1 - np.round(node2['total_cpu_used']/node2['cpu_capacity'],4)
                mem_info_before[node['name']] = 1 - np.round(node['total_memory_used'] / node['memory_capacity'],4)
                mem_info_after[node2['name']] = 1- np.round(node2['total_memory_used'] / node2['memory_capacity'],4)

                pod_info_before[node['name']] = 1 - np.round(node['pod_count'] / node['pod_limit'],4)
                pod_info_after[node2['name']] = 1- np.round(node2['pod_count'] / node2['pod_limit'],4)
        # 3. Calculate balance score for CPU and Memory
        #cpu_balance_score = np.round(self.calculate_balance_reward_avg(cpu_info),3)
        #memory_balance_score = np.round(self.calculate_balance_reward(memory_info),3)
        #pod_info_score = np.round(self.calculate_balance_reward(pod_info),3)
        #cpu_reward = self.calculate_improvement_reward(cpu_info_before,cpu_info_after)
        #mem_reward = self.calculate_improvement_reward(mem_info_before,mem_info_after)
        cpu_reward= np.round(self.reward_for_balance(cpu_info_before,cpu_info_after)*100,5)
        mem_reward= np.round(self.reward_for_balance(mem_info_before,mem_info_after)*100,5)
        pod_reward = np.round(self.reward_for_balance(pod_info_before,pod_info_after)*100,5)
        logging.info(f"  ENV :: Reward: CPU {cpu_reward} MEM {mem_reward} POD {pod_reward}")
        #3. Now calculate reward -- note that defference functions can be tried here. 
        #reward = self.calculate_balance_reward_avg(cpu_info)
        #reward = min(cpu_balance_score,memory_balance_score,pod_info_score)
        return cpu_reward



    def done(self):
        '''
            TODO Need to sort out the done method.  
            Determines if the eposide is done or not
        '''
        return False