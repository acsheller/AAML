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
        beforeState = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=False)
        if not dqn:
            node_name = self.apply_action(pod,action)
            # For a GNN
            afterState = self.kube_info.get_nodes_data(sort_by_cpu=True,include_controller=False)
            new_state = self.graph_to_torch_data(self.create_graph(afterState))
            reward = self.calc_reward(beforeState,afterState,action)
            #reward = self.calc_reward()
            done = self.done()
        else:
            node_name = self.apply_action(pod,action)
            # For an NN or DQN
            new_state = self.kube_info.get_node_data_single_input(include_controller=False)
            afterState = self.kube_info.get_nodes_data()
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


    def calculate_balance_reward_avg(self, node_values, max_value_per_node=0.80):
        values = list(node_values.values())
        num_nodes = len(values)
        
        # Calculate the average absolute difference
        average_difference = sum(abs(values[i] - values[j]) for i in range(num_nodes) for j in range(i + 1, num_nodes)) / (num_nodes * (num_nodes - 1) / 2)
        
        # Define a realistic maximum average difference (e.g., half the nodes are full, the other half are empty)
        max_average_difference = max_value_per_node / 2
        
        # Normalize the average difference
        normalized_difference = min(average_difference / max_average_difference, 1)
        
        # Invert the result so that a higher value means more balance
        balance_score = 1 - normalized_difference
        
        # Ensure the reward is non-negative
        reward = max(balance_score, 0)
        
        return reward


    def calculate_resource_balance_metric(self,node_resources):
        """
        Calculate a metric representing the balance of resource usage across nodes.
        :param node_resources: A list of resource usage values for each node.
        :return: A metric value representing resource balance.
        """
        # You can use standard deviation as a metric for balance
        return np.std(node_resources)

    def reward_for_balance(self,previous_state, current_state):
        """
        Calculate the reward based on improvement in resource balance.
        :param previous_state: Resource usage of each node in the previous state.
        :param current_state: Resource usage of each node in the current state.
        :return: Reward value.
        """
        prev_balance_metric = self.calculate_resource_balance_metric(list(previous_state.values()))
        curr_balance_metric = self.calculate_resource_balance_metric(list(current_state.values()))

        # Reward is based on the improvement in balance (reduction in std deviation)
        improvement = prev_balance_metric - curr_balance_metric
        reward = max(improvement, 0)  # Ensure non-negative reward
        return reward

    def calculate_improvement_reward(self, previous_state, current_state):
        # Calculate the standard deviation of resource usage for both states
        prev_std_dev = self.calculate_resource_std_dev(previous_state)
        curr_std_dev = self.calculate_resource_std_dev(current_state)

        # Reward is based on the improvement in standard deviation
        improvement = prev_std_dev - curr_std_dev
        reward = max(improvement, 0)  # Ensure that the reward is non-negative

        return reward

    def calculate_resource_std_dev(self, state):
        # Calculate the standard deviation of resource usage across nodes

        return np.std(list(state.values()))


    def calculate_balance_reward(self,node_values, max_value_per_node=0.80):
        '''
        Calculates the level of "balance" by taking pairwise different and summing
        Then the max value per 
        '''
        values = list(node_values.values())
        num_nodes = len(values)
        
        # Calculate all pairwise absolute differences
        pairwise_differences = sum(abs(values[i] - values[j]) for i in range(num_nodes) for j in range(i + 1, num_nodes))
        
        # The maximum possible sum of differences occurs when one node is at max value and all others are at zero
        # NOTE -- for future, the -1 is for the one controller- there may be more so this can be an argument or a feature.
        max_possible_difference = (num_nodes - 1) * max_value_per_node
        
        # Normalize the sum of differences
        normalized_difference = pairwise_differences / max_possible_difference
        
        # Invert the result so that a higher value means more balance <--- TODO Verify and Double check this!
        balance_score = 1 - normalized_difference
        
        # Scale the balance score to determine the final reward
        base_reward = 1.0  # Assume a base reward of 1.0 for maximum balance
        reward = balance_score * base_reward
        
        return reward


    def calc_scaled_reward(self,beforeState,action):
        #1. Get the nodes in sorted order
        #2. 
        step  = 2 / (len(beforeState)-1)
        values = [np.round(1 - i*step,4) for i in range(len(beforeState))]
        for index, ii in enumerate(beforeState):
            if ii['name'] == self.kube_info.node_index_to_name_mapping[action]:
                return values[index]



    def calc_reward(self,beforeState,afterState,action):
        '''

        '''
        # 1. Get the state of the cluster
        #node_info = self.kube_info.get_nodes_data()
        node_info_before = beforeState
        node_info_after = afterState

        reward = self.calc_scaled_reward(beforeState,action)
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


    def calc_reward2(self):

        # 1. Get the state of the cluster
        node_info = self.kube_info.get_nodes_data()
        #2. Get the CpuInfo for each node
        cpu_info = {}
        memory_info = {}
        pod_info = {}
        for node in node_info:
            if node['roles'] != 'control-plane':
                cpu_info[node['name']] = node['total_cpu_used']/node['cpu_capacity']
                memory_info[node['name']] = np.round(node['total_memory_used'] / node['memory_capacity'],3)
                pod_info[node['name']] = np.round(node['pod_count']/node['pod_limit'],3)
        # 3. Calculate balance score for CPU and Memory
        cpu_balance_score = np.round(self.calculate_balance_reward_avg(cpu_info),3)
        memory_balance_score = np.round(self.calculate_balance_reward(memory_info),3)
        pod_info_score = np.round(self.calculate_balance_reward(pod_info),3)

        #3. Now calculate reward -- note that defference functions can be tried here. 
        #reward = self.calculate_balance_reward_avg(cpu_info)
        reward = min(cpu_balance_score,memory_balance_score,pod_info_score)
        return reward



    def done(self):
        '''
            TODO Need to sort out the done method.  
            Determines if the eposide is done or not
        '''
        return False