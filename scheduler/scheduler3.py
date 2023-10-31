from kubernetes import client, config, watch
from kinfo import KubeInfo
import redis
import json
import random
import networkx as nx

import torch
from torch_geometric.data import Data


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class CustomScheduler:


    def __init__(self,scheduler_name ="custom-scheduler"):
        self.scheduler_name = scheduler_name
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.watcher = watch.Watch()
        self.kube_info = KubeInfo()

        self.gnn_input = []
        logging.info("scheduler constructed.")


    def needs_scheduling(self, pod):
        return (
            pod.status.phase == "Pending" and
            not pod.spec.node_name and
            pod.spec.scheduler_name == self.scheduler_name
        )


    def select_best_random_node(self,nodes,pod):
        selected_node = random.choice(nodes)
        return selected_node


    def schedule_pod(self, pod):

        nodes = self.api.list_node().items
        best_node = None
        gpu_agents = []
        agents = []

        G = self.create_graph()
        self.gnn_input.append(self.graph_to_torch_data(G))

        for node in nodes:
            # It's not a master -- orgnize the nodes into two types for now
            if "node-role.kubernetes.io/control-plane" not in node.metadata.labels:
                if node.metadata.labels.get("kubernetes.io/role") == "gpu-agent":
                    gpu_agents.append(node)
                elif node.metadata.labels.get("kubernetes.io/role") == "agent":
                    agents.append(node)


        # Check if the pod requests a GPU
        requests_gpu = False
        for container in pod.spec.containers:
            if container.resources and container.resources.limits:
                if "nvidia.com/gpu" in container.resources.limits:
                    requests_gpu = True
                    break

        for node in nodes:
            # If the pod requests a GPU, only consider nodes labeled with gpu=true
            if requests_gpu:
                best_node = self.select_best_node(gpu_agents,pod)
            # Note that more logic can be inserted here to schedule properly
            else:
                # It's an ordinary pod wanting memory and CPU
                best_node = self.select_best_random_node(agents,pod)
        return best_node.metadata.name


    def bind_pod_to_node(self, pod, node_name):
        if node_name is None:
            logging.error("Node name is None. Cannot bind pod.")

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

        logging.info(f"Binding object: {binding}")

        try:
            self.api.create_namespaced_binding(namespace=pod.metadata.namespace, body=binding,_preload_content=False)
            self.create_graph()
        except Exception as e:
            logging.error(f"Exception when calling CoreV1Api->create_namespaced_binding: {e}")


    def create_graph(self):
        '''
        The graph will be the state of the system after adding the node
        the graph will be created and then added to the dataframe for storage
        '''
        G = nx.Graph()

        # Add the controller as the central node
        #G.add_node("controller", type="controller")

        # Fetch all nodes data
        nodes_data = self.kube_info.get_nodes_data()

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
        # Edge indices
        edge_index = [[node_to_int[edge[0]], node_to_int[edge[1]]] for edge in G.edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create Data object
        data = Data(x=x, edge_index=edge_index)

        return data



    def run(self):
        while True:
            try:
                for event in self.watcher.stream(self.api.list_pod_for_all_namespaces):
                    pod = event['object']
                    if self.needs_scheduling(pod):
                        best_node = self.schedule_pod(pod)
                        logging.info(f"Best Node Selected {best_node}")
                        self.bind_pod_to_node(pod, best_node)

            except client.exceptions.ApiException as e:
                if e.status == 410:
                    logging.warning("Watch timed out or resourceVersion too old. Restarting watch...")
                else:
                    logging.error(f"Unexpected API exception: {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    scheduler = CustomScheduler()
    scheduler.run()
