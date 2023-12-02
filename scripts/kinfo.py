from kubernetes import client, config
from datetime import datetime, timezone
import json
import numpy as np
class KubeInfo:

    POD_LIMIT = 110

    def __init__(self):
        config.load_kube_config()
        self.v1 = client.CoreV1Api()
        self.nodes = []
        self.update_node_index_mapping()

    def get_pod_count_per_node(self, node_name):
        """Get the number of pods running on a specific node."""
        pods = self.v1.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node_name}").items
        return len(pods)

    def calculate_age(self, creation_timestamp):
        if not isinstance(creation_timestamp, datetime):
            creation_date = datetime.strptime(creation_timestamp, '%Y-%m-%dT%H:%M:%SZ')
        else:
            creation_date = creation_timestamp

        creation_date = creation_date.replace(tzinfo=timezone.utc)
        age_delta = datetime.now(timezone.utc) - creation_date
        age_str = "{}d".format(age_delta.days)
        return age_str

    def convert_memory_to_gigabytes(self,memory_str):
        """Convert memory string to Gigabytes (GB)"""
        if memory_str.endswith('Ki'):
            return int(memory_str.rstrip('Ki')) / (1024 * 1024)
        elif memory_str.endswith('Mi'):
            return np.round(int(memory_str.rstrip('Mi')) / 1024,3)
        elif memory_str.endswith('Gi'):
            return int(memory_str.rstrip('Gi'))
        else:
            return int(memory_str)  # Assuming it's already in Gi

    def get_node_resources(self, name):
        v1 = client.CoreV1Api()
        node = v1.read_node(name)
        capacity = node.status.capacity
        allocatable = node.status.allocatable
        pod_limit = node.status.allocatable.get('pods',self.POD_LIMIT)
        total_cpu_used = 0
        total_memory_used = 0
        total_gpu_used = 0
        pods = v1.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={name}").items
        for pod in pods:
            for container in pod.spec.containers:
                resources = container.resources.requests if container.resources and container.resources.requests else {}
                if not resources.get('cpu', '0m').endswith('m'):
                    total_cpu_used += int(resources.get('cpu', '0m').rstrip('m')) *1000
                else:
                    total_cpu_used += int(resources.get('cpu', '0m').rstrip('m'))
                total_memory_used += self.convert_memory_to_gigabytes(resources.get('memory', '0Gi'))
                total_gpu_used += int(resources.get('nvidia.com/gpu', '0'))
        return {
            'cpu_capacity': int(capacity['cpu']) * 1000,
            'memory_capacity': capacity['memory'],
            'gpu_capacity': capacity.get('nvidia.com/gpu', '0'),
            'total_cpu_used': total_cpu_used,
            'total_memory_used': total_memory_used,
            'total_gpu_used': total_gpu_used
        }


    def get_node_status(self, conditions):
        # If there's only one condition, return its status
        if len(conditions) == 1:
            return "Ready" if conditions[0].status == "True" else "Not Ready"
        
        # If there are multiple conditions, search for the 'Ready' condition
        ready_status = next((condition.status for condition in conditions if condition.type == 'Ready'), 'Unknown')
        return "Ready" if ready_status == "True" else "Not Ready"

    def update_node_index_mapping(self):
        self.node_index_to_name_mapping = {}
        self.node_name_to_index_mapping = {}
        index =0
        nodes = self.v1.list_node().items
        for node in self.get_nodes_data():
            if node['roles'] == 'agent':
                self.node_index_to_name_mapping[index] = node['name']
                self.node_name_to_index_mapping[node['name']] = index
                index += 1


    def display_nodes_info(self):
        POD_LIMIT = 110
        nodes = self.v1.list_node().items
        print("{:<18} {:<8} {:<14} {:<7} {:<10} {:>7} {:>16} {:>8} {:>6}".format(
            "NAME", "STATUS", "ROLES", "AGE", "VERSION", "CPU", "MEMORY", "GPU","PODS"
        ))
        for node in nodes:
            name = node.metadata.name
            pod_count = self.get_pod_count_per_node(name)
            status = self.get_node_status(node.status.conditions)
            roles = node.metadata.labels['kubernetes.io/role']
            #roles = ', '.join(role_labels) if role_labels else 'None'
            age = self.calculate_age(node.metadata.creation_timestamp)
            version = node.status.node_info.kubelet_version
            if 'control-plane' in roles:
                resources = self.get_node_resources(name)
                print("{:<18} {:<8} {:<14} {:<7} {:<10} {:>5}/{:<5}  {:>7}/{:<5}    {}/{}   {}/{}".format(
                    name, status, roles, age, version,
                    resources['total_cpu_used'], resources['cpu_capacity'],
                    np.round(resources['total_memory_used'],4), int(resources['memory_capacity'].rstrip("Gi").rstrip("Ki"))//1000**2,
                    resources['total_gpu_used'], resources['gpu_capacity'],
                    pod_count,POD_LIMIT))
                continue
            resources = self.get_node_resources(name)
            print("{:<18} {:<8} {:<14} {:<7} {:<10} {:>5}/{:<5}  {:>7}/{:<5}    {}/{}   {}/{}".format(
                name, status, roles, age, version,
                resources['total_cpu_used'], resources['cpu_capacity'],
                resources['total_memory_used'], int(resources['memory_capacity'].rstrip("Gi")),
                resources['total_gpu_used'], resources['gpu_capacity'],
                pod_count,POD_LIMIT
            ))

    def get_valid_nodes(self,cpu_limit=0.8,mem_limit=0.8,pod_limit=0.8):
        """
        Returns a list of node names that are under the specified capacity threshold.
        :param capacity_threshold: The maximum capacity percentage that a node can be at to be considered valid.
        :return: A list of valid node names.
        """
        valid_nodes = []
        node_info = self.get_nodes_data() 

        for node in node_info:
            if node['roles'] != 'control-plane':
                cpu_capacity_used = node['total_cpu_used'] / node['cpu_capacity']
                memory_capacity_used = node['total_memory_used'] / node['memory_capacity']
                pod_capacity_used = node['pod_count'] / node['pod_limit']

                # Check if all capacities are below the threshold
                if cpu_capacity_used < cpu_limit and memory_capacity_used < mem_limit and pod_capacity_used < pod_limit:
                    valid_nodes.append(node)
            

        return valid_nodes


    def get_nodes_data(self,sort_by_cpu=False,include_controller=True):
        
        POD_LIMIT=110
        nodes_data = []
        nodes = self.v1.list_node().items

        for node in nodes:

            name = node.metadata.name

            pod_count = self.get_pod_count_per_node(name)
            status = self.get_node_status(node.status.conditions)
            roles = node.metadata.labels['kubernetes.io/role']
            age = self.calculate_age(node.metadata.creation_timestamp)
            version = node.status.node_info.kubelet_version
            resources = self.get_node_resources(name)
            cpu_capacity = resources['cpu_capacity']
            memory_capacity = np.round(int(resources['memory_capacity'].rstrip('Gi').strip('K')))

            node_data = {
                'name': name,
                'status': status,
                'roles': roles,
                'age': age,
                'version': version,
                'cpu_capacity': cpu_capacity,
                'memory_capacity': memory_capacity,
                'gpu_capacity': resources['gpu_capacity'],
                'total_cpu_used': resources['total_cpu_used'],
                'total_memory_used': resources['total_memory_used'],
                'total_gpu_used': resources['total_gpu_used'],
                'pod_count':pod_count,
                'pod_limit':POD_LIMIT
            }
            if not include_controller and node_data['roles'] != 'control-plane':
                nodes_data.append(node_data)
            if include_controller:
                nodes_data.append(node_data)
        if sort_by_cpu:
            nodes_data.sort(key=lambda x: x['total_cpu_used'] / x['cpu_capacity'] if x['cpu_capacity'] > 0 else float('inf'))
        return nodes_data

    def get_node_data_single_input(self,sort_by_cpu=False,include_controller=False):
        '''
        To use as input to ordinary NN.  One single row of data as input.  
        '''
        single_row_of_data = []
        data = self.get_nodes_data(sort_by_cpu=False,include_controller=include_controller)
        for node in data:
            #node_index = self.node_name_to_index_mapping(node['name'])
            single_row_of_data.append(np.round(1- np.round(node['total_cpu_used']/node['cpu_capacity'],4),4))
            single_row_of_data.append(np.round(1- np.round(node['total_memory_used']/node['memory_capacity'],4),4))
            single_row_of_data.append(np.round(1- np.round(node['pod_count']/node['pod_limit'],4)))

        return single_row_of_data

    def get_node_data_single_inputCPU(self,sort_by_cpu=False,include_controller=False):
        '''
        To use as input to ordinary NN.  One single row of data as input.  
        '''
        single_row_of_data = []
        data = self.get_nodes_data(sort_by_cpu=False,include_controller=include_controller)
        for node in data:
            #node_index = self.node_name_to_index_mapping(node['name'])
            single_row_of_data.append(np.round(1-np.round(node['total_cpu_used']/node['cpu_capacity'],4),4))
            #single_row_of_data.append(np.round(1- np.round(node['total_memory_used']/node['memory_capacity'],4),4))
            #single_row_of_data.append(np.round(1- np.round(node['pod_count']/node['pod_limit'],4)))

        return single_row_of_data

if __name__ == "__main__":
    kube_info = KubeInfo()
    kube_info.display_nodes_info()
