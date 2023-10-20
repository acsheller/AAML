from kubernetes import client, config
from datetime import datetime, timezone
import json
import numpy as np
class KubeInfo:

    POD_LIMIT = 110

    def __init__(self):
        config.load_kube_config()
        self.v1 = client.CoreV1Api()

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
            role_labels = [role.split("/")[-1] for role in node.metadata.labels.keys() if 'node-role.kubernetes.io/' in role]
            roles = ', '.join(role_labels) if role_labels else 'None'
            age = self.calculate_age(node.metadata.creation_timestamp)
            version = node.status.node_info.kubelet_version
            if 'control-plane' in roles:
                resources = self.get_node_resources(name)
                print("{:<18} {:<8} {:<14} {:<7} {:<10} {:>5}/{:<5}  {:>7}/{:<5}    {}/{}   {}/{}".format(
                    name, status, roles, age, version,
                    resources['total_cpu_used'], resources['cpu_capacity'],
                    np.round(resources['total_memory_used'],4), str(int(resources['memory_capacity'].rstrip("Gi").rstrip("Ki"))//1024),
                    resources['total_gpu_used'], resources['gpu_capacity'],
                    pod_count,POD_LIMIT))
                continue
            resources = self.get_node_resources(name)
            print("{:<18} {:<8} {:<14} {:<7} {:<10} {:>5}/{:<5}  {:>7}/{:<5}    {}/{}   {}/{}".format(
                name, status, roles, age, version,
                resources['total_cpu_used'], resources['cpu_capacity'],
                resources['total_memory_used'], resources['memory_capacity'].rstrip("Gi") + "Mi",
                resources['total_gpu_used'], resources['gpu_capacity'],
                pod_count,POD_LIMIT
            ))

    def get_nodes_data(self):
        POD_LIMIT=110
        nodes_data = []
        nodes = self.v1.list_node().items

        for node in nodes:
            name = node.metadata.name
            pod_count = self.get_pod_count_per_node(name)
            status = self.get_node_status(node.status.conditions)
            role_labels = [role.split("/")[-1] for role in node.metadata.labels.keys() if 'node-role.kubernetes.io/' in role]
            roles = ', '.join(role_labels) if role_labels else 'None'
            age = self.calculate_age(node.metadata.creation_timestamp)
            version = node.status.node_info.kubelet_version
            resources = self.get_node_resources(name)
            
            node_data = {
                'name': name,
                'status': status,
                'roles': roles,
                'age': age,
                'version': version,
                'cpu_capacity': resources['cpu_capacity'],
                'memory_capacity': resources['memory_capacity'],
                'gpu_capacity': resources['gpu_capacity'],
                'total_cpu_used': resources['total_cpu_used'],
                'total_memory_used': resources['total_memory_used'],
                'total_gpu_used': resources['total_gpu_used'],
                'pod_count':pod_count,
                'pod_limit':POD_LIMIT
            }
            nodes_data.append(node_data)
        return nodes_data


if __name__ == "__main__":
    kube_info = KubeInfo()
    kube_info.display_nodes_info()
    ab = kube_info.get_nodes_data()
    print("horray")