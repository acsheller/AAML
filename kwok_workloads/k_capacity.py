import subprocess
import json
from datetime import datetime, timezone
import numpy as np

def convert_memory_to_gigabytes(memory_str):
    """Convert memory string to Gigabytes (GB)"""
    if memory_str.endswith('Ki'):
        return int(memory_str.rstrip('Ki')) / (1024 * 1024)
    elif memory_str.endswith('Mi'):
        return int(memory_str.rstrip('Mi')) / 1024
    elif memory_str.endswith('Gi'):
        return int(memory_str.rstrip('Gi'))
    else:
        return int(memory_str)  # Assuming it's already in Gi




def calculate_age(creation_timestamp):
    creation_date = datetime.strptime(creation_timestamp, '%Y-%m-%dT%H:%M:%SZ')
    creation_date = creation_date.replace(tzinfo=timezone.utc)
    age_delta = datetime.now(timezone.utc) - creation_date
    age_str = "{}d".format(age_delta.days)
    return age_str

def get_node_resources(name):
    node_raw = subprocess.check_output(["kubectl", "get", "node", name, "-o", "json"])
    node_json = json.loads(node_raw)
    capacity = node_json['status']['capacity']
    allocatable = node_json['status']['allocatable']
    total_cpu_used = 0
    total_memory_used = 0
    total_gpu_used = 0
    pods_raw = subprocess.check_output(["kubectl", "get", "pods", "--all-namespaces", "-o", "json", "--field-selector=spec.nodeName=" + name])
    pods_json = json.loads(pods_raw)
    pods = pods_json['items']
    for pod in pods:
        for container in pod['spec']['containers']:
            resources = container.get('resources', {}).get('requests', {})
            total_cpu_used += int(resources.get('cpu', '0m').rstrip('m'))
            total_memory_used += convert_memory_to_gigabytes(resources.get('memory', '0Gi'))
            total_gpu_used += int(resources.get('nvidia.com/gpu', '0'))
    return {
        'cpu_capacity': capacity['cpu'],
        'memory_capacity': capacity['memory'],
        'gpu_capacity': capacity.get('nvidia.com/gpu', '0'),
        'total_cpu_used': total_cpu_used,
        'total_memory_used': total_memory_used,
        'total_gpu_used': total_gpu_used
    }

def get_node_status(conditions):
    # If there's only one condition, return its status
    if len(conditions) == 1:
        return "Ready" if conditions[0]['status'] == "True" else "Not Ready"
    
    # If there are multiple conditions, search for the 'Ready' condition
    ready_status = next((condition['status'] for condition in conditions if condition['type'] == 'Ready'), 'Unknown')
    return "Ready" if ready_status == "True" else "Not Ready"

def get_pod_count_for_node(node_name):
    pods_raw = subprocess.check_output(["kubectl", "get", "pods", "--all-namespaces", "--field-selector=spec.nodeName=" + node_name, "-o", "json"])
    pods_json = json.loads(pods_raw)
    return len(pods_json['items'])


def main():
    nodes_raw = subprocess.check_output(["kubectl", "get", "nodes", "-o", "json"])
    nodes_json = json.loads(nodes_raw)
    nodes = nodes_json['items']
    print("{:<20} {:<10} {:<15} {:<10} {:<13} {:<7} {:>13} {:>7}".format(
        "NAME", "STATUS", "ROLES", "AGE", "VERSION", "CPU", "MEMORY", "Pods"
    ))
    for node in nodes:
        name = node['metadata']['name']
        pod_count = get_pod_count_for_node(name)
        status = get_node_status(node['status']['conditions'])
        role_labels = [role.split("/")[-1] for role in node['metadata']['labels'].keys() if 'node-role.kubernetes.io/' in role]
        roles = ', '.join(role_labels) if role_labels else 'None'
        age = calculate_age(node['metadata']['creationTimestamp'])
        version = node['status']['nodeInfo']['kubeletVersion']
        if 'control-plane' in roles:
            resources = get_node_resources(name)
            print("{:<20} {:<10} {:<15} {:<10} {:<10}  {:>5}/{:<5}  {:>4}/{:<6}  {:>5}".format(
                name, status, roles, age, version,
                np.round(resources['total_cpu_used']/1000,2), resources['cpu_capacity'],
                np.round(resources['total_memory_used']/1000,2), str(np.round(float(resources['memory_capacity'].rstrip("Gi").rstrip("K"))/(1000**2),2)) + "G",
                pod_count
            ))
            continue
        resources = get_node_resources(name)
        print("{:<20} {:<10} {:<15} {:<10} {:<10}  {:>5}/{:<5}  {:>6}/{:<7}  {}".format(
            name, status, roles, age, version,
            np.round(resources['total_cpu_used']/1000,2), resources['cpu_capacity'],
            np.round(resources['total_memory_used'],2), resources['memory_capacity'].rstrip("Mi"),
            pod_count
        ))

if __name__ == "__main__":
    main()
