import subprocess
import json
from datetime import datetime, timezone


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

def main():
    nodes_raw = subprocess.check_output(["kubectl", "get", "nodes", "-o", "json"])
    nodes_json = json.loads(nodes_raw)
    nodes = nodes_json['items']
    print("{:<20} {:<10} {:<15} {:<10} {:<13} {:<7} {:<10} {:<4}".format(
        "NAME", "STATUS", "ROLES", "AGE", "VERSION", "CPU", "MEMORY", "GPU"
    ))
    for node in nodes:
        name = node['metadata']['name']
        status = get_node_status(node['status']['conditions'])
        role_labels = [role.split("/")[-1] for role in node['metadata']['labels'].keys() if 'node-role.kubernetes.io/' in role]
        roles = ', '.join(role_labels) if role_labels else 'None'
        age = calculate_age(node['metadata']['creationTimestamp'])
        version = node['status']['nodeInfo']['kubeletVersion']
        if 'control-plane' in roles:
            resources = get_node_resources(name)
            print("{:<20} {:<10} {:<15} {:<10} {:<10}  {}/{}  {}/{}  {}/{}".format(
                name, status, roles, age, version,
                resources['total_cpu_used'], resources['cpu_capacity'],
                resources['total_memory_used'], resources['memory_capacity'].rstrip("Gi") + "Mi",
                resources['total_gpu_used'], resources['gpu_capacity']))
            continue
        resources = get_node_resources(name)
        print("{:<20} {:<10} {:<15} {:<10} {:<10} {}/{}  {}/{}  {}/{}".format(
            name, status, roles, age, version,
            resources['total_cpu_used'], resources['cpu_capacity'],
            resources['total_memory_used'], resources['memory_capacity'].rstrip("Gi") + "Mi",
            resources['total_gpu_used'], resources['gpu_capacity']
        ))

if __name__ == "__main__":
    main()
