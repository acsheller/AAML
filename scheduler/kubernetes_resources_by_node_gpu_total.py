from kubernetes import client, config

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

def convert_cpu_to_millicores(cpu_str):
    """Convert CPU string to millicores"""
    if 'm' in cpu_str:
        return int(cpu_str.rstrip('m'))
    else:
        return int(cpu_str) * 1000  # Convert cores to millicores

# Load kube config
config.load_kube_config()

# Initialize the API client
v1 = client.CoreV1Api()

# Get list of nodes
nodes = v1.list_node().items

# Variables to store total cluster resources
cluster_total_cpu_capacity = 0
cluster_total_memory_capacity = 0
cluster_total_gpu_capacity = 0
cluster_total_cpu_used = 0
cluster_total_memory_used = 0
cluster_total_gpu_used = 0

for node in nodes:
    node_name = node.metadata.name
    total_cpu_capacity = convert_cpu_to_millicores(node.status.capacity.get('cpu', '0'))
    total_memory_capacity = convert_memory_to_gigabytes(node.status.capacity.get('memory', '0Gi'))
    total_gpu_capacity = int(node.status.capacity.get('nvidia.com/gpu', 0))
    
    # Get all pods on the node
    field_selector = f"spec.nodeName={node_name}"
    pods = v1.list_pod_for_all_namespaces(field_selector=field_selector).items
    
    # Calculate total CPU, memory, and GPU requested by all pods on the node
    total_cpu_used_by_pods = 0
    total_memory_used_by_pods = 0
    total_gpu_used_by_pods = 0
    for pod in pods:
        for container in pod.spec.containers:
            resources = container.resources.requests or {}
            total_cpu_used_by_pods += convert_cpu_to_millicores(resources.get('cpu', '0m'))
            total_memory_used_by_pods += convert_memory_to_gigabytes(resources.get('memory', '0Gi'))
            total_gpu_used_by_pods += int(resources.get('nvidia.com/gpu', 0))
    
    # Calculate available resources
    available_cpu = total_cpu_capacity - total_cpu_used_by_pods
    available_memory = total_memory_capacity - total_memory_used_by_pods
    available_gpu = total_gpu_capacity - total_gpu_used_by_pods
    
    # Update cluster totals
    cluster_total_cpu_capacity += total_cpu_capacity
    cluster_total_memory_capacity += total_memory_capacity
    cluster_total_gpu_capacity += total_gpu_capacity
    cluster_total_cpu_used += total_cpu_used_by_pods
    cluster_total_memory_used += total_memory_used_by_pods
    cluster_total_gpu_used += total_gpu_used_by_pods

    print(f"Node: {node_name}")
    print(f"  Total CPU Capacity: {total_cpu_capacity}m")
    print(f"  Total Memory Capacity: {total_memory_capacity:.2f}GB")
    print(f"  Total GPU Capacity: {total_gpu_capacity} GPUs")
    print(f"  CPU Used by Pods: {total_cpu_used_by_pods}m")
    print(f"  Memory Used by Pods: {total_memory_used_by_pods:.2f}GB")
    print(f"  GPU Used by Pods: {total_gpu_used_by_pods} GPUs")
    print(f"  Available CPU: {available_cpu}m")
    print(f"  Available Memory: {available_memory:.2f}GB")
    print(f"  Available GPU: {available_gpu} GPUs")
    print("------")

# Print cluster totals
print("Total Cluster Resources:")
print(f"  Total CPU Capacity: {cluster_total_cpu_capacity}m")
print(f"  Total Memory Capacity: {cluster_total_memory_capacity:.2f}GB")
print(f"  Total GPU Capacity: {cluster_total_gpu_capacity} GPUs")
print(f"  Total CPU Used by Pods: {cluster_total_cpu_used}m")
print(f"  Total Memory Used by Pods: {cluster_total_memory_used:.2f}GB")
print(f"  Total GPU Used by Pods: {cluster_total_gpu_used} GPUs")
print(f"  Total Available CPU: {cluster_total_cpu_capacity - cluster_total_cpu_used}m")
print(f"  Total Available Memory: {cluster_total_memory_capacity - cluster_total_memory_used:.2f}GB")
print(f"  Total Available GPU: {cluster_total_gpu_capacity - cluster_total_gpu_used} GPUs")
