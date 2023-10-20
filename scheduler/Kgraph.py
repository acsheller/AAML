import networkx as nx
import matplotlib.pyplot as plt
from kubernetes import client, config

# Set up the Kubernetes client
config.load_kube_config()
v1 = client.CoreV1Api()
metrics_api = client.MetricsV1beta1Api()

# Create a new directed graph
G = nx.DiGraph()

# Add a central Controller node
G.add_node("Controller", type="Controller")

# Fetch and add K8s Nodes to the graph, and connect them to the Controller
nodes = v1.list_node().items
for node in nodes:
    node_name = node.metadata.name
    G.add_node(node_name, type="Node")
    G.add_edge(node_name, "Controller")

# Fetch pods and their metrics, then add them to the graph
pod_metrics_list = metrics_api.list_pod_metrics_for_all_namespaces().items
for pod_metrics in pod_metrics_list:
    pod_name = pod_metrics.metadata.name
    memory_usage = None
    cpu_usage = None
    for container in pod_metrics.containers:
        # Summing memory and CPU usage for each container in the pod
        memory_usage = (memory_usage or 0) + int(container.usage['memory'].strip('Ki'))
        cpu_usage = (cpu_usage or 0) + int(container.usage['cpu'].strip('m'))

    label = f"{pod_name}\nCPU: {cpu_usage}m\nMem: {memory_usage}Ki"
    G.add_node(label, type="Pod")

# Visualization
pos = nx.spring_layout(G)
node_colors = ["red" if node == "Controller" else ("blue" if G.nodes[node]["type"] == "Node" else "green") for node in G.nodes()]

nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=10)
plt.show()
