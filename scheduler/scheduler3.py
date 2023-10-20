from kubernetes import client, config, watch
from kinfo import KubeInfo
import redis
import json
import random
import networkx as nx
class CustomScheduler:

    def __init__(self):
        config.load_kube_config()
        self.api = client.CoreV1Api()
        self.watcher = watch.Watch()
        self.kube_info = KubeInfo()
        #self.redis_client = redis.StrictRedis(host = "localhost",port=6379, db=0)

    def push_scheduler_status_to_redis(self):
        nodes_data = self.kube_info.get_nodes_data()
        for node_data in nodes_data:
            node_name = node_data['name']
            serialized_data = json.dumps(node_data)
            self.redis_client.set(node_name, serialized_data)

    def get_node_data_from_redis(self, node_name):
        node_data_json = self.redis_client.get(node_name)
        if node_data_json:
            return json.loads(node_data_json)
        return None


    def needs_scheduling(self, pod):
        return (
            pod.status.phase == "Pending" and
            not pod.spec.node_name and
            pod.spec.scheduler_name == "custom-scheduler"
        )


    def select_best_random_node(self,nodes,pod):
 
        selected_node = random.choice(nodes)
        return selected_node

    def schedule_pod(self, pod):
        print("Entering Scheduler Pod")
        nodes = self.api.list_node().items
        best_node = None
        gpu_agents = []
        agents = []

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


        print("Leaving Scheduler Pod {}".format(best_node.metadata.name))
        return best_node.metadata.name


    def bind_pod_to_node(self, pod, node_name):
        if node_name is None:
            print("Node name is None. Cannot bind pod.")
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

        print(f"Binding object: {binding}")

        try:
            self.api.create_namespaced_binding(namespace=pod.metadata.namespace, body=binding,_preload_content=False)
        except Exception as e:
            print(f"Exception when calling CoreV1Api->create_namespaced_binding: {e}")


    def create_graph(self):
        G = nx.Graph()

        # Add the controller as the central node
        G.add_node("controller", type="controller")

        # Fetch all nodes data
        nodes_data = self.kube_info.get_nodes_data()

        # Add nodes to the graph
        for node_data in nodes_data:
            node_name = node_data['name']
            
            # Add or update node attributes
            for key, value in node_data.items():
                G.nodes[node_name][key] = value
            
            G.add_edge("controller", node_name, weight=1)

        return G

    def run(self):
        
        # Push Iniital Status
        #print("Initial Status Pushed to Redis")
        #self.push_scheduler_status_to_redis()
        for event in self.watcher.stream(self.api.list_pod_for_all_namespaces):
            pod = event['object']
            if self.needs_scheduling(pod):
                best_node = self.schedule_pod(pod)
                print("Best Node Selected {}".format(best_node))
                self.bind_pod_to_node(pod, best_node)



if __name__ == "__main__":
    scheduler = CustomScheduler()
    scheduler.run()
