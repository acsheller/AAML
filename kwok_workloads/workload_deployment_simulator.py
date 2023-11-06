from kubernetes import client, config
import random
import numpy as np
import time
from datetime import datetime, timezone
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import networkx


class WorkloadDeploymentSimulator:

    POD_LIMIT = 110

    def __init__(self,cpu_load=0.5,mem_load=0.5,pod_load=0.5,namespace='default',scheduler='default-scheduler'):

        self.namespace = namespace
        self.scheduler = scheduler
        # Load the kube config from the default location
        config.load_kube_config()
        
        # Create an instance of the API class
        self.api_instance = client.AppsV1Api()
        
        self.v1 = client.CoreV1Api()
        self.batch_api_instance = client.BatchV1Api()
        self.my_deployments = self.get_all_deployments()

        self.cpu_load = cpu_load
        self.mem_load = mem_load
        self.pod_load = pod_load

        self.current_load = {}
        self.current_load['pod_pct'] = 0
        self.current_load['cpu_pct'] = 0
        self.current_load['mem_pct'] = 0

        self.cpu_request_values = {100: "100m", 250: "250m", 500: "500m", 1000: "1000m", 1500: "1500m",2000: "2000m",2500: "2500m",3000: "3000m",3500: "3500m"}
        self.memory_request_values = {256: "256Mi", 512: "512Mi", 1024: "1.0Gi", 1536: "1.5Gi",2048: "2.0Gi",2560: "2.5Gi", 3072: "3.0Gi",3584:"3.5Gi"}
        self.cpu_limit_values = {500: "500m", 750: "750m", 1000: "1000m",1500: "1500m",2000: "2000m",2500: "2500m",3000: "3000m",3500: "3500m",4000:"4000m"}
        self.memory_limit_values = {512: "512Mi", 1024: "1Gi",1536: "1.5Gi", 2048: "2Gi",2560: "2.5Gi", 3072: "3.0Gi",3584:"3.5Gi",4096: "4.0Gi"}

        columns = ['name', 'pod_count', 'cpu_request', 'cpu_limit', 'mem_request', 'mem_limit', 'action']
        self.df = pd.DataFrame(columns=columns)
        
        logging.info("Deployment Simulator Initialized")


        
    def get_all_deployments(self):
        deployments = self.api_instance.list_namespaced_deployment(namespace=self.namespace)
        deployment_names = [deployment.metadata.name for deployment in deployments.items]
        return deployment_names
            
            
    def create_kwok_deployment(self, deployment_name, replicas=1, cpu_request="250m", memory_request="512Mi", cpu_limit="500m", memory_limit="1Gi"):
        # Define the deployment manifest
        try:
            sleep_duration = str(random.randint(5,300))
            deployment = client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=client.V1ObjectMeta(name=deployment_name),
                spec=client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=client.V1LabelSelector(
                        match_labels={"app": deployment_name}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(labels={"app": deployment_name}),
                        spec=client.V1PodSpec(
                            scheduler_name=self.scheduler,
                            containers=[
                                client.V1Container(
                                    name=f"{deployment_name}-container",
                                    image="busybox",
                                    resources=client.V1ResourceRequirements(
                                        requests={"cpu": cpu_request, "memory": memory_request},
                                        limits={"cpu": cpu_limit, "memory": memory_limit}
                                    ),
                                    command=["sleep", sleep_duration]
                                )
                            ]
                        )
                    )
                )
            )
            # Create the deployment
            self.api_instance.create_namespaced_deployment(namespace=self.namespace, body=deployment)
            logging.info(f" {deployment_name } Deployed, cpu  r {cpu_request} l {cpu_limit}, mem r {memory_request} l {memory_limit}")
            self.df.loc[len(self.df)] = {'name': deployment_name,'pod_count': replicas,'cpu_request': cpu_request,'cpu_limit': cpu_limit,'mem_request':memory_request,'mem_limit': memory_limit,'action': 'add'}
            self.my_deployments.append(deployment_name)
        except:
            logging.error(f"something wrong with deployment {deployment_name}-- ignoring")

    def contains_non_alpha_or_dash(self,s):
        for char in s:
            if not char.isalpha() and char != '-':
                return True
        return False
            
    def generate_funny_name(self):
        import randomname
        while True:
            name = randomname.get_name().lower()
            if not self.contains_non_alpha_or_dash(name):
                if name not in self.my_deployments:    
                    return name


    def create_fully_random_deployment(self):
        # Generate a random deployment name
        deployment_name = self.generate_funny_name()
        
        # Generate random values for other parameters
        replicas = random.randint(1, 5)  # Random number of replicas between 1 and 5
        
        # Select a random CPU request and then choose a CPU limit that's greater than or equal to the request
        cpu_request_key = random.choice(list(self.cpu_request_values.keys()))
        cpu_request = self.cpu_request_values[cpu_request_key]
        valid_cpu_limits = {k: v for k, v in self.cpu_limit_values.items() if k >= cpu_request_key}
        cpu_limit = valid_cpu_limits[random.choice(list(valid_cpu_limits.keys()))]

        # Select a random Memory request and then choose a Memory limit that's greater than or equal to the request
        memory_request_key = random.choice(list(self.memory_request_values.keys()))
        memory_request = self.memory_request_values[memory_request_key]
        valid_memory_limits = {k: v for k, v in self.memory_limit_values.items() if k >= memory_request_key}
        memory_limit = valid_memory_limits[random.choice(list(valid_memory_limits.keys()))]

        self.create_kwok_deployment(deployment_name, replicas, cpu_request, memory_request, cpu_limit, memory_limit)
        return deployment_name

    def get_node_status(self, conditions):
        # If there's only one condition, return its status
        if len(conditions) == 1:
            return "Ready" if conditions[0].status == "True" else "Not Ready"
        
        # If there are multiple conditions, search for the 'Ready' condition
        ready_status = next((condition.status for condition in conditions if condition.type == 'Ready'), 'Unknown')
        return "Ready" if ready_status == "True" else "Not Ready"
 
    def calculate_age(self, creation_timestamp):
        if not isinstance(creation_timestamp, datetime):
            creation_date = datetime.strptime(creation_timestamp, '%Y-%m-%dT%H:%M:%SZ')
        else:
            creation_date = creation_timestamp

        creation_date = creation_date.replace(tzinfo=timezone.utc)
        age_delta = datetime.now(timezone.utc) - creation_date
        age_str = "{}d".format(age_delta.days)
        return age_str
    
    def get_pod_count_per_node(self, node_name):
        """Get the number of pods running on a specific node."""
        pods = self.v1.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node_name}").items
        return len(pods)


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

    def get_nodes_data(self): 
        POD_LIMIT=110
        nodes_data = []
        nodes = self.v1.list_node().items

        for node in nodes:

            name = node.metadata.name
            if name == 'control_plane':
                continue
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
                'memory_capacity': np.round(self.convert_memory_to_gigabytes(resources['memory_capacity']),2),
                'gpu_capacity': resources['gpu_capacity'],
                'total_cpu_used': resources['total_cpu_used'],
                'total_memory_used': resources['total_memory_used'],
                'total_gpu_used': resources['total_gpu_used'],
                'pod_count':pod_count,
                'pod_limit':POD_LIMIT
            }
            nodes_data.append(node_data)
        return nodes_data


    def get_load(self):
        total_pods = 0
        total_pod_count = 0
        total_cpu_used = 0
        total_cpu = 0
        total_mem = 0
        total_mem_used =0

        for node in self.get_nodes_data():

            if node['roles'] == 'control-plane':
                continue
            total_pods += node['pod_count']
            total_pod_count += node['pod_limit']
            total_cpu += node['cpu_capacity']
            total_cpu_used += node['total_cpu_used']
            total_mem += node['memory_capacity']
            total_mem_used += node['total_memory_used']
        self.current_load['pod_pct'] = np.round(total_pods/total_pod_count,2)
        self.current_load['cpu_pct'] = np.round(total_cpu_used/total_cpu,2)
        self.current_load['mem_pct'] = np.round(total_mem_used/total_mem,2)
        return self.current_load

    def initial_deployments(self):
        '''
        Creates a set of initial deployments that will simulate a load
        '''
        self.get_load()
        while self.current_load['cpu_pct'] < self.cpu_load and self.current_load['mem_pct'] < self.mem_load and self.current_load['pod_pct'] < self.pod_load:
            self.create_fully_random_deployment()
            self.get_load()
            logging.info(f'Current load {self.current_load}')
            
    def poll_deployments(self, interval =20,duration = 1):
        '''
        Run for one hour after reaching capacity then
        Every 20 seconds, do something to a deployment

        interval: polling interval -- default is every 20 seconds.
        duration: runtime in hours -- default is 1 
        '''
        start_time = time.time()
        run_this_long = duration*3600 # 3600 seconds in an hour

        while time.time() - start_time < run_this_long:
            # Keep track of the load on the system
            self.get_load()
            logging.info(f"Polling: Load currently at {self.current_load}")  
            # Try to add more frequently than delete
            action = random.choice(['add','delete','nothing','add'])
            if action == 'add':
                if self.current_load['cpu_pct'] < self.cpu_load and self.current_load['mem_pct'] < self.mem_load and self.current_load['pod_pct'] < self.pod_load:
                    d_name = self.create_fully_random_deployment()
                    self.my_deployments.append(d_name)
                    logging.info("added a deployment")
                    
                else:
                    logging.info(f"Tried to add a deployment but would exceed {self.current_load} Limits")
            elif action == 'delete':
                if len(self.my_deployments) > 0:
                    d_name = random.choice(self.my_deployments)
                    record = self.df[self.df['name']==d_name].iloc[0].copy()
                    try: 
                        self.api_instance.delete_namespaced_deployment(name=d_name, namespace='default')
                        record['action'] = 'delete'
                        self.df.loc[len(self.df)] = record
                    except client.rest.ApiException as e:
                        logging.error(f"Error removing deployment {d_name}: {e.reason}")
                    except Exception as e:
                        logging.error(f"Unexpected error while removing deployment {d_name}: {str(e)}")

                    self.my_deployments.remove(d_name)
                    logging.info(f'Removed deployment  {d_name}. Lenght of Deployment is {len(self.my_deployments)}')
                else:
                    logging.info("No deployments to delete")
            else:
                logging.info("Skipping this one and doing nothing")
        else:
            logging.info(f"Cluster load: {self.current_load}, specified load: {self.load} ")
        time.sleep(20)

  
    def run(self):
        """
        Run method to start the simulator.
        """
        self.initial_deployments()
        self.poll_deployments()


if __name__ == "__main__":
    simulator = WorkloadDeploymentSimulator(cpu_load=0.75,mem_load=0.80,pod_load=0.80,scheduler='custom-scheduler')
    #simulator.create_fully_random_deployment()
    simulator.run()