from kubernetes import client, config
import random
import numpy as np
import time
from datetime import datetime, timezone
import pandas as pd
import logging
import os

# For progress display
start_time = time.time()

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"logs/WDS_log_{current_time}.log"

# Create a named logger
logger = logging.getLogger('MyWDSLogger')
logger.setLevel(logging.INFO)

# Create file handler which logs even debug messages
fh = logging.FileHandler(filename)
fh.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)

# Prevent the log messages from being propagated to the Jupyter notebook
logger.propagate = False


class WorkloadDeploymentSimulator:


    def __init__(self,cpu_load=0.5,mem_load=0.5,pod_load=0.5,pod_limit=110,namespace='default',scheduler='default-scheduler',progress_indication=True,epochs=2):
        '''
        Constructor
        '''
        self.progress_indication = progress_indication
        self.namespace = namespace
        self.scheduler = scheduler
        # Load the kube config from the default location
        config.load_kube_config()
        
        # Create an instance of the API class
        self.api_instance = client.AppsV1Api()
        
        self.v1 = client.CoreV1Api()
        self.batch_api_instance = client.BatchV1Api()
        self.my_deployments = self.get_all_deployments()
        self.total_pods_deployed = 0

        self.epochs = epochs     

        self.pod_load = pod_load
        self.pod_limit = pod_limit

        self.max_load = {}
        self.max_load['pod_pct'] = pod_load
        self.max_load['cpu_pct'] = cpu_load
        self.max_load['mem_pct'] = mem_load



        self.current_load = {}
        self.current_load['pod_pct'] = 0
        self.current_load['cpu_pct'] = 0
        self.current_load['mem_pct'] = 0


        #self.cpu_request_values = {100: "100m", 250: "250m", 500: "500m", 1000: "1000m", 1500: "1500m",2000: "2000m",2500: "2500m",3000: "3000m",3500: "3500m"}
        #self.memory_request_values = {256: "256Mi", 512: "512Mi", 1024: "1.0Gi", 1536: "1.5Gi",2048: "2.0Gi",2560: "2.5Gi", 3072: "3.0Gi",3584:"3.5Gi",4096: "4.0Gi",5120:"5.0Gi",10240:"10Gi",20480:"20Gi"}
        #self.cpu_limit_values = {500: "500m", 750: "750m", 1000: "1000m",1500: "1500m",2000: "2000m",2500: "2500m",3000: "3000m",3500: "3500m"}
        #self.memory_limit_values = { 1024: "1Gi",1536: "1.5Gi", 2048: "2Gi",2560: "2.5Gi", 3072: "3.0Gi",3584:"3.5Gi",4096: "4.0Gi",5120:"5.0Gi",10240:"10Gi",20480:"20Gi"}

        #self.cpu_request_values = {500: "500m", 1000: "1000m", }
        #self.memory_request_values = {1024: "1.0Gi",2048: "2.0Gi"}
        #self.cpu_limit_values = {1000: "1000m",2000: "2000m"}
        #self.memory_limit_values = { 2048: "2Gi",3072: "3.0Gi"}


        self.cpu_request_values = {1000: "1000m", }
        self.memory_request_values = {2048: "2.0Gi"}
        self.cpu_limit_values = {1000: "1000m"}
        self.memory_limit_values = { 2048: "2Gi"}

        columns = ['name', 'pod_count', 'cpu_request', 'cpu_limit', 'mem_request', 'mem_limit', 'action']
        self.df = pd.DataFrame(columns=columns)
        
        logger.info("Deployment Simulator Initialized")


        
    def get_all_deployments(self):
        '''
        Get the deployments in the cluster.
        '''
        deployments = self.api_instance.list_namespaced_deployment(namespace=self.namespace)
        deployment_names = [deployment.metadata.name for deployment in deployments.items]
        return deployment_names
            
            
    def create_kwok_deployment(self, deployment_name, replicas=1, cpu_request="250m", memory_request="512Mi", cpu_limit="500m", memory_limit="1Gi"):
        '''
        Create a deployment consisting of 'replicas' number of pods with specific attribute requests.
        '''
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
            logger.info(f"WDS :: {deployment_name } Deployed, cpu requests {cpu_request} limits {cpu_limit}, mem requests {memory_request} limits {memory_limit} pods {replicas}")
            self.df.loc[len(self.df)] = {'name': deployment_name,'pod_count': replicas,'cpu_request': cpu_request,'cpu_limit': cpu_limit,'mem_request':memory_request,'mem_limit': memory_limit,'action': 'add'}
            self.my_deployments.append(deployment_name)
            self.total_pods_deployed += replicas
        except:
            logger.error(f"something wrong with deployment {deployment_name}-- ignoring")


    def contains_non_alpha_or_dash(self,s):
        '''
        This is necessary because some of the names have funny characters in them. 
        '''
        for char in s:
            if not char.isalpha() and char != '-':
                return True
        return False


    def generate_funny_name(self):
        '''
        I cannot recall where I saw this done but I like it so crafted this version of it.
        '''
        import randomname
        while True:
            name = randomname.get_name().lower()
            if not self.contains_non_alpha_or_dash(name):
                if name not in self.my_deployments:    
                    return name




    def create_fully_random_deployment_element(self):
        '''
        Creates a random deployment element set that can be saved to a dataframe
        '''
        # Generate a random deployment name
        deployment_name = self.generate_funny_name()
        
        # Generate random values for other parameters
        replicas = random.randint(5, 15)  # Random number of replicas between 5 and 15
        
        # Select a random CPU request and then choose a CPU limit that's greater than or equal to the request
        cpu_request_key = random.choice(list(self.cpu_request_values.keys()))
        cpu_request = self.cpu_request_values[cpu_request_key]
        valid_cpu_limits = {k: v for k, v in self.cpu_limit_values.items() if k >= cpu_request_key}
        cpu_limit = valid_cpu_limits[random.choice(list(valid_cpu_limits.keys()))]

        # Select a random Memory request that is larger than or equal to the CPU request
        valid_memory_request_keys = [k for k in self.memory_request_values.keys() if k >= cpu_request_key]
        if not valid_memory_request_keys:
            raise ValueError("No memory request values larger than or equal to the CPU request are available.")
        
        memory_request_key = random.choice(valid_memory_request_keys)
        memory_request = self.memory_request_values[memory_request_key]
        
        # Choose a Memory limit that's greater than or equal to the memory request
        valid_memory_limits = {k: v for k, v in self.memory_limit_values.items() if k >= memory_request_key}
        memory_limit = valid_memory_limits[random.choice(list(valid_memory_limits.keys()))]

        return [deployment_name, replicas, cpu_request, memory_request, cpu_limit, memory_limit]

    def create_fully_random_deployment(self):
        '''
        Creates a random deployment based on some key attributes of server nodes
        '''
        # Generate a random deployment name
        deployment_name = self.generate_funny_name()
        
        # Generate random values for other parameters
        replicas = random.randint(5, 15)  # Random number of replicas between 5 and 15
        
        # Select a random CPU request and then choose a CPU limit that's greater than or equal to the request
        cpu_request_key = random.choice(list(self.cpu_request_values.keys()))
        cpu_request = self.cpu_request_values[cpu_request_key]
        valid_cpu_limits = {k: v for k, v in self.cpu_limit_values.items() if k >= cpu_request_key}
        cpu_limit = valid_cpu_limits[random.choice(list(valid_cpu_limits.keys()))]

        # Select a random Memory request that is larger than or equal to the CPU request
        valid_memory_request_keys = [k for k in self.memory_request_values.keys() if k >= cpu_request_key]
        if not valid_memory_request_keys:
            raise ValueError("No memory request values larger than or equal to the CPU request are available.")
        
        memory_request_key = random.choice(valid_memory_request_keys)
        memory_request = self.memory_request_values[memory_request_key]
        
        # Choose a Memory limit that's greater than or equal to the memory request
        valid_memory_limits = {k: v for k, v in self.memory_limit_values.items() if k >= memory_request_key}
        memory_limit = valid_memory_limits[random.choice(list(valid_memory_limits.keys()))]

        # Create the deployment with the selected values
        self.create_kwok_deployment(deployment_name, replicas, cpu_request, memory_request, cpu_limit, memory_limit)
        return deployment_name


    def get_node_status(self, conditions):
        '''
        get the status of the node
        '''
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
        '''
        I suspect get_node_resources and get_node_data can be combined but this works so will consider it for a later task.
        TODO Revisit the get_node_resources and get_node_data and make it more efficient.

        '''
        v1 = client.CoreV1Api()
        node = v1.read_node(name)
        capacity = node.status.capacity
        allocatable = node.status.allocatable
        pod_limit = node.status.allocatable.get('pods',self.pod_limit)
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
        '''
        Primary method for getting information about nodes in the cluster.
        '''
        POD_LIMIT=self.pod_limit
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
        '''
        Get the load on the system
        '''

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

    def initial_deployments(self,epoch,interval=10):
        '''
        Creates a set of initial deployments that will simulate a load
        '''
        self.get_load()
        while self.current_load['cpu_pct'] < self.max_load['cpu_pct'] and self.current_load['mem_pct'] < self.max_load['mem_pct'] and self.current_load['pod_pct'] < self.max_load['pod_pct']:
            minutes,seconds = divmod(time.time()-start_time,60)
            if self.progress_indication:
                print(f"\rEpoch {epoch+1}/{self.epochs} {np.round(self.current_load['cpu_pct']/self.max_load['cpu_pct']*100,2)}% Complete. Notebook Elapsed time so far: {int(minutes)} minutes and {int(seconds)} seconds",end='', flush=True)
            pods = self.v1.list_namespaced_pod(namespace = self.namespace)
            if sum(1 for pod in pods.items if pod.status.phase == 'Pending') < 1:
                self.create_fully_random_deployment()
                self.get_load()
                logger.info('WDS :: Current load CPU {} MEM {} POD {}'.format(self.current_load['cpu_pct'],self.current_load['mem_pct'],self.current_load['pod_pct']))
            time.sleep(interval)

    def deployment_pending(self,deployment_name):
        # Get all pods in the specified namespace
        pods = self.v1.list_namespaced_pod(self.namespace)

        # Check if any pod related to the deployment is in pending state
        for pod in pods.items:
            if pod.metadata.owner_references and \
            any(ref.kind == 'ReplicaSet' and deployment_name in ref.name for ref in pod.metadata.owner_references) and \
            pod.status.phase == 'Pending':
                return True
        return False


  
    def run(self):
        """
        Run method to start the simulator.
        """
        for epoch in range(self.epochs):
            logger.info(f"WDS :: Epoch {epoch+1}/{self.epochs} Running")
            self.initial_deployments(epoch, interval =40)
            self.my_deployments = self.get_all_deployments()
            while sum(1 for pod in self.v1.list_namespaced_pod(namespace = self.namespace).items if pod.status.phase == 'Pending') >0:
                if not os.path.exists('epoch_complete.txt'):
                    with open(f"epoch_complete.txt","w") as file:
                        file.write(f"{epoch}")
                while  os.path.exists('epoch_complete.txt'):
                    logger.info("WDS :: Waiting on scheduler to stop scheduling")
                    # Give the scheduling agent time to catch up.
                    time.sleep(5)

            try:
                if 1 == 1:  # Did it this way so I could turn it on and off as needed.
                    #self.api_instance.delete_namespaced_deployment(name=d_name, namespace=self.namespace)
                    self.api_instance.delete_collection_namespaced_deployment(namespace=self.namespace)
                    time.sleep(2)
                    self.v1.delete_collection_namespaced_pod(namespace = self.namespace)
                    self.my_deployments = []
                    logger.info(f'WDS :: {self.namespace} Namespace cleared')
            except client.rest.ApiException as e:
                logger.error(f"WDS :: Error removing deployments: {e.reason}")
            except Exception as e:
                logger.error(f"WDS :: Unexpected error while removing deployment: {str(e)}")
            time.sleep(10)
        unique_filename = f"deployment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.df.to_csv(unique_filename,index=False)
        logger.info(f"Data saved to {unique_filename}")
        with open("shutdown_signal.txt","w") as file:
            file.write("shutdown")
        print("\n")
        return


if __name__ == "__main__":
    # Add this to the constructor to use custom scheduler: scheduler='custom-scheduler'
    simulator = WorkloadDeploymentSimulator(cpu_load=0.15,mem_load=0.50,pod_load=0.50,scheduler='custom-scheduler',epochs=3)
    simulator.run()
    
    ## Uncomment this for playback.
    #playback_df = pd.read_csv("deployment_data_20231107_094336.csv")
    #simulator.playback(playback_df,sleep_interval=15)
