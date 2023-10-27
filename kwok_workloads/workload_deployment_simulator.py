from kubernetes import client, config
import random
import numpy as np
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkloadDeploymentSimulator:
    def __init__(self,load=0.5):
        # Load the kube config from the default location
        config.load_kube_config()

        # Create an instance of the API class
        self.api_instance = client.AppsV1Api()
        self.v1 = client.CoreV1Api()
        self.batch_api_instance = client.BatchV1Api()
        self.my_deployments = []
        self.my_pods = []
        self.load = load
        self.current_load = 0.0
        logging.info("Deployment Simulator Initialized")


    def create_kwok_deployment(self, deployment_name, replicas=1, cpu_request="250m", memory_request="512Mi", cpu_limit="500m", memory_limit="1Gi"):
        # Define the deployment manifest
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
                        scheduler_name="custom-scheduler",
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
        self.api_instance.create_namespaced_deployment(namespace="default", body=deployment)
        self.my_deployments.append(deployment_name)


    def create_kwok_job(self, job_name, cpu_request="250m", memory_request="512Mi", cpu_limit="500m", memory_limit="1Gi"):
        # Define the job manifest
        sleep_duration = str(random.randint(5, 300))
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels={"app": job_name}),
                    spec=client.V1PodSpec(
                        restart_policy="Never",  # Important for Jobs
                        containers=[
                            client.V1Container(
                                name=f"{job_name}-container",
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
        namespace="default"
        self.batch_api_instance.create_namespaced_job(namespace, job)


    def generate_funny_name(self):
        import randomname
        while True:
            name = randomname.get_name().lower()
            if name not in self.my_deployments:
                return name


    def create_random_deployment_from_prompt(self):
        default_name = self.generate_funny_name()
        deployment_name = input(f"Enter the deployment name (default: '{default_name}'): ") or default_name
        replicas = int(input("Enter the number of replicas (default: 1): ") or 1)
        cpu_request = input("Enter CPU request (default: 250m for 0.25 CPU): ") or "250m"
        memory_request = input("Enter memory request (default: 512Mi): ") or "512Mi"
        cpu_limit = input("Enter CPU limit (default: 500m for 0.5 CPU): ") or "500m"
        memory_limit = input("Enter memory limit (default: 1Gi): ") or "1Gi"

        self.create_kwok_deployment(deployment_name, replicas, cpu_request, memory_request, cpu_limit, memory_limit)

    def create_fully_random_deployment(self):
        # Generate a random deployment name
        deployment_name = self.generate_funny_name()
        
        # Generate random values for other parameters
        replicas = random.randint(1, 5)  # Random number of replicas between 1 and 5
        
        cpu_request_values = {100: "100m", 250: "250m", 500: "500m"}
        memory_request_values = {256: "256Mi", 512: "512Mi", 1024: "1Gi"}
        cpu_limit_values = {500: "500m", 750: "750m", 1000: "1000m"}
        memory_limit_values = {512: "512Mi", 1024: "1Gi", 2048: "2Gi"}

        # Select a random CPU request and then choose a CPU limit that's greater than or equal to the request
        cpu_request_key = random.choice(list(cpu_request_values.keys()))
        cpu_request = cpu_request_values[cpu_request_key]
        valid_cpu_limits = {k: v for k, v in cpu_limit_values.items() if k >= cpu_request_key}
        cpu_limit = valid_cpu_limits[random.choice(list(valid_cpu_limits.keys()))]

        # Select a random Memory request and then choose a Memory limit that's greater than or equal to the request
        memory_request_key = random.choice(list(memory_request_values.keys()))
        memory_request = memory_request_values[memory_request_key]
        valid_memory_limits = {k: v for k, v in memory_limit_values.items() if k >= memory_request_key}
        memory_limit = valid_memory_limits[random.choice(list(valid_memory_limits.keys()))]

        self.create_kwok_deployment(deployment_name, replicas, cpu_request, memory_request, cpu_limit, memory_limit)
        return deployment_name

 

    def get_pod_count_per_node(self, node_name):
        """Get the number of pods running on a specific node."""
        pods = self.v1.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node_name}").items
        return len(pods)
    
    def get_nodes_data(self):
        POD_LIMIT=110
        nodes_data = []
        nodes = self.v1.list_node().items

        for node in nodes:
            name = node.metadata.name
            pod_count = self.get_pod_count_per_node(name)
  
            node_data = {
                'name': name,
                'pod_count':pod_count,
                'pod_limit':POD_LIMIT,
                'pod_percent': np.round(pod_count/POD_LIMIT,2)
            }
            nodes_data.append(node_data)
        return nodes_data
 
    def get_load(self):
        total_pct = 0
        for node in self.get_nodes_data():
            total_pct += node['pod_percent']
        self.current_load = total_pct
        return total_pct

    def initial_deployments(self):
        '''
        Creates a set of initial deployments that will simulate a load
        '''
        total_pct = self.get_load()
        while total_pct < self.load:
            self.create_fully_random_deployment()
            total_pct = total_pct/len(self.get_nodes_data())
            logging.info(f'total percentage is {total_pct}')
            self.current_load = total_pct
    
    def poll_deployments(self):
        '''
        Every 20 seconds, do something to a deployment
        '''
        while True:
            if self.get_load() < 0.5:
                action = random.choice(['add','delete','nothing'])
                if action == 'add':
                    d_name = self.create_fully_random_deployment()
                    logging.info("added a deployment")
                elif action == 'delete':
                    d_name = random.choice(self.my_deployments)
                    self.api_instance.delete_namespaced_deployment(name=d_name,namespace='default')
                    self.my_deployments.remove(d_name)
                    logging.info(f'Removed deployment  {d_name}')
                else:
                    logging.info("Doing Nothing")
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
    simulator = WorkloadDeploymentSimulator(load=0.5)
    #simulator.create_fully_random_deployment()
    simulator.run()
