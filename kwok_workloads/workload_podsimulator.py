from kubernetes import client, config
import random
import numpy as np
import time

class WorkloadPodSimulator:
    def __init__(self):
        # Load the kube config from the default location
        config.load_kube_config()

        # Create an instance of the API class
        self.v1 = client.CoreV1Api()
        self.my_pods = []

    def create_kwok_pod(self, pod_name, cpu_request="250m", memory_request="512Mi", cpu_limit="500m", memory_limit="1Gi"):
        # Define the pod manifest
        sleep_duration = str(random.randint(5,300))
        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(name=pod_name),
            spec=client.V1PodSpec(
                scheduler_name="custom-scheduler",
                containers=[
                    client.V1Container(
                        name=f"{pod_name}-container",
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
        # Create the pod
        self.v1.create_namespaced_pod(namespace="default", body=pod)
        self.my_pods.append(pod_name)

    def delete_random_pod(self):
        if self.my_pods:
            pod_name = random.choice(self.my_pods)
            self.v1.delete_namespaced_pod(name=pod_name, namespace="default")
            self.my_pods.remove(pod_name)

    def generate_funny_name(self):
        import randomname
        while True:
            name = randomname.get_name().lower()
            if name not in self.my_pods:
                return name

    def poll_pods(self, period=20):
        '''
        Every 20 seconds, do something to a pod
        '''
        while True:
            action = random.choice(['add','delete','nothing'])
            if action == 'add':
                self.create_kwok_pod(self.generate_funny_name())
                print("Added a pod.")
            elif action == 'delete':
                self.delete_random_pod()
                print("Deleted a pod.")
            time.sleep(period)



    def run(self):
        """
        Run method to start the simulator.
        """
        self.initial_deployments()
        self.poll_deployments()


simulator = WorkloadPodSimulator()
simulator.poll_pods()
