from kubernetes import client, config
import random

class KwokDeploymentSimulator:
    def __init__(self):
        # Load the kube config from the default location
        config.load_kube_config()

        # Create an instance of the API class
        self.api_instance = client.AppsV1Api()

        self.batch_api_instance = client.BatchV1Api()

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
        adjectives = ["funky", "jazzy", "silly", "wacky", "zany", "quirky", "goofy", "wonky", "spunky", "loony"]
        nouns = ["penguin", "llama", "unicorn", "gnome", "kitten", "chimp", "octopus", "dino", "robot", "alien"]
        return random.choice(adjectives) + '-' + random.choice(nouns)

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
        cpu_request_values = ["100m", "250m", "500m"]
        memory_request_values = ["256Mi", "512Mi", "1Gi"]
        cpu_limit_values = ["500m", "750m", "1"]
        memory_limit_values = ["512Mi", "1Gi", "2Gi"]

        cpu_request = random.choice(cpu_request_values)
        memory_request = random.choice(memory_request_values)
        cpu_limit = random.choice(cpu_limit_values)
        memory_limit = random.choice(memory_limit_values)

        self.create_kwok_deployment(deployment_name, replicas, cpu_request, memory_request, cpu_limit, memory_limit)

    def create_fully_random_job(self):
        # Generate a random job name
        job_name = self.generate_funny_name()
        
        # Generate random values for other parameters
        cpu_request_values = ["100m", "250m", "500m"]
        memory_request_values = ["256Mi", "512Mi", "1Gi"]
        cpu_limit_values = ["500m", "750m", "1"]
        memory_limit_values = ["512Mi", "1Gi", "2Gi"]

        cpu_request = random.choice(cpu_request_values)
        memory_request = random.choice(memory_request_values)
        cpu_limit = random.choice(cpu_limit_values)
        memory_limit = random.choice(memory_limit_values)

        self.create_kwok_job(job_name, cpu_request, memory_request, cpu_limit, memory_limit)


if __name__ == "__main__":
    simulator = KwokDeploymentSimulator()
    #simulator.create_fully_random_deployment()
    simulator.create_fully_random_job()
    print()
