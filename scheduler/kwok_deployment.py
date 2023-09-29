import subprocess
import random

def create_kwok_deployment(deployment_name, replicas=1, cpu_request="250m", memory_request="512Mi", cpu_limit="500m", memory_limit="1Gi"):
    deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
  namespace: default
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {deployment_name}
  template:
    metadata:
      labels:
        app: {deployment_name}
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: type
                operator: In
                values:
                - kwok
              - key: kubernetes.io/role
                operator: In
                values:
                - agent
      tolerations:
      - key: "kwok.x-k8s.io/node"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: {deployment_name}-container
        image: busybox
        resources:
          requests:
            cpu: {cpu_request}
            memory: {memory_request}
          limits:
            cpu: {cpu_limit}
            memory: {memory_limit}
        command:
        - sleep
        - "3600"
"""
    subprocess.run(["kubectl", "apply", "-f", "-"], input=deployment_yaml, text=True)


def generate_funny_name():
    # Lists of adjectives and nouns for generating random funny names
    adjectives = ["funky", "jazzy", "silly", "wacky", "zany", "quirky", "goofy", "wonky", "spunky", "loony"]
    nouns = ["penguin", "llama", "unicorn", "gnome", "kitten", "chimp", "octopus", "dino", "robot", "alien"]
    # Generate a random funny name by combining a random adjective and noun
    return random.choice(adjectives) + '-' + random.choice(nouns)

def create_random_kwok_deployment():
    default_name = generate_funny_name()
    deployment_name = input(f"Enter the deployment name (default: '{default_name}'): ") or default_name
    replicas = int(input("Enter the number of replicas (default: 1): ") or 1)
    cpu_request = input("Enter CPU request (default: 250m for 0.25 CPU): ") or "250m"
    memory_request = input("Enter memory request (default: 512Mi): ") or "512Mi"
    cpu_limit = input("Enter CPU limit (default: 500m for 0.5 CPU): ") or "500m"
    memory_limit = input("Enter memory limit (default: 1Gi): ") or "1Gi"

    create_kwok_deployment(deployment_name, replicas, cpu_request, memory_request, cpu_limit, memory_limit)

if __name__ == "__main__":
    create_random_kwok_deployment()

