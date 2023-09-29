import subprocess

def create_kwok_gpu_deployment(deployment_name, replicas, cpu_request="250m", memory_request="512Mi", cpu_limit="500m", memory_limit="1Gi", gpu_limit=1):
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
                - gpu-agent
      tolerations:
      - key: "kwok.x-k8s.io/node"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: {deployment_name}-container
        image: nvidia/cuda:11.0-base
        resources:
          requests:
            cpu: {cpu_request}
            memory: {memory_request}
            nvidia.com/gpu: "{gpu_limit}"
          limits:
            cpu: {cpu_limit}
            memory: {memory_limit}
            nvidia.com/gpu: "{gpu_limit}"
        command:
        - sleep
        - "3600"
"""
    subprocess.run(["kubectl", "apply", "-f", "-"], input=deployment_yaml, text=True)

if __name__ == "__main__":
    deployment_name = input("Enter the deployment name: ")
    replicas = int(input("Enter the number of replicas: "))
    
    cpu_request = input(f"Enter CPU request (default is {cpu_request}): ")
    if not cpu_request:
        cpu_request = "250m"
    
    memory_request = input(f"Enter memory request (default is {memory_request}): ")
    if not memory_request:
        memory_request = "512Mi"
    
    cpu_limit = input(f"Enter CPU limit (default is {cpu_limit}): ")
    if not cpu_limit:
        cpu_limit = "500m"
    
    memory_limit = input(f"Enter memory limit (default is {memory_limit}): ")
    if not memory_limit:
        memory_limit = "1Gi"
    
    gpu_limit = input("Enter GPU limit (default is 1): ")
    if not gpu_limit:
        gpu_limit = 1

    create_kwok_gpu_deployment(deployment_name, replicas, cpu_request, memory_request, cpu_limit, memory_limit, gpu_limit)
