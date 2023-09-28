import subprocess

def create_kwok_gpu_node(node_count, gpus_per_node):
    for i in range(1, node_count + 1):
        node_yaml = f"""
apiVersion: v1
kind: Node
metadata:
  annotations:
    node.alpha.kubernetes.io/ttl: "0"
    kwok.x-k8s.io/node: fake
  labels:
    beta.kubernetes.io/arch: amd64
    beta.kubernetes.io/os: linux
    kubernetes.io/arch: amd64
    kubernetes.io/hostname: kwok-gpu-node-{i}
    kubernetes.io/os: linux
    kubernetes.io/role: gpu-agent
    node-role.kubernetes.io/gpu-agent: ""
    type: kwok
  name: kwok-gpu-node-{i}
spec:
  taints:
  - effect: NoSchedule
    key: kwok.x-k8s.io/node
    value: fake
status:
  allocatable:
    cpu: "32"
    memory: "256Gi"
    nvidia.com/gpu: "{gpus_per_node}"
    pods: "110"
  capacity:
    cpu: "32"
    memory: "256Gi"
    nvidia.com/gpu: "{gpus_per_node}"
    pods: "110"
  nodeInfo:
    architecture: amd64
    bootID: ""
    containerRuntimeVersion: ""
    kernelVersion: ""
    kubeProxyVersion: fake
    kubeletVersion: fake
    machineID: ""
    operatingSystem: linux
    osImage: ""
    systemUUID: ""
  phase: Running
"""
        subprocess.run(["kubectl", "apply", "-f", "-"], input=node_yaml, text=True)

if __name__ == "__main__":
    node_count = int(input("Enter the total number of GPU nodes to create: "))
    gpus_per_node = int(input("Enter the number of GPUs per GPU node: "))
    create_kwok_gpu_node(node_count, gpus_per_node)
