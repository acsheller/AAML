import subprocess

def create_kwok_node(node_count=10,cpu_count = 32,memory="256Gi"pod_limit=110):
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
    kubernetes.io/hostname: kwok-std-node-{i}
    kubernetes.io/os: linux
    kubernetes.io/role: agent
    node-role.kubernetes.io/agent: ""
    type: kwok
  name: kwok-std-node-{i}
status:
  allocatable:
    cpu: "{cpu_count}"
    memory: {memory}
    pods: "{pod_limit}"
  capacity:
    cpu: "{cpu_count}"
    memory: {memory}
    pods: "{pod_limit}"
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
    node_count = int(input("Enter the total number of standard nodes to create: "))
    create_kwok_node(node_count)
