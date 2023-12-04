import subprocess
import argparse
import time

def create_kwok_node(node_count=10,cpu_count = 64,memory="256Gi",pod_limit=110,node_type='agent'):
    base_name = 'kwok-std-node'
    if node_type =='control-plane':
      base_name = 'kwok-ctl-node'
    for i in range(0, node_count):
        node_name = f"{base_name}-0"
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
    kubernetes.io/hostname: {node_name}
    kubernetes.io/os: linux
    kubernetes.io/role: {node_type}
    #node-role.kubernetes.io/{base_name}: ""
    type: kwok
  name: {base_name}-{i}
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

        if node_type == "control-plane":
          taint_node(node_name)


def taint_node(node_name):
    taint = "donotschedule=true:NoSchedule"
    print("Running taint node")
    subprocess.run(["kubectl", "taint", "nodes", node_name, taint])


if __name__ == "__main__":
    #node_count = int(input("Enter the total number of standard nodes to create: "))
     # Create the parser
    parser = argparse.ArgumentParser(description='Create Kwok nodes')

    # Add arguments
    parser.add_argument('--node_count', type=int, default=10, help='Number of nodes to create')
 
    args = parser.parse_args()

    time.sleep(4)
    create_kwok_node(node_count=1,node_type='control-plane')
    create_kwok_node(node_count=args.node_count)

