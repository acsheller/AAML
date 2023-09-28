import subprocess

def delete_kwok_gpu_node(node_count):
    for i in range(1, node_count + 1):
        node_name = f"kwok-gpu-node-{i}"
        subprocess.run(["kubectl", "delete", "node", node_name])

if __name__ == "__main__":
    node_count = int(input("Enter the total number of GPU nodes to delete: "))
    delete_kwok_gpu_node(node_count)
