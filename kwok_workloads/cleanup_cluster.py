import subprocess

def delete_all_deployments_and_pods(namespace):
    try:
        # Delete all deployments in the specified namespace
        subprocess.check_call(["kubectl", "delete", "deployments", "--all", "-n", namespace])
        
        # Delete all pods in the specified namespace
        subprocess.check_call(["kubectl", "delete", "pods", "--all", "-n", namespace])
        
        print(f"All deployments and pods in the '{namespace}' namespace have been deleted.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    namespace = input("Enter the namespace to clean up: ")
    delete_all_deployments_and_pods(namespace)
