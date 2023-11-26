import pandas as pd
import sys
import io
import time
import os
import numpy as np
import sqlite3

from threading import Thread
sys.path.append('./scheduler/')
from scheduler_dqn import CustomSchedulerDQN
sys.path.append('./kwok_workloads/')
from workload_deployment_simulator import WorkloadDeploymentSimulator
from kinfo import KubeInfo
from kubernetes import client, config
from kubernetes.client.rest import ApiException



import optuna

names_used = []
 
k_inf = KubeInfo()

def contains_non_alpha_or_dash(s):
    '''
    This is necessary because some of the names have funny characters in them. 
    '''
    for char in s:
        if not char.isalpha() and char != '-':
            return True
    return False


def generate_funny_name():
    '''
    I cannot recall where I saw this done but I like it so crafted this version of it.
    '''
    import randomname
    while True:
        name = randomname.get_name().lower()
        if not contains_non_alpha_or_dash(name):
            if name not in names_used:
                names_used.append(name)
                return name


def delete_all_deployments_and_pods(namespace):
    # Load the kube config
    config.load_kube_config()

    # Setup API instances
    api_instance = client.AppsV1Api()
    core_v1_api = client.CoreV1Api()

    try:
        # Delete all deployments in the specified namespace
        api_instance.delete_collection_namespaced_deployment(namespace=namespace)
        #print(f"All deployments in the '{namespace}' namespace have been deleted.")

        # Delete all pods in the specified namespace
        core_v1_api.delete_collection_namespaced_pod(namespace=namespace)
        #print(f"All pods in the '{namespace}' namespace have been deleted.")

    except ApiException as e:
        print(f"Kubernetes API exception occurred: {e.reason}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def run_agent(model):
    model.run()

def run_simulator(model):
    model.run(interval=10, duration=1, epochs=1)


def objective(trial):
    global names_used
    # 1. Select Hyperparameters
    init_epsi = trial.suggest_float('init_epsi', 0.8, 1.0)
    gamma = trial.suggest_float('gamma', 0.8, 0.99)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2,log=True)
    epsi_decay = trial.suggest_float('epsi_decay', 0.99, 0.9999)
    replay_buffer_size = trial.suggest_categorical('replay_buffer_size', [50, 100, 200])
    batch_size = trial.suggest_categorical('batch_size', [20, 32, 64])
    target_update_frequency = trial.suggest_int('target_update_frequency', 10, 100)

    # 2. Create Agent
    dqn = CustomSchedulerDQN(init_epsi=init_epsi, gamma=gamma, learning_rate=learning_rate,
                             epsi_decay=epsi_decay, replay_buffer_size=replay_buffer_size,
                             batch_size=batch_size, target_update_frequency=target_update_frequency,progress_indication=False,tensorboard_name=names_used[-1])

    # 3. Agent Run
    agent_thread = Thread(target=run_agent, args=(dqn,))
    agent_thread.start()

    # 4. Simulator Run
    simulator = WorkloadDeploymentSimulator(cpu_load=0.10, mem_load=0.50, pod_load=0.50, scheduler='custom-scheduler',progress_indication=False)
    simulator_thread = Thread(target=run_simulator, args=(simulator,))
    simulator_thread.start()

    # 5.  Wait for threads to complete
    agent_thread.join()
    simulator_thread.join()

    # 6. Remove the shutdown signal so everyone can start back up.
    os.remove('./shutdown_signal.txt')

    # 7. Calculate the reward 
    state = k_inf.get_nodes_data(sort_by_cpu=False,include_controller=False)
    cpu_info = []
    for node in state:
        cpu_info.append(np.round(node['total_cpu_used']/node['cpu_capacity'],4))

    # Calculate the variance of the cpu usage
    variance = np.var(cpu_info)

    # 8. Clear cluster
    delete_all_deployments_and_pods('default')

    # Sleep some to let the cluster settledown.
    time.sleep(10)
    # Return the optimization metric (e.g., total reward, loss)
    # Optuna will minimize this.
    return variance

if __name__ == "__main__":
    storage_url = "sqlite:///optuna.db"
    delete_all_deployments_and_pods('default')
    time.sleep(5)
    name = generate_funny_name()
    study = optuna.create_study(direction='minimize',study_name=name+'dqn',storage=storage_url)
    study.optimize(objective, n_trials=50)