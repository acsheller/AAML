import pandas as pd
import sys
import io
import time
import os
import numpy as np
import sqlite3
sys.path.append('./scripts/')
from threading import Thread
from actorcritic_dqn import ActorCriticDQN
from workload_deployment_simulator import WorkloadDeploymentSimulator
from kinfo import KubeInfo
from kubernetes import client, config
from kubernetes.client.rest import ApiException



import optuna
import logging

# Set the logging level to DEBUG to get detailed information
logging.basicConfig(level=logging.DEBUG)
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
    model.run()


def objective(trial):
    global names_used
    # 1. Select Hyperparameters
    hidden_layers = trial.suggest_categorical('hidden_layers',[64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.9, 0.99)
    actor_learning_rate = trial.suggest_float('actor_learning_rate', 1e-4, 1e-2,log=True)
    
    critic_learning_rate = trial.suggest_float('critic_learning_rate', 1e-4, 1e-2,log=True)
    
    epochs = trial.suggest_categorical('epochs', [2, 3])

    # 2. Create Agent
    dqn = ActorCriticDQN(hidden_layers=hidden_layers, gamma=gamma, actor_learning_rate=actor_learning_rate,
                             critic_learning_rate=critic_learning_rate,
                             progress_indication=False,tensorboard_name=generate_funny_name()+'+dqn')

    # 3. Agent Run
    agent_thread = Thread(target=run_agent, args=(dqn,))
    agent_thread.start()

    # 4. Simulator Run
    simulator = WorkloadDeploymentSimulator(cpu_load=0.10, mem_load=0.50, pod_load=0.50, scheduler='custom-scheduler',progress_indication=False,epochs=epochs)
    simulator_thread = Thread(target=run_simulator, args=(simulator,))
    simulator_thread.start()

    # 5.  Wait for threads to complete
    agent_thread.join(timeout=120)
    simulator_thread.join(timeout=120)

    # 6. Remove the shutdown signal so everyone can start back up.
    #os.remove('./shutdown_signal.txt')

    # 7. Calculate the reward 
    state = k_inf.get_nodes_data(sort_by_cpu=False,include_controller=False)
    cpu_info = []
    for node in state:
        cpu_info.append(np.round(node['total_cpu_used']/node['cpu_capacity'],4))

    print(f"CPU INFO is {cpu_info}")
    # Calculate the variance of the cpu usage
    variance = np.var(cpu_info)
    print(f'variance is {variance}')

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
    name = name + '_dqn'
    study = optuna.create_study(direction='minimize',study_name=name,storage=storage_url)
    study.optimize(objective, n_trials=10,timeout=300)

    purne