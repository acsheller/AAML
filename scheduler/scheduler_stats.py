import pandas as pd
from datetime import datetime
from kubernetes import client, config, watch
import logging
from kinfo import KubeInfo
import numpy as np
import signal
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SchedulerStatisticsLogger:
    def __init__(self, scheduler_type='default-scheduler',namespace='default'):
        self.namespace = namespace
        config.load_kube_config()  # Load kube config from .kube/config
        self.api = client.CoreV1Api()
        self.watcher = watch.Watch()
        self.kube_info = KubeInfo()
        self.scheduler_type = scheduler_type
        self.stats_df = pd.DataFrame(columns=['timestamp', 'pod_name', 'cpu_score', 'mem_score', 'pod_score'])

        logging.info("STATS :: Scheduler Statistics Logger Started")



    def calculate_balance_reward(self,node_values, max_value_per_node=0.80):
        '''
        Calculates the level of "balance" by taking pairwise different and summing
        

        Reused from the Environment
        '''
        values = list(node_values.values())
        num_nodes = len(values)
        
        # Calculate all pairwise absolute differences
        pairwise_differences = sum(abs(values[i] - values[j]) for i in range(num_nodes) for j in range(i + 1, num_nodes))
        
        # The maximum possible sum of differences occurs when one node is at max value and all others are at zero
        # NOTE -- for future, the -1 is for the one controller- there may be more so this can be an argument or a feature.
        max_possible_difference = (num_nodes - 1) * max_value_per_node
        
        # Normalize the sum of differences
        normalized_difference = pairwise_differences / max_possible_difference
        
        # Invert the result so that a higher value means more balance <--- TODO Verify and Double check this!
        balance_score = 1 - normalized_difference
        
        # Scale the balance score to determine the final reward
        base_reward = 1.0  # Assume a base reward of 1.0 for maximum balance
        reward = balance_score * base_reward
        return reward


    def log_scheduler_stats(self,pod_name):
        # 1. Get the state of the cluster
        node_info = self.kube_info.get_nodes_data()
        # 2. Get the CPU and Memory info for each node
        cpu_info = {}
        memory_info = {}
        pod_info = {}
        for node in node_info:
            if 'control-plane' not in node['roles']:
                cpu_info[node['name']] = np.round(node['total_cpu_used'] / node['cpu_capacity'],3)
                memory_info[node['name']] = np.round(node['total_memory_used'] / node['memory_capacity'],3)
                pod_info[node['name']] = np.round(node['pod_count']/node['pod_limit'],3)
        # 3. Calculate balance score for CPU and Memory
        cpu_balance_score = np.round(self.calculate_balance_reward(cpu_info),3)
        memory_balance_score = np.round(self.calculate_balance_reward(memory_info),3)
        pod_info_score = np.round(self.calculate_balance_reward(pod_info),3)
        
        # 4. Log the statistics
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_stats = {
            'timestamp': current_time,
            'pod_name': pod_name,
            'cpu_score': cpu_balance_score,
            'mem_score': memory_balance_score,  # Convert dict to string to store in CSV
            'pod_score': pod_info_score  # Convert dict to string to store in CSV
        }
        new_stats_df = pd.DataFrame([new_stats])
        self.stats_df = pd.concat([self.stats_df, new_stats_df], ignore_index=True)

    def save_statistics(self):
        # Save the dataframe to a CSV file
        filename = f"scheduler_statistics_{self.scheduler_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.stats_df.to_csv(filename, index=False)
        logging.info(f"Statistics saved to {filename}")

    def should_shutdown(self):
        '''
        So everything can shutdown about the same time
        '''
        logging.info("AGENT :: checking shutdown status")
        return os.path.exists("shutdown_signal.txt")



    def run(self):
        while not self.should_shutdown():
            try:
                #for event in self.watcher.stream(self.api.list_namespaced_pod, namespace=self.namespace,timeout_seconds=30):
                for event in self.watcher.stream(self.api.list_pod_for_all_namespaces):
                    # Log statistics at a fixed interval or based on an event
                    if event['raw_object']['metadata']['namespace'] == self.namespace and \
                       event['type'] in ['ADDED', 'DELETED']:
                        logging.info(f"{event['type']}   {event['raw_object']['metadata']['name']}")
                        self.log_scheduler_stats(event['raw_object']['metadata']['name'])

            except client.exceptions.ApiException as e:
                if e.status == 410:
                    logging.warning("STATS :: Watch timed out or resourceVersion too old.")
                    break
                else:
                    logging.error(f"STATS :: Unexpected API exception: {e}")
            except KeyboardInterrupt:
                # Handle any other cleanup here
                logging.info("STATS :: Interrupted by user.")
                break
            except SystemExit:
                # Handle the cleanup when the system exits
                logging.info("STATS :: System exit detected.")
                break

        # This block will run after the loop exits
        self.save_statistics()  # Save the statistics one last time before exiting
        logging.info("STATS :: Final statistics saved.")


# Usage example
# You would need to define `kube_info` which should be an instance of a class that has a `get_nodes_data` method.
if __name__ == "__main__":
    scheduler_statistics_logger = SchedulerStatisticsLogger(scheduler_type='custom_scheduler')
    scheduler_statistics_logger.run()
