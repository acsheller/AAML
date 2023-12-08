# Deep Reinforcement Learning using Graph Neural Networks for Kubernetes Scheduling.





## Overview 

Instead of rehashing it all here please [Review the paper](link to paper). This repository supports. To summarize, the project use Docker compose to launch two containers. One container is a [Kwok](https://kwok.sigs.k8s.io/) simulated cluster, the other is the scheduler.  
 
## Docker and Docker Compose
Docker and Docker compose both need to be installed. The recommended version of Docker Compose is `docker-compose-v2`.  This lets one:

```
docker compose down
docker compose build
docker compose up
```  

perform the build of the  containers in the above format.  


## Useful commands
There are many aliases, or shortcuts that have been provided that need to be run from the base of the project in the deployed container. These are:

- `runsim` - Start a simulation running. This calls the script /home/appuser/scripts/workload_deployment_simulator.py`
- `rundqm` - Start the DQN Scheduler based on artificial neural network. Neural network node types used include dropout and batchnormalization for information.  This is the Python script `/home/appuser/scripts/scheduler_dqn.py`.
- `rungnn` - Start a GNN Scheduler based on Graph Neural Network node types.  This is the Python script `/home/appuser/scripts/scheduler_gnn.py`
- `runacdqn` - Run the Actor Critic DQN scheduling agent. This is the Python script `/home/appuser/scripts/actorcritic_dqn.py`.
- `runacgnn` - Run the Actor Critic GNN based scheduler. Thisis the Python script `/home/appuser/scripts/actorcritic_gnn.py`.
- `cleancluster` - Will reset the cluster back to a state where all nodes have 0 pods deployed.
- `ktop` - This is `watch python /home/appuser/scripts/kinfo.py`.  Just Ctrol-C to exit out of this. It shows the distribution of pods across the fake cluster.

For a listing of aliases used as shortcuts simply type `aliases` at a command prompt.

## Getting Started
the  primary container called `aaml_kwok` needs to be built and stored localy. The container extends the latest PyTorch container which is already 3GB+ in size. Once built with PyTorch Geometric and all dependencies, the local container is 8GB. After cloning the repository, perform a docker compose build, this may take 15 minutes but is much quicker on future efforts as images are cached locally. 


```
git clone git@github.com:acsheller/AAML.git

docker compose build   # Remember to build if changes are made

docker compose up      # To bring them up

docker compose down    # To take the docker containers down

```

Jupyter lab will be served on port 8888. It can be accessed with `http://127.0.0.1:8888/lab?token=AAML`  Note the token is `AAML`.

Tensorboard will be served on port 6006. It can be reached at `http://127.0.0.1:6006/`. The data for Tensorboard is stored in the `tlogs` folder.

Logging is performed extensively; data is  stoed in the folder `logs` in chronological order.

No data is removed from the container. one can mount in a directory and copy data over.  

After a completed run, the model is saved to the `/home/appuser` folder.  It can be reloaded.  The run data is also saved and can be correlated with logging.   

Only one scheduler and simulator can be run at a time.  

## Getting Help
Help is  built into most of the models because they can be configured at the command line. The below was performed by opening up a terminal in Jupyter Lab.

```
appuser@e10ec6e981f6:~$ rundqn -h
usage: scheduler_dqn.py [-h] [--hidden_layers HIDDEN_LAYERS] [--init_epsi INIT_EPSI] [--epsi_decay EPSI_DECAY] [--gamma GAMMA] [--learning_rate LEARNING_RATE] [--replay_buffer_size REPLAY_BUFFER_SIZE]
                        [--update_frequency UPDATE_FREQUENCY] [--target_update_frequency TARGET_UPDATE_FREQUENCY] [--batch_size BATCH_SIZE] [--progress] [--log_scrolling]

rundqn is an alias to scheduler_dqn.py. It is used for running the DQN Scheduler with various configurations.

options:
  -h, --help            show this help message and exit
  --hidden_layers HIDDEN_LAYERS
                        Number of Hidden Layers (default: 32)
  --init_epsi INIT_EPSI
                        Initial Epsilon Starting Value (default: 1.0)
  --epsi_decay EPSI_DECAY
                        Epsilon Decay Rate (default: 0.995)
  --gamma GAMMA         Discount Factor (default: 0.99)
  --learning_rate LEARNING_RATE
                        Learning Rate (default: 0.001)
  --replay_buffer_size REPLAY_BUFFER_SIZE
                        Length of the Replay Buffer (default: 2000)
  --update_frequency UPDATE_FREQUENCY
                        Network Update Frequency (default: 25)
  --target_update_frequency TARGET_UPDATE_FREQUENCY
                        Target Network Update Frequency (default: 50)
  --batch_size BATCH_SIZE
                        Batch Size of replay Buffer sample to pass during training (default: 200)
  --progress            Enable progress indication. Only when logs are not scrolling (default: False)
  --log_scrolling       Enable Log Scrolling to Screen. Disables progress Indication (default: False)
appuser@e10ec6e981f6:~$ 

```



## References:
[1] “Configure Multiple Schedulers,” Kubernetes. https://kubernetes.io/docs/tasks/extend-kubernetes/configure-multiple-schedulers/ (accessed Sep. 06, 2023).V