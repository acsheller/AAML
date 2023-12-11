# A Comparative Analysis of Actor-Critic and Deep Q Network Methods Using Graph Neural Networks and Multi-Layer Perceptrons for Efficient Scheduling in Kubernetes Clusters


## Overview 

The project use Docker compose to launch two containers. One container is a [Kwok](https://kwok.sigs.k8s.io/) simulated cluster, the other is the scheduler. Various DRL algorithms can be used as scheduling agents. 
 
## Docker and Docker Compose
Docker and Docker compose both need to be installed. The recommended version of Docker Compose is `docker-compose-v2`.  This lets one:

```
docker compose down
docker compose build
docker compose up
```  

perform the build of the  containers in the above format.  

`docker compose build` on the first time around takes time.  It will build the container, install software, and add the mounnts to the directories discussed below.


## Useful commands
There are many aliases, or shortcuts that have been provided that need to be run from the base of the project in the deployed container. These are:

- `runsim` - Start a simulation running. This calls the script /home/appuser/scripts/workload_deployment_simulator.py`
- `rundqm` - Start the DQN Scheduler based on artificial neural network. This is the Python script `/home/appuser/scripts/scheduler_dqn.py`.
- `rungnn` - Start a GNN Scheduler based on Graph Neural Network node types.  This is the Python script `/home/appuser/scripts/scheduler_gnn.py`
- `runacdqn` - Run the Actor Critic DQN scheduling agent. This is the Python script `/home/appuser/scripts/actorcritic_dqn.py`.
- `runacgnn` - Run the Actor Critic GNN based scheduler. Thisis the Python script `/home/appuser/scripts/actorcritic_gnn.py`.
- `cleancluster` - Will reset the cluster back to a state where all nodes have 0 pods deployed.
- `ktop` - This is `watch python /home/appuser/scripts/kinfo.py`.  Just Ctrol-C to exit out of this. It shows the distribution of pods across the fake cluster.

For a listing of aliases used as shortcuts simply type `aliases` at a command prompt.

Aliases are also provided to shorten kubectl commands such as:
- `kgp` -- kubectl get pods
- `kgn` -- kubectl get nodes

## Getting Started
the  primary container called `aaml_kwok` needs to be built and stored localy. The container extends the latest PyTorch container which is already 3GB+ in size. Once built with PyTorch Geometric and all dependencies, the local container is 8GB. After cloning the repository, perform a docker compose build, this may take 15 minutes but is much quicker on future efforts as images are cached locally. 


```
git clone git@github.com:acsheller/AAML.git

docker compose build   # Remember to build if changes are made

docker compose up      # To bring them up

docker compose down    # To take the docker containers down

```

Jupyter lab will be served on port 8888. It can be accessed with `http://127.0.0.1:8888/lab?token=AAML`  Note the token is `AAML`. 
The terminal that Jupyter Lab launches is configured to run all scripts in this project.

Tensorboard will be served on port 6006. It can be reached at `http://127.0.0.1:6006/`. 

### Folder Structure
These folders are mounted into the running container. They can be accessed either ono the local filesystem or from within the Jupyter Lab notebook.

- `tlogs` The data for Tensorboard is stored in the `tlogs` folder.
- `logs` Logging is performed extensively; data is  stoed in the folder `logs` in chronological order.
- `data` This data is used to prepopulate the replay buffers of DQN agents.
- `models` Models are stored here after a run is completed
- `deployment_data` The simulator outputs this file which are the list of deployments sent to Kubernetes simulator.
- `docker` Contains the docker file that sets up the AAML container.
- `notebooks` a Jupyter Lab notebook with some information is provided.  New notebooks can be created to take advantage of the PyTorch environment.
- `scripts` Contains all the Python scripts used in this project.

Only one scheduler and simulator can be run at a time.  All commands must be run from the base of the project.

### Getting Help
Help is built into most of the models because they can be configured at the command line. The below was performed by opening up a terminal in Jupyter Lab.

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

### Getting Started
After cloning the repository and ensuring both Docker and Docker Compose are installed, the `Docker compose build` will take serveral minutes initially to pull the latest PyTorch container and expand on it.  If building again, this delay will be much shorter. 

A `docker compose down` will look like this:
```
(AAML) asheller: AAML$ pwd
/mnt/e/onedrive/__EN.705.742.AAML/FinalProject/AAML
(AAML) asheller: AAML$ docker compose down
[+] Running 3/3
 ✔ Container aaml-pytorch-container-1  Removed                                                           1.8s
 ✔ Container aaml-kwok-cluster-1       Removed                                                          10.5s
 ✔ Network aaml_my-network             Removed                                                           0.2s
(AAML) asheller: AAML$
```

A `docker compose build` will look like  the below. This can take varying amounts but usually only a few seconds after the first run.

```
(AAML) asheller: AAML$ docker compose build
[+] Building 0.8s (30/30) FINISHED                                                             docker:default
 => [pytorch-container internal] load build definition from Dockerfile                                   0.0s
 => => transferring dockerfile: 3.69kB                                                                   0.0s
 => [pytorch-container internal] load .dockerignore                                                      0.0s
 => => transferring context: 2B                                                                          0.0s
 => [pytorch-container internal] load metadata for docker.io/pytorch/pytorch:latest                      0.6s
 => [pytorch-container auth] pytorch/pytorch:pull token for registry-1.docker.io                         0.0s
 => [pytorch-container  1/24] FROM docker.io/pytorch/pytorch:latest@sha256:b32443a58a60c4ca4d10651e3c5b  0.0s
 => [pytorch-container internal] load build context                                                      0.0s
 => => transferring context: 144B                                                                        0.0s
 => CACHED [pytorch-container  2/24] RUN apt-get update                                                  0.0s
 => CACHED [pytorch-container  3/24] RUN apt-get install -y curl vim                                     0.0s
 => CACHED [pytorch-container  4/24] RUN pip3 install jupyterlab kubernetes networkx torch-geometric te  0.0s
 => CACHED [pytorch-container  5/24] RUN curl -L -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$  0.0s
 => CACHED [pytorch-container  6/24] RUN chmod +x /usr/local/bin/kubectl                                 0.0s
 => CACHED [pytorch-container  7/24] RUN useradd -m -d /home/appuser -s /bin/bash appuser                0.0s
 => CACHED [pytorch-container  8/24] RUN chown -R appuser:appuser /home/appuser                          0.0s
 => CACHED [pytorch-container  9/24] RUN mkdir /home/appuser/.kube                                       0.0s
 => CACHED [pytorch-container 10/24] COPY docker/config /home/appuser/.kube/config                       0.0s
 => CACHED [pytorch-container 11/24] RUN mkdir /home/appuser/tlogs                                       0.0s
 => CACHED [pytorch-container 12/24] RUN mkdir /home/appuser/logs                                        0.0s
 => CACHED [pytorch-container 13/24] RUN mkdir /home/appuser/models                                      0.0s
 => CACHED [pytorch-container 14/24] RUN mkdir /home/appuser/deployment_data                             0.0s
 => CACHED [pytorch-container 15/24] RUN mkdir /home/appuser/scripts                                     0.0s
 => CACHED [pytorch-container 16/24] RUN mkdir /home/appuser/data                                        0.0s
 => CACHED [pytorch-container 17/24] COPY docker/entrypoint.sh /home/appuser/entrypoint.sh               0.0s
 => CACHED [pytorch-container 18/24] COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.co  0.0s
 => CACHED [pytorch-container 19/24] WORKDIR /home/appuser/                                              0.0s
 => CACHED [pytorch-container 20/24] RUN echo "alias k='kubectl'" >> .bashrc &&     echo "alias kdp='ku  0.0s
 => CACHED [pytorch-container 21/24] RUN echo "if [ -f ~/.bashrc ]; then source ~/.bashrc; fi" >> .bash  0.0s
 => CACHED [pytorch-container 22/24] RUN ["chmod", "+x", "/home/appuser/entrypoint.sh"]                  0.0s
 => CACHED [pytorch-container 23/24] RUN  ["chmod", "+r", "/etc/supervisor/conf.d/supervisord.conf"]     0.0s
 => CACHED [pytorch-container 24/24] RUN chown -R appuser:appuser /home/appuser                          0.0s
 => [pytorch-container] exporting to image                                                               0.0s
 => => exporting layers                                                                                  0.0s
 => => writing image sha256:19d9c0a57bfa14a7b6eedd8ac9055ca42190ff17297ce3486b85aed3ac841ede             0.0s
 => => naming to docker.io/library/aaml_kwok:latest                                                      0.0s
(AAML) asheller: AAML$

```

A `docker compose up` looks like the below.  It will lock the terminal unless you use `--detach` which will start the containers in the background and keep them running. 

```
(AAML) asheller: AAML$ docker compose up
[+] Building 0.0s (0/0)                                                                        docker:default
[+] Running 3/3
 ✔ Network aaml_my-network             Created                                                           0.0s
 ✔ Container aaml-kwok-cluster-1       Created                                                           0.0s
 ✔ Container aaml-pytorch-container-1  Created                                                           0.1s
Attaching to aaml-kwok-cluster-1, aaml-pytorch-container-1
aaml-kwok-cluster-1       | {"time":"2023-12-11T02:19:04.769972828Z","level":"INFO","source":{"function":"sigs.k8s.io/kwok/pkg/kwokctl/cmd/create/cluster.runE","file":"/workspace/pkg/kwokctl/cmd/create/cluster/cluster.go","line":238},"msg":"Cluster is creating","cluster":"kwok"}
aaml-kwok-cluster-1       | {"time":"2023-12-11T02:19:05.750511906Z","level":"INFO","source":{"function":"sigs.k8s.io/kwok/pkg/kwokctl/cmd/create/cluster.runE","file":"/workspace/pkg/kwokctl/cmd/create/cluster/cluster.go","line":267},"msg":"Cluster is created","cluster":"kwok","elapsed":{"nanosecond":980612208,"human":"980.612208ms"}}
aaml-kwok-cluster-1       | {"time":"2023-12-11T02:19:05.750584676Z","level":"INFO","source":{"function":"sigs.k8s.io/kwok/pkg/kwokctl/cmd/create/cluster.runE","file":"/workspace/pkg/kwokctl/cmd/create/cluster/cluster.go","line":273},"msg":"Cluster is starting","cluster":"kwok"}
aaml-kwok-cluster-1       | {"time":"2023-12-11T02:19:07.412108113Z","level":"INFO","source":{"function":"sigs.k8s.io/kwok/pkg/kwokctl/cmd/create/cluster.runE","file":"/workspace/pkg/kwokctl/cmd/create/cluster/cluster.go","line":278},"msg":"Cluster is started","cluster":"kwok","elapsed":{"nanosecond":1661486499,"human":"1.661486499s"}}
aaml-kwok-cluster-1       | Starting to serve on [::]:8080
aaml-kwok-cluster-1       | ###############################################################################
aaml-kwok-cluster-1       | > kubectl -s :8080 version
aaml-kwok-cluster-1       | Client Version: v1.28.0
aaml-kwok-cluster-1       | Kustomize Version: v5.0.4-0.20230601165947-6ce0bf390ce3
aaml-kwok-cluster-1       | Server Version: v1.28.0
aaml-kwok-cluster-1       | ###############################################################################
aaml-kwok-cluster-1       | # The following kubeconfig can be used to connect to the Kubernetes API server
aaml-kwok-cluster-1       | apiVersion: v1
aaml-kwok-cluster-1       | clusters:
aaml-kwok-cluster-1       | - cluster:
aaml-kwok-cluster-1       |     server: http://127.0.0.1:8080
aaml-kwok-cluster-1       |   name: kwok
aaml-kwok-cluster-1       | contexts:
aaml-kwok-cluster-1       | - context:
aaml-kwok-cluster-1       |     cluster: kwok
aaml-kwok-cluster-1       |   name: kwok
aaml-kwok-cluster-1       | current-context: kwok
aaml-kwok-cluster-1       | kind: Config
aaml-kwok-cluster-1       | preferences: {}
aaml-kwok-cluster-1       | users: null
aaml-kwok-cluster-1       | ###############################################################################
aaml-kwok-cluster-1       | > kubectl -s :8080 get ns
aaml-kwok-cluster-1       | NAME              STATUS   AGE
aaml-kwok-cluster-1       | default           Active   1s
aaml-kwok-cluster-1       | kube-node-lease   Active   1s
aaml-kwok-cluster-1       | kube-public       Active   1s
aaml-kwok-cluster-1       | kube-system       Active   1s
aaml-kwok-cluster-1       | ###############################################################################
aaml-kwok-cluster-1       | # The above example works if your host's port is the same as the container's,
aaml-kwok-cluster-1       | # otherwise, change it to your host's port
aaml-pytorch-container-1  | node/kwok-ctl-node-0 created
aaml-pytorch-container-1  | node/kwok-ctl-node-0 tainted
aaml-pytorch-container-1  | node/kwok-std-node-0 created
aaml-pytorch-container-1  | node/kwok-std-node-1 created
aaml-pytorch-container-1  | node/kwok-std-node-2 created
aaml-pytorch-container-1  | node/kwok-std-node-3 created
aaml-pytorch-container-1  | node/kwok-std-node-4 created
aaml-pytorch-container-1  | node/kwok-std-node-5 created
aaml-pytorch-container-1  | node/kwok-std-node-6 created
aaml-pytorch-container-1  | node/kwok-std-node-7 created
aaml-pytorch-container-1  | node/kwok-std-node-8 created
aaml-pytorch-container-1  | node/kwok-std-node-9 created
aaml-pytorch-container-1  | Running taint node
aaml-pytorch-container-1  | 2023-12-11 02:19:11,607 INFO supervisord started with pid 1
aaml-pytorch-container-1  | 2023-12-11 02:19:12,609 INFO spawned: 'jupyterlab' with pid 313
aaml-pytorch-container-1  | 2023-12-11 02:19:12,611 INFO spawned: 'tensorboard' with pid 314
aaml-pytorch-container-1  | 2023-12-11 02:19:13,698 INFO success: jupyterlab entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
aaml-pytorch-container-1  | 2023-12-11 02:19:13,698 INFO success: tensorboard entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
```


### Questions
for help or quesitons please contact Anthony sheller ashelle5@jhu.edu