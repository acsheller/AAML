# Deep Reinforcement Learning using Graph Neural Networks for Kubernetes Scheduling.





## Overview 

Instead of rehashing it all here please [Review the paper](link to paper). This repository supports. To summarize, the project use Docker compose to launch two containers. One container is a [Kwok](https://kwok.sigs.k8s.io/) simulated cluster, the other is the scheduler.  
 
## Getting Started
the  primary container called `aaml_kwok` needs to be built and stored localy. The container extends the latest PyTorch container which is already 3GB+ in size. Once built with PyTorch Geometric and all dependencies. The local container is 8GB. After cloning the repository, perform a docker compose build, this may take 15 minutes but is much quicker on future efforts as images are cached locally. 

```
git clone git@github.com:acsheller/AAML.git

docker compose build   # Remember to build if changes are made

docker compose up      # To bring them up

docker compose down    # To take the docker containers down

```

Jupyter lab will be served on port 8888. It can be accessed with `http://127.0.0.1:8888/lab?token=AAML`  Note the token is `AAML`.

Tensorboard will be served on port 6006. It can be reached at `http://127.0.0.1:6006/`.

### Is your cluster setup?
Be sure to complete the steps in the [setup instructions]([setup instructions](docs/Setup_dev_env.md)).
The cluster I setup has 10 fake nodes:
```
(base) asheller: many_nodes$ python create_nodes.py 
Enter the total number of standard nodes to create: 10
node/kwok-std-node-1 created
node/kwok-std-node-2 created
node/kwok-std-node-3 created
node/kwok-std-node-4 created
node/kwok-std-node-5 created
node/kwok-std-node-6 created
node/kwok-std-node-7 created
node/kwok-std-node-8 created
node/kwok-std-node-9 created
node/kwok-std-node-10 created

(base) asheller: many_nodes$ kgn
NAME               STATUS   ROLES           AGE     VERSION
docker-desktop     Ready    control-plane   3h30m   v1.27.2
kwok-std-node-1    Ready    agent           6s      fake
kwok-std-node-10   Ready    agent           5s      fake
kwok-std-node-2    Ready    agent           6s      fake
kwok-std-node-3    Ready    agent           6s      fake
kwok-std-node-4    Ready    agent           6s      fake
kwok-std-node-5    Ready    agent           6s      fake
kwok-std-node-6    Ready    agent           5s      fake
kwok-std-node-7    Ready    agent           5s      fake
kwok-std-node-8    Ready    agent           5s      fake
kwok-std-node-9    Ready    agent           5s      fake

```


## External Scheduler

The scheduler will run and listen specificially for 


## References:
[1] “Configure Multiple Schedulers,” Kubernetes. https://kubernetes.io/docs/tasks/extend-kubernetes/configure-multiple-schedulers/ (accessed Sep. 06, 2023).V