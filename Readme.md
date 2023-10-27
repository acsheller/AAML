# Deep Reinforcement Learning using Attention Graph Neural Networks for Kubernetes Scheduling.

## Overview

Resource Management in Kubernetes clusters can be challenging. The default scheduler, kube-scheduler, may not meet all requirements for some applications. Fortunately, one can extend the functionality of the default scheduler to meet specific demands [1].  

This project uses [KWOK](https://kwok.sigs.k8s.io/) to establish a cluster of potentially thousands of nodes.  For this project a thousand will not be used but a mix of resource types will be provided.  

Applications or workflows that run in Kubernetes specify resources they need. during launch Kubernetes will check to see if a node (a physcial server) has the resource types the application needs.  Resource types include: CPU, memory, and storage. Additional resource that could be evaluated include GPU and networking resources.

## Getting Started

### System Setup

The system can be of any type. The particular system developed on is specified in the [setup instructions](docs/Setup_dev_env.md). what's needed is a Linux-based system running K8s.  The enviroment used in this work is WSL2 running Docker Desktop with Kubernetes enabled.

### Setup Python Virtual Environment

Access to the cluster can be obtained external to the cluster using a Python module called `kubernetes`. This way the Python script can be developed externally and then built into a Docker container to run as a deployment in Kubernetes. Anaconda is a good choice. Visit the Anaconda website for [instructions on installing, setting up, and using virtual environments](https://docs.anaconda.com/free/anaconda/install/index.html). More on thi after the cluster is setup.

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