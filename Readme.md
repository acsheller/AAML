# Deep Reinforcement Learning using Attention Graph Neural Networks for Kubernetes Scheduling.

## Overview

Resource Management in Kubernetes clusters can be challenging. The default scheduler, kube-scheduler, may not meet all requirements for some applications. Fortunately, one can extend the functionality of the default scheduler to meet specific demands [1].  

This project uses [KWOK](https://kwok.sigs.k8s.io/) to establish a cluster of potentially thousands of nodes.  For this project a thousand will not be used but a mix of resource types will be provided.  

Applications or workflows that run in Kubernetes specify resources they need. during launch Kubernetes will check to see if a node (a physcial server) has the resource types the application needs.  Resource types include: CPU, memory, and storage. Additional resource that could be evaluated include GPU and networking resources.

## System Setup

The system can be of any type. The particular system developed on is specified in the [setup instructions](docs/Setup.md). what's needed is a Linux-based system running K8s.  The enviroment used in this work is WSL2 running Docker Desktop with Kubernetes enabled.

## Setup Python Virtual Environment (Optional but Recommended)
This is not necessary but it helps with development. Why?  Access to the cluster can be obtained external to the cluster using a Python module called `kubernetes`. This way the Python script can be developed externally and then built into a Docker container to run as a deployment in Kubernetes. Anaconda is a good choice. Visit the Anaconda website for [instructions on installing, setting up, and using virtual environments](). 

## Setup Kwok

Kwok was deployed to the single node cluster specified earlier. 


## References:
