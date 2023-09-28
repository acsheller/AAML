# build a mnay node cluster

This folder contains scripts for dynamically creating many clusters.
There is also a script for adding additional resources and simulating the allocation of those in the cluster. 

## The Python scripts can be launched as follows

```
(base) asheller: many_nodes$ python create_nodes.py 
Enter the total number of standard nodes to create: 20
node/kwok-standard-node-1 created
node/kwok-standard-node-2 created
node/kwok-standard-node-3 created
node/kwok-standard-node-4 created
node/kwok-standard-node-5 created
node/kwok-standard-node-6 created
node/kwok-standard-node-7 created
node/kwok-standard-node-8 created
node/kwok-standard-node-9 created
node/kwok-standard-node-10 created
node/kwok-standard-node-11 created
node/kwok-standard-node-12 created
node/kwok-standard-node-13 created
node/kwok-standard-node-14 created
node/kwok-standard-node-15 created
node/kwok-standard-node-16 created
node/kwok-standard-node-17 created
node/kwok-standard-node-18 created
node/kwok-standard-node-19 created
node/kwok-standard-node-20 created
(base) asheller: many_nodes$ kg nodes
NAME                    STATUS   ROLES           AGE   VERSION
docker-desktop          Ready    control-plane   55d   v1.27.2
kwok-standard-node-1    Ready    agent           6s    fake
kwok-standard-node-10   Ready    agent           4s    fake
kwok-standard-node-11   Ready    agent           4s    fake
kwok-standard-node-12   Ready    agent           4s    fake
kwok-standard-node-13   Ready    agent           4s    fake
kwok-standard-node-14   Ready    agent           3s    fake
kwok-standard-node-15   Ready    agent           3s    fake
kwok-standard-node-16   Ready    agent           3s    fake
kwok-standard-node-17   Ready    agent           3s    fake
kwok-standard-node-18   Ready    agent           2s    fake
kwok-standard-node-19   Ready    agent           2s    fake
kwok-standard-node-2    Ready    agent           6s    fake
kwok-standard-node-20   Ready    agent           2s    fake
kwok-standard-node-3    Ready    agent           6s    fake
kwok-standard-node-4    Ready    agent           6s    fake
kwok-standard-node-5    Ready    agent           5s    fake
kwok-standard-node-6    Ready    agent           5s    fake
kwok-standard-node-7    Ready    agent           5s    fake
kwok-standard-node-8    Ready    agent           5s    fake
kwok-standard-node-9    Ready    agent           5s    fake
(base) asheller: many_nodes$ 

```

## Similarly for GPU nodes 

```
(base) asheller: many_nodes$ python create_gpu_nodes.py 
Enter the total number of GPU nodes to create: 10
Enter the number of GPUs per GPU node: 7
node/kwok-gpu-node-1 created
node/kwok-gpu-node-2 created
node/kwok-gpu-node-3 created
node/kwok-gpu-node-4 created
node/kwok-gpu-node-5 created
node/kwok-gpu-node-6 created
node/kwok-gpu-node-7 created
node/kwok-gpu-node-8 created
node/kwok-gpu-node-9 created
node/kwok-gpu-node-10 created
(base) asheller: many_nodes$ kg nodes
NAME                    STATUS   ROLES           AGE   VERSION
docker-desktop          Ready    control-plane   55d   v1.27.2
kwok-gpu-node-1         Ready    gpu-agent       5s    fake
kwok-gpu-node-10        Ready    gpu-agent       3s    fake
kwok-gpu-node-2         Ready    gpu-agent       4s    fake
kwok-gpu-node-3         Ready    gpu-agent       4s    fake
kwok-gpu-node-4         Ready    gpu-agent       4s    fake
kwok-gpu-node-5         Ready    gpu-agent       4s    fake
kwok-gpu-node-6         Ready    gpu-agent       4s    fake
kwok-gpu-node-7         Ready    gpu-agent       3s    fake
kwok-gpu-node-8         Ready    gpu-agent       3s    fake
kwok-gpu-node-9         Ready    gpu-agent       3s    fake
kwok-standard-node-1    Ready    agent           63s   fake
kwok-standard-node-10   Ready    agent           61s   fake
kwok-standard-node-11   Ready    agent           61s   fake
kwok-standard-node-12   Ready    agent           61s   fake
kwok-standard-node-13   Ready    agent           61s   fake
kwok-standard-node-14   Ready    agent           60s   fake
kwok-standard-node-15   Ready    agent           60s   fake
kwok-standard-node-16   Ready    agent           60s   fake
kwok-standard-node-17   Ready    agent           60s   fake
kwok-standard-node-18   Ready    agent           59s   fake
kwok-standard-node-19   Ready    agent           59s   fake
kwok-standard-node-2    Ready    agent           63s   fake
kwok-standard-node-20   Ready    agent           59s   fake
kwok-standard-node-3    Ready    agent           63s   fake
kwok-standard-node-4    Ready    agent           63s   fake
kwok-standard-node-5    Ready    agent           62s   fake
kwok-standard-node-6    Ready    agent           62s   fake
kwok-standard-node-7    Ready    agent           62s   fake
kwok-standard-node-8    Ready    agent           62s   fake
kwok-standard-node-9    Ready    agent           62s   fake
(base) asheller: many_nodes$ 
```
