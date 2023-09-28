# kwok single worker node

These yaml create one node each. One provides for GPU Resources.

Simply apply them to the cluster
```
k apply -f kwok_node.yaml

node/kwok-node-0 created

(base) asheller: single_node$ kg nodes
NAME             STATUS   ROLES           AGE   VERSION
docker-desktop   Ready    control-plane   54d   v1.27.2
kwok-node-0      Ready    agent           3s    fake

```
```
k apply -f kwok_node_gpu.yaml 

node/kwok-gpu-node created

(base) asheller: single_node$ kg nodes
NAME             STATUS   ROLES           AGE   VERSION
docker-desktop   Ready    control-plane   54d   v1.27.2
kwok-gpu-node    Ready    agent           5s    fake
kwok-node-0      Ready    agent           60s   fake
```

## View the node resources

Be sure to run this for yourself to view the 

```
kd node kwok-gpu-node
..
...
Addresses:
  InternalIP:  10.1.0.183
Capacity:
  cpu:             4
  memory:          16Gi
  nvidia.com/gpu:  2
  pods:            110
Allocatable:
  cpu:             4
  memory:          16Gi
  nvidia.com/gpu:  2
  pods:            110
System Info:
  Machine ID:                 
  System UUID:                
  Boot ID:                    
  Kernel Version:             kwok-v0.4.0
  OS Image:                   
  Operating System:           linux
  Architecture:               amd64
  Container Runtime Version:  kwok-v0.4.0
  Kubelet Version:            fake
  Kube-Proxy Version:         fake
Non-terminated Pods:          (1 in total)
  Namespace                   Name                CPU Requests  CPU Limits  Memory Requests  Memory Limits  Age
  ---------                   ----                ------------  ----------  ---------------  -------------  ---
  kube-system                 kube-proxy-h7ccz    0 (0%)        0 (0%)      0 (0%)           0 (0%)         77s
Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests  Limits
  --------           --------  ------
  cpu                0 (0%)    0 (0%)
  memory             0 (0%)    0 (0%)
  ephemeral-storage  0 (0%)    0 (0%)
  nvidia.com/gpu     0         0
Events:
  Type    Reason          Age   From             Message
  ----    ------          ----  ----             -------
  Normal  RegisteredNode  66s   node-controller  Node kwok-gpu-node event: Registered Node kwok-gpu-node in Controller


```
