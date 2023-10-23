# Setup 


Windows 10 systems using WSL2 `Ubuntu 22.04` with Docker Desktop with Kubernetes started. This is one of hundreds of configurations that developers can build on -- it's just my setup. It's not ncessary as all steps are generic enough.  One needs a one node cluster of some sorts, with a Linux environemnt.


**<u>WSL2</u>** -
This is one particular way of setting a "system" up to run this program.  The system developed on is a Windows 10 system. [Windows Subsystem for Linux 2](https://www.windowscentral.com/how-install-wsl2-windows-10) (WSL2) is a linux environment that runs inside of Windows 10 or Wndows 11, used to provide a full-featured Linux environment for development. One can follow the instructions provided at the link for installation.

**<u>Docker Desktop with Kubernetes</u>** - 
Install Docker Desktop and have it run a one node Kubernetes cluster: ([Instructions for installing Docker Desktop ](https://birthday.play-with-docker.com/kubernetes-docker-desktop/)). This is a great way of getting started with Docker and Kubernetes (K8s). 

**<u>kubectl</u>** - 
Kubectl needs to be installed as well. This is a single binary that lets one interact with the cluster. To install it, [follow these instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/).

**<u>Kwok Stuff</u>** - 
Install Kwok stuff-- [follow these instructions](https://kwok.sigs.k8s.io/docs/user/installation/). This document walks through the installation of kwokctl and kwok binaries.  

**<u>Aliases</u> ** (optional but useful) - 
I use aliases because I get tired of typing kubectl all the time. Here are a list that will be useful. You are encouraged to make your own.
```
alias k='kubectl'
alias kdp='kubectl describe pod'
alias kd='kubectl describe'
alias kdel='kubectl delete'
alias kgp='kubectl get pods'
alias kgn='kubectl get nodes'
alias ap='ansible-playbook'
alias kge='kubectl get events --sort-by=.metadata.creationTimestamp'
alias kgp='kubectl get pods'
alias wkgp='watch kubectl get pods'
alias kg='kubectl get'
alias kd='kubectl describe'
alias kl='kubectl logs'
alias runredis='docker run --name redis-container -p 6379:6379 redis'

```
Try these out for yourself to see what they do -- most are self explanatory.

**<u>Setup kwok</u>** -
At this point Docker Desktop is running and Kubernetes is active. If you are not sure, try and figure out how to be sure. I use [Windows Terminal](https://apps.microsoft.com/detail/windows-terminal/9N0DX20HK701?hl=en-gb&gl=US) which I have setup to launch a WSL2 Ubuntu 22.04 shell as the default.

Run a WSL2 prompt.  
```
k get nodes

NAME             STATUS   ROLES           AGE   VERSION
docker-desktop   Ready    control-plane   3h    v1.27.2

```

This is what you see now.  The approach used in this work was `Kwok in Cluster`. Given a one node cluster one can simulate many more nodes.
 Follow these instructions to [Deploy kwok in a Cluster](https://kwok.sigs.k8s.io/docs/user/kwok-in-cluster/).  Note that you may need to install `jq` if you haven't already.

 ```
 (DRL) asheller: kwok_yaml$ KWOK_REPO=kubernetes-sigs/kwok
(DRL) asheller: kwok_yaml$ echo $KWOK_REPO
kubernetes-sigs/kwok
(DRL) asheller: kwok_yaml$ KWOK_LATEST_RELEASE=$(curl "https://api.github.com/repos/${KWOK_REPO}/releases/latest" | jq -r '.tag_name')
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 41628    0 41628    0     0   123k      0 --:--:-- --:--:-- --:--:--  123k

(DRL) asheller: kwok_yaml$ k apply -f "https://github.com/${KWOK_REPO}/releases/download/${KWOK_LATEST_RELEASE}/kwok.yaml"
customresourcedefinition.apiextensions.k8s.io/attaches.kwok.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/clusterattaches.kwok.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/clusterexecs.kwok.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/clusterlogs.kwok.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/clusterportforwards.kwok.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/execs.kwok.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/logs.kwok.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/metrics.kwok.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/portforwards.kwok.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/stages.kwok.x-k8s.io created
serviceaccount/kwok-controller created
clusterrole.rbac.authorization.k8s.io/kwok-controller created
clusterrolebinding.rbac.authorization.k8s.io/kwok-controller created
service/kwok-controller created
deployment.apps/kwok-controller created

(DRL) asheller: kwok_yaml$ kubectl apply -f "https://github.com/${KWOK_REPO}/releases/download/${KWOK_LATEST_RELEASE}/stage-fast.yaml"
stage.kwok.x-k8s.io/node-heartbeat-with-lease created
stage.kwok.x-k8s.io/node-initialize created
stage.kwok.x-k8s.io/pod-complete created
stage.kwok.x-k8s.io/pod-delete created
stage.kwok.x-k8s.io/pod-ready created

 ```

This has installed the necessary Kubernetes manifests to emulate a cluster.

**<u>Add fake nodes</u>** --<br>

- [Single Node](../kwok_cluster/single_node/kwok_node.yaml) This is the K8s artifact to setup one additional node in the cluster.
```
(base) asheller: single_node$ ls
Readme.md  kwok_node.yaml  kwok_node_gpu.yaml
(base) asheller: single_node$ kgn
NAME             STATUS   ROLES           AGE     VERSION
docker-desktop   Ready    control-plane   3h20m   v1.27.2
(base) asheller: single_node$ k apply -f kwok_node.yaml 
node/kwok-node-0 created
(base) asheller: single_node$ kgn
NAME             STATUS   ROLES           AGE     VERSION
docker-desktop   Ready    control-plane   3h20m   v1.27.2
kwok-node-0      Ready    agent           3s      fake

```
To delete the fake node

```
base) asheller: single_node$ k delete -f kwok_node.yaml 
node "kwok-node-0" deleted
(base) asheller: single_node$ kg nodes
NAME             STATUS   ROLES           AGE     VERSION
docker-desktop   Ready    control-plane   3h22m   v1.27.2
```

- [Many Nodes](../kwok_cluster/many_nodes/create_nodes.py)

```
(base) asheller: many_nodes$ python create_nodes.py 
Enter the total number of standard nodes to create: 5
node/kwok-std-node-1 created
node/kwok-std-node-2 created
node/kwok-std-node-3 created
node/kwok-std-node-4 created
node/kwok-std-node-5 created
(base) asheller: many_nodes$ kgn
NAME              STATUS   ROLES           AGE     VERSION
docker-desktop    Ready    control-plane   3h25m   v1.27.2
kwok-std-node-1   Ready    agent           4s      fake
kwok-std-node-2   Ready    agent           3s      fake
kwok-std-node-3   Ready    agent           3s      fake
kwok-std-node-4   Ready    agent           3s      fake
kwok-std-node-5   Ready    agent           3s      fake
(base) asheller: many_nodes$ 

```
To delete the nodes

```
(base) asheller: many_nodes$ python delete_nodes.py 
Enter the total number of ordinary nodes to delete: 5
node "kwok-std-node-1" deleted
node "kwok-std-node-2" deleted
node "kwok-std-node-3" deleted
node "kwok-std-node-4" deleted
node "kwok-std-node-5" deleted
(base) asheller: many_nodes$ kgn
NAME             STATUS   ROLES           AGE     VERSION
docker-desktop   Ready    control-plane   3h26m   v1.27.2
(base) asheller: many_nodes$ 
```

**<u>Summary</u>** - 
so now a fake cluster can be created to explore scheduling problems.

