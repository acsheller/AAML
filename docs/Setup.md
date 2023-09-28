# Setup 

This is one particular way of setting a "system" up to run this program.  The system developed on is a Windows 10 system. [Windows Subsystem for Linux 2](https://www.windowscentral.com/how-install-wsl2-windows-10) (WSL2) is a linux environment that runs inside of Windows 10 or Wndows 11, used to provide a full-featured Linux environment for development. One can follow the instructions provided at the link for installation.

Install Docker Desktop and have it run a one node Kubernetes cluster: ([Instructions for installing Docker Desktop ](https://birthday.play-with-docker.com/kubernetes-docker-desktop/)). This is a great way of getting started with Docker and Kubernetes (K8s). 

Kubectl needs to be installed as well. This is a single binary that lets one interact with the cluster. To install it, [follow these instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/).
