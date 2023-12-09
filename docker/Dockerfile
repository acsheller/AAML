## 
## Build in the directory below this with
##
## docker build -f docker/Dockerfile it aaml_kwok:latest .
##
## Use --no-cache and it will re-download everything 
##
## This is necessasry to copy the folders in properly and
## to maintain a good directory structure that makes sense.

ARG BASE_IMAGE=pytorch/pytorch:latest
# Use the specified base image
FROM $BASE_IMAGE

# Update package information
RUN apt-get update

# Install Python and pip
RUN apt-get install -y curl vim

# Install Jupyter Lab and kubernetes
RUN pip3 install jupyterlab kubernetes networkx torch-geometric tensorboard supervisor streamlit redis randomname optuna

# Get kubectl 
RUN curl -L -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
RUN chmod +x /usr/local/bin/kubectl
# Create a non-root user
RUN useradd -m -d /home/appuser -s /bin/bash appuser

# Give the user ownership of their home directory
RUN chown -R appuser:appuser /home/appuser

# Expose port 8888 for Jupyter Lab
EXPOSE 8888 6006



# Set default shell for the user to bash
ENV SHELL=/bin/bash

# For Jupyter Lab -- A simpler Token
ENV JUPYTER_TOKEN='AAML'

# Make .kube folder and copy config to it
RUN mkdir /home/appuser/.kube
COPY docker/config /home/appuser/.kube/config

RUN mkdir /home/appuser/tlogs
RUN mkdir /home/appuser/logs
RUN mkdir /home/appuser/models
RUN mkdir /home/appuser/deployment_data
RUN mkdir /home/appuser/scripts
RUN mkdir /home/appuser/data
# Copy the contents of /app/build into /desired/directory within the container
## Changed to mounting in folders in the docker compose script.
#COPY scripts /home/appuser/scripts
#COPY data /home/appuser/data
#COPY kwok_workloads /home/appuser/kwok_workloads
#COPY scheduler /home/appuser/scheduler
COPY docker/entrypoint.sh /home/appuser/entrypoint.sh
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY ProjectNotebook.ipynb /home/appuser/ProjectNotebook.ipynb


#set a working directory
WORKDIR /home/appuser/

# Handy Aliases
RUN echo "alias k='kubectl'" >> .bashrc && \
    echo "alias kdp='kubectl describe pod'" >> .bashrc && \
    echo "alias kd='kubectl describe'" >> .bashrc && \
    echo "alias kdel='kubectl delete'" >> .bashrc && \
    echo "alias kgp='kubectl get pods'" >> .bashrc && \
    echo "alias kgn='kubectl get nodes'" >> .bashrc && \
    echo "alias ap='ansible-playbook'" >> .bashrc && \
    echo "alias kge='kubectl get events --sort-by=.metadata.creationTimestamp'" >> .bashrc && \
    echo "alias kg='kubectl get'" >> .bashrc && \
    echo "alias kl='kubectl logs'" >> .bashrc && \
    echo "alias rungnn='python3 /home/appuser/scripts/scheduler_gnn.py'" >> .bashrc && \
    echo "alias rundqn='python3 /home/appuser/scripts/scheduler_dqn.py'" >> .bashrc && \
    echo "alias runsim='python3 /home/appuser/scripts/workload_deployment_simulator.py '" >> .bashrc && \ 
    echo "alias ktop='watch python3 /home/appuser/scripts/kinfo.py'" >> .bashrc && \
    echo "alias cleancluster='python3 /home/appuser/scripts/cleanup_cluster.py'" >> .bashrc && \
    echo "alias runacdqn='python3 /home/appuser/scripts/actorcritic_dqn.py'" >> .bashrc && \
    echo "alias runacgnn='python3 /home/appuser/scripts/actorcritic_gnn.py'" >> .bashrc && \
    echo ""

# Create or modify .bash_profile to source .bashrc
RUN echo "if [ -f ~/.bashrc ]; then source ~/.bashrc; fi" >> .bash_profile

RUN ["chmod", "+x", "/home/appuser/entrypoint.sh"]
RUN  ["chmod", "+r", "/etc/supervisor/conf.d/supervisord.conf"]
# Switch to the non-root user
# Give the user ownership of their home directory
RUN chown -R appuser:appuser /home/appuser


USER appuser
ENTRYPOINT ["/home/appuser/entrypoint.sh"]


