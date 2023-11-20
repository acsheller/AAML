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
RUN pip3 install jupyterlab kubernetes networkx torch-geometric tensorboard supervisor streamlit redis randomname

# Get kubectl 
RUN curl -L -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
RUN chmod +x /usr/local/bin/kubectl
# Create a non-root user
RUN useradd -m -d /home/appuser -s /bin/bash appuser

# Give the user ownership of their home directory
RUN chown -R appuser:appuser /home/appuser

# Expose port 8888 for Jupyter Lab
EXPOSE 8888 6006

# Switch to the non-root user
USER appuser

# Set default shell for the user to bash
ENV SHELL=/bin/bash
ENV JUPYTER_TOKEN='AAML'

# Make .kube folder and copy config to it
RUN mkdir /home/appuser/.kube
COPY docker/config /home/appuser/.kube/config

RUN mkdir /home/appuser/tlogs

# Copy the contents of /app/build into /desired/directory within the container
COPY kwok_cluster /home/appuser/kwok_cluster
COPY kwok_workloads /home/appuser/kwok_workloads
COPY kwok_yaml /home/appuser/kwok_yaml
COPY scheduler /home/appuser/scheduler
COPY docker/entrypoint.sh /home/appuser/entrypoint.sh
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

#COPY ../tmux /home/appuser/tmux

#set a working directory
WORKDIR /home/appuser/

# Handy Aliases to have on hand
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
    echo "alias runredis='docker run --name redis-container -p 6379:6379 redis'" >> .bashrc && \
    echo "alias cleancluster='python3 /home/appuser/kwok_workloads/cleanup_cluster.py'" >> .bashrc

# Create or modify .bash_profile to source .bashrc
RUN echo "if [ -f ~/.bashrc ]; then source ~/.bashrc; fi" >> .bash_profile


ENTRYPOINT ["/home/appuser/entrypoint.sh"]


