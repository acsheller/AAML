ARG BASE_IMAGE=pytorch/pytorch:latest
# Use the specified base image
FROM $BASE_IMAGE

# Update package information
RUN apt-get update

# Install Python and pip
RUN apt-get install -y curl

# Install Jupyter Lab and kubernetes
RUN pip3 install jupyterlab kubernetes

# Get kubectl 
RUN curl -L -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
RUN chmod +x /usr/local/bin/kubectl
# Create a non-root user
RUN useradd -m -d /home/appuser -s /bin/bash appuser

# Give the user ownership of their home directory
RUN chown -R appuser:appuser /home/appuser

# Expose port 8888 for Jupyter Lab
EXPOSE 8888

# Switch to the non-root user
USER appuser

# Add debugging command to list the current directory contents
RUN ls -l

# Copy the contents of /app/build into /desired/directory within the container
COPY kwok_cluster /home/appuser/kwok_cluster
COPY kwok_workloads /home/appuser/kwok_workloads
COPY kwok_yaml /home/appuser/kwok_yaml
COPY scheduler /home/appuser/scheduler
COPY docker/entrypoint.sh /home/appuser/entrypoint.sh
#COPY ../tmux /home/appuser/tmux

#set a working directory
WORKDIR /home/appuser/

# Optionally, start Jupyter Lab (you can customize the command)
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
#CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
#CMD "jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
#ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["/home/appuser/entrypoint.sh"]


