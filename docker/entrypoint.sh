#!/usr/bin/env bash


# Run the Python script
python kwok_cluster/create_nodes.py --node_count=10

# Use supervisord to start the srevices we need.
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
