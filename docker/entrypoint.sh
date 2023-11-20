#!/bin/bash
#jupyter-lab --ip=0.0.0.0 --no-browser --port=8888
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
