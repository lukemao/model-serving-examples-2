#!/bin/bash

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker rmi $(docker images -a -q)
docker build -t rest_fe:latest .
docker run -d -p 5000:5000 rest_fe
