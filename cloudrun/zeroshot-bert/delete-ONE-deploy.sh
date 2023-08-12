#!/bin/bash

docker stop $1
docker rm $1
docker rmi $2
docker build -t rest_fe:latest .
docker run -d -p 5000:5000 rest_fe
