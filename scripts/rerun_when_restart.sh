#!/bin/bash

# /home/ubuntu/my_kaggle_dockerに移動
cd /home/ubuntu/my_kaggle_docker

# Docker Composeを使用してコンテナを起動
docker compose up -d

# コンテナが立ち上がるまで少し待機する（例: 5秒）
sleep 8

# 最新のDockerコンテナIDを取得
CONTAINER_ID=$(docker ps -l -q)

# Dockerコンテナ内でシェルスクリプトを実行
docker exec $CONTAINER_ID /bin/bash -c "/kaggle/scripts/train.sh"

sudo systemctl disable kaggle_docker.service