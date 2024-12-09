#!/bin/bash

# Create HDFS directories and upload data
docker exec namenode bash -c "
  hdfs dfs -mkdir -p /input &&
  hdfs dfs -put /tmp/data/ecommerce_data_with_trends.csv /input/ &&
  hdfs dfs -mkdir -p /output &&
  hdfs dfs -chown -R spark:spark /output &&
  hdfs dfs -chmod -R 770 /output
"

## Install Python dependencies in the Spark container
#docker exec spark bash -c "
#  if [ -f scripts/requirements.txt ]; then
#    pip install --user -r scripts/requirements.txt
#  else
#    echo 'Error: scripts/requirements.txt not found!'
#    exit 1
#  fi
#"