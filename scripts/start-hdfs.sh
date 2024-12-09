#!/bin/bash

# Variables
HADOOP_HOME=/opt/hadoop
NAMENODE_DIR=/opt/hadoop/data/nameNode

# Format the NameNode if necessary
if [ ! -d "$NAMENODE_DIR/current" ]; then
  echo "Formatting NameNode..."
  $HADOOP_HOME/bin/hdfs namenode -format -force -nonInteractive
else
  echo "NameNode already formatted."
fi

# Start the NameNode in the background to ensure HDFS commands work
echo "Starting NameNode in the background..."
$HADOOP_HOME/bin/hdfs namenode &

# Wait for NameNode to start
sleep 5

# Create HDFS directories and upload data
echo "Setting up HDFS directories and uploading data..."
$HADOOP_HOME/bin/hdfs dfs -mkdir -p /input
$HADOOP_HOME/bin/hdfs dfs -put /tmp/data/ecommerce_data_with_trends.csv /input/
$HADOOP_HOME/bin/hdfs dfs -mkdir -p /output
$HADOOP_HOME/bin/hdfs dfs -chown -R spark:spark /output
$HADOOP_HOME/bin/hdfs dfs -chmod -R 770 /output

# Bring NameNode to the foreground
wait