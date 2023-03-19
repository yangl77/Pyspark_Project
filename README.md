# Pyspark_Project

## Requirements
* Java, Python (if pyspark)
* Hadoop
* Spark

## Steps
* dfs format- hadoop namenode -format
* start dfs - start-dfs.sh (if data is stored at HDFS system)
* start-yarn (if need)
* Enter spark/sbin/
* start-all.sh (Start spark cluster)
* spark-submit **--master [param]** **--executor-cores [number]**xxx.py (this is standlone mode)
