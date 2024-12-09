import pandas as pd
import logger as lg


def get_spark_df(spark):
    logger = lg.get_module_logger(__name__)
    logger.debug("spark begin")
    data_path = "hdfs://namenode:9000/input/ecommerce_data_with_trends.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    logger.debug("spark done")
    logger.debug("writting df in hdfs...")
    df.write.csv("hdfs://namenode:9000/output/cleaned_data.csv", mode="overwrite", header=True)
    return df
