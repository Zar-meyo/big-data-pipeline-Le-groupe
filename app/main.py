from pyspark.sql import SparkSession

from ingestion import get_spark_df
from data_processing import process_data
from spark_analysis import run_spark_analysis
from MLClustering_Analysis import ml_clustering
from Predicting_modeling import ml_prediction

import logger as lg


def get_spark_session():
    return SparkSession.builder \
        .master("spark://spark:7077") \
        .appName("ingestion") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .getOrCreate()


if __name__ == '__main__':
    logger = lg.get_module_logger(__name__)
    logger.info('Getting spark dataframe')
    spark = get_spark_session()
    df = get_spark_df(spark)
    logger.info("Processing data")
    df_preprocessed = process_data(df)
    logger.info('Starting spark analysis')
    run_spark_analysis(df_preprocessed, is_processed=True)

    logger.info("Starting ML clustering...")
    ml_clustering(df, spark)
    logger.info("Starting ML prediction...")
    ml_prediction(df, spark)

