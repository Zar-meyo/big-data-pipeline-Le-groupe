from ingestion import get_spark_df
from data_processing import process_data
from spark_analysis import run_spark_analysis

import logger as lg

if __name__ == '__main__':
    logger = lg.get_module_logger(__name__)
    logger.debug('Getting spark dataframe')
    df = get_spark_df()
    logger.debug("Processing data")
    df = process_data(df)
    df.show()
    logger.debug('Starting spark analysis')
    run_spark_analysis(df, is_processed=True)
