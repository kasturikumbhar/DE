import yfinance as yf
import pandas as pd
from pyspark.sql import SparkSession
from  transform import trasnform
def create_spark_session():
    return SparkSession.builder.\
            appName("StockMarketPipeline").\
            config("spark.jars.packages", 
                 "io.delta:delta-core_2.12:2.4.0").\
            config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension").\
            config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog").\
            master("local[*]").\
            getOrCreate()
    
def fetch_stock_data(tickers,period='1y'):

    all_data =[]
    for ticker in tickers:
        df=yf.download(ticker,period=period)
        df['ticker']=ticker
        df.reset_index(inplace=True)
        df.columns=[ col[0] if isinstance(col,tuple)
                     else col for col in df.columns]
        all_data.append(df)
    return pd.concat(all_data)

def clean_data(pandas_df):
    df=pandas_df.dropna()
    #how do i change names of cols here? i see col names in tuple format ("Close","APPL") etc

def ingest_to_spark(spark,pandas_df):
    return spark.createDataFrame(pandas_df)

def save_raw(spark_df):
# Replace Delta save with Parquet
    spark_df.write \
    .partitionBy("ticker") \
    .mode("overwrite") \
    .parquet("output/raw_stocks")


if __name__=="__main__":
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    spark=create_spark_session()

    print("Fetching Stock Data!")

    pandas_df= fetch_stock_data(tickers)
    print(f' Fetched data from yahoo finance {len(pandas_df)} records')

    spark_df= ingest_to_spark(spark,pandas_df)
    spark_df.printSchema()
    spark_df.show(5)
    df=trasnform(spark_df)
    df.show(5)
    # save_raw(spark_df)

    print("Raw data saved!")
