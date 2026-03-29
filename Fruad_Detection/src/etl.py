from pyspark.sql import SparkSession

def create_spark_session() -> SparkSession :

    spark=SparkSession.builder\
            .appName("FraudDetectioNPipeline")\
            .master("local[*]")\
            .getOrCreate()
    return spark


def load_data(spark, path):
    df=spark.read.csv(path,header=True, inferSchema=True)
    print(df.count())
    print(df.printSchema())
    return df


def save_data(df, output_path):
    df.write.mode("overwrite").parquet(output_path)
    print("data writted to parquet format")

if __name__=="__main__":
    spark= create_spark_session()
    load_data(spark, "../data/fraudTrain.csv")
    df.show(5)
    save_data(df,"../output/fraud_data")