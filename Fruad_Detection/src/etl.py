from pyspark.sql import SparkSession
from pyspark.sql import functions as F

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


def transform_data(df):
    keep_columns = [
    "trans_date_trans_time",  # → extract hour, day
    "cc_num",                 # → tokenize to first4_last4
    "merchant",
    "category",
    "amt",
    "city_pop",
    "lat", "long",            # cardholder location
    "merch_lat", "merch_long",# merchant location
    "gender",
    "dob",                    # → derive age
    "trans_num",              # unique transaction ID
    "is_fraud"                # target variable!
    ]
        # Step 1: Select req columns
    
    df= df.select(keep_columns)

    # Step 2: Add new column "trans_hour" 
    # extracted from trans_date_trans_time
    df=df.withColumn("trans_hour", F.hour(F.col("trans_date_trans_time")))

    
    # Step 3: Add "trans_day" - day of week
    df=df.withColumn("trans_day",F.dayofweek(F.col("trans_date_trans_time")))
    #  Derive age from dob
    df=df.withColumn("age", F.floor(F.months_between(F.current_date(), F.col("dob"))/12) )
    # 5. Tokenize cc_num → first4_last4 TODO
    df=df.withColumn("cc_token",F.concat(F.substring("cc_num",1,4), 
                                         F.lit("****"),F.substring("cc_num", -4,4)))
    # 6. Calculate distance between cardholder and merchant TODO
    df=df.withColumn("dist", 
                     F.sqrt(
                        (F.pow(F.col("merch_lat")-F.col("lat"), 2)+ 
                        F.pow(F.col("merch_long")-F.col("long"), 2) )
                        ))
    # 7. Drop nulls or fill them
    df=df.fillna(0)
    df.drop("cc_num","dob","merch_lat","merch_long","lat","long", "trans_date_trans_time")
    # return transformed df
    return df

if __name__=="__main__":
    spark= create_spark_session()
    df=transform_data(load_data(spark, "data/fraudTrain.csv")) #data downloaded from kaggle 
    df.show(5)
    save_data(df,"output/fraud_data")