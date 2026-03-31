from pyspark.sql import Window
from pyspark.sql import functions as F

def add_velocity_features(df):

    card_1hr= Window.partitionBy("cc_num").orderBy("unix_time").rangeBetween(-3600,0)
    card_all =Window.partitionBy("cc_num")
    card_ordered= Window.partitionBy("cc_num").orderBy("unix_time")
    
    df=df.withColumn("tx_count_1hr", F.count("*").over(card_1hr))
    df=df.withColumn("amt_sum_1hr", F.sum("amt").over(card_1hr))
    df=df.withColumn("avg_amt_historical",F.avg("amt").over(card_all))
    df=df.withColumn("amt_ratio",F.col("amt")/F.col("avg_amt_historical"))
    df=df.withColumn("prev_time", F.lag("unix_time",1).over(card_ordered))
    df=df.withColumn("time_since_last_tx", (F.col("unix_time")-F.col("prev_time")))
    df=df.fillna(0,subset=["time_since_last_tx"])

    return df

