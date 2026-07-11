from  pyspark.sql import Window
from pyspark.sql import functions as F


def trasnform(df):
    past_7= Window.partitionBy("ticker").orderBy("Date").rowsBetween(-6,0)
    past_30=Window.partitionBy("ticker").orderBy("Date").rowsBetween(-29,0)
    df=df.withColumn("ma_7", F.avg("Close").over(past_7))
    df=df.withColumn("ma_30",F.avg("Close").over(past_30))
        # 1. daily_return = (close - open) / open * 100
    df=df.withColumn("daily_return", (F.col("Close")-F.col("Open"))/ (F.col("Open"))*100)
        # 2. price_range = high - low
    df=df.withColumn("price_range",F.col("High")-F.col("Low"))
    # 3. is_bullish = close > open
    df=df.withColumn("is_bullish", F.when(F.col("Close")>F.col("Open"), True).otherwise(False))
    # 6. volatility = (high - low) / open * 100
    df=df.withColumn("volatility", (F.col("High")-F.col("Low"))/ F.col("Open") * 100)

    return df
