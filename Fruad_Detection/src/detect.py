from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.classification import  RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator 

def prepare_feature_vector(df):

    feature_cols = [
        "amt", "city_pop", "lat", "long",
        "merch_lat", "merch_long", "trans_hour",
        "trans_day", "age", "dist", "tx_count_1hr",
        "amt_sum_1hr", "avg_amt_historical",
        "amt_ratio", "time_since_last_tx"
    ]
    assembler= VectorAssembler(
        inputCols= feature_cols,
        outputCol ="features"
    )

    df=assembler.transform(df)

    return df

def train_model(df):
    train_df,test_df=df.randomSplit([0.8,0.2],seed=42)
    rf=RandomForestClassifier(
        labelCol="is_fraud",
        featuresCol="features",
            numTrees =10    )
    
    model=rf.fit(train_df)
    predictions=model.transform(test_df)
    predictions.select("is_fraud","prediction","probability").show(5)
    return model, predictions

def evaluate_model(predictions):
    evaluator=BinaryClassificationEvaluator(
        labelCol ="is_fraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc_score=evaluator.evaluate(
        predictions
    )
    print(f"AUC Score: {auc_score}")
    return auc_score

    