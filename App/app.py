#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NYC Taxi ML Pipeline - EMR Production Version
Usage:
spark-submit nyc_taxi_ml.py s3://nyc-taxi-project-2026
"""

import sys
from datetime import datetime
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    DecisionTreeRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator


# ============================
# 0. Détection automatique IP driver
# ============================
def get_driver_ip():
    """
    Retourne l'IP interne du conteneur sur eth0 pour Spark driver.
    """
    try:
        ip = subprocess.check_output(
            "ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1",
            shell=True,
            text=True
        ).strip()
        print(f"Driver IP détectée : {ip}")
        return ip
    except Exception as e:
        print("Impossible de détecter l'IP automatiquement, fallback sur 127.0.0.1:", e)
        return "127.0.0.1"

DRIVER_IP = get_driver_ip()


# ============================
# 1. Argument validation
# ============================

import argparse

parser = argparse.ArgumentParser(description='NYC Taxi ML Pipeline')
parser.add_argument('--input', type=str, help='S3 input path')
parser.add_argument('--output', type=str, help='S3 output path')
parser.add_argument('bucket', type=str, nargs='?', help='Legacy S3 bucket positional argument')

args = parser.parse_args()

if args.input and args.output:
    INPUT_PATH = args.input
    BASE_OUTPUT_PATH = args.output.rstrip("/")
elif args.bucket:
    S3_BUCKET = args.bucket.rstrip("/")
    INPUT_PATH = f"{S3_BUCKET}/input/yellow_tripdata_2016-01.csv"
    BASE_OUTPUT_PATH = f"{S3_BUCKET}/output"
else:
    parser.print_help()
    sys.exit(1)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL_PATH = f"{BASE_OUTPUT_PATH}/model_{timestamp}"
PREDICTIONS_PATH = f"{BASE_OUTPUT_PATH}/predictions_{timestamp}"
METRICS_PATH = f"{BASE_OUTPUT_PATH}/metrics_{timestamp}"


# ============================
# 2. Spark Session
# ============================

spark = SparkSession.builder \
    .appName("NYC_Taxi_EMR_Project") \
    .config("spark.driver.host", DRIVER_IP) \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("JOB STARTED")
print(f"Input path: {INPUT_PATH}")
print(f"Output base path: {BASE_OUTPUT_PATH}")


# ============================
# 3. Ingestion
# ============================

try:
    df = spark.read.csv(
        INPUT_PATH,
        header=True,
        inferSchema=True
    )
except Exception as e:
    print("Error loading dataset:", str(e))
    spark.stop()
    sys.exit(1)

print("Dataset loaded successfully")


# ============================
# 4. Data Cleaning
# ============================

df = df.filter((col("trip_distance") > 0) & (col("trip_distance") < 100))
df = df.filter((col("passenger_count") > 0) & (col("passenger_count") < 7))

df = df.dropna(subset=[
    "trip_distance",
    "passenger_count",
    "total_amount",
    "payment_type",
    "tpep_pickup_datetime"
])


# ============================
# 5. Feature Engineering
# ============================

df = df.withColumn("hour", hour("tpep_pickup_datetime"))
df = df.withColumn("day_of_week", dayofweek("tpep_pickup_datetime"))


# ============================
# 6. Train / Test Split
# ============================

train, test = df.randomSplit([0.8, 0.2], seed=42)


# ============================
# 7. Pipeline Components
# ============================

indexer = StringIndexer(
    inputCol="payment_type",
    outputCol="payment_index",
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=[
        "trip_distance",
        "passenger_count",
        "hour",
        "day_of_week",
        "payment_index"
    ],
    outputCol="features"
)

evaluator_rmse = RegressionEvaluator(
    labelCol="total_amount",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="total_amount",
    predictionCol="prediction",
    metricName="r2"
)

models = {
    "LinearRegression": LinearRegression(
        featuresCol="features",
        labelCol="total_amount"
    ),
    "RandomForest": RandomForestRegressor(
        featuresCol="features",
        labelCol="total_amount",
        numTrees=50
    ),
    "DecisionTree": DecisionTreeRegressor(
        featuresCol="features",
        labelCol="total_amount"
    )
}


# ============================
# 8. Training and Evaluation
# ============================

best_model = None
best_rmse = float("inf")
best_name = ""
metrics_results = []

for name, model in models.items():

    pipeline = Pipeline(stages=[indexer, assembler, model])
    fitted_model = pipeline.fit(train)

    predictions = fitted_model.transform(test)

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    metrics_results.append((name, float(rmse), float(r2)))

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = fitted_model
        best_name = name

print(f"Best model selected: {best_name}")


# ============================
# 9. Save Model and Results
# ============================

best_model.write().save(MODEL_PATH)

final_predictions = best_model.transform(test)

final_predictions.select(
    "trip_distance",
    "total_amount",
    "prediction"
).write.mode("overwrite").parquet(PREDICTIONS_PATH)

metrics_df = spark.createDataFrame(
    metrics_results,
    ["model_name", "rmse", "r2"]
)

metrics_df.write.mode("overwrite").parquet(METRICS_PATH)

print("MODEL SAVED TO:", MODEL_PATH)
print("PREDICTIONS SAVED TO:", PREDICTIONS_PATH)
print("METRICS SAVED TO:", METRICS_PATH)

print("JOB COMPLETED SUCCESSFULLY")

spark.stop()
