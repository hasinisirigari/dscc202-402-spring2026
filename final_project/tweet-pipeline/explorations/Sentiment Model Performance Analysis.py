# Databricks notebook source
# MAGIC %md
# MAGIC # Sentiment Model Performance Analysis
# MAGIC
# MAGIC ## Purpose
# MAGIC Evaluate sentiment classification model performance by comparing predictions against ground truth.
# MAGIC Generate classification metrics and log results to MLflow for experiment tracking.
# MAGIC
# MAGIC ## Requirements
# MAGIC - Load tweets_gold table with predictions and ground truth
# MAGIC - Calculate classification metrics (accuracy, precision, recall, F1)
# MAGIC - Generate confusion matrix visualization
# MAGIC - Log metrics, parameters, and artifacts to MLflow
# MAGIC
# MAGIC ## Expected Output
# MAGIC - Classification report with per-class metrics
# MAGIC - Confusion matrix visualization
# MAGIC - MLflow experiment with accuracy metric and confusion matrix artifact
# MAGIC
# MAGIC ## Reference
# MAGIC See Lab 0.5 (MLops) for MLflow experiment tracking patterns

# COMMAND ----------

# TODO: Import necessary libraries
# You will need:
# - pyspark.sql functions
# - pandas
# - mlflow and MlflowClient
# - delta.tables.DeltaTable
# - matplotlib.pyplot
# - sklearn.metrics (confusion_matrix, classification_report, ConfusionMatrixDisplay)

from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient

from delta.tables import DeltaTable

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Load Gold Data
# MAGIC
# MAGIC TODO: Read the tweets_gold table to get predicted and actual sentiments
# MAGIC - Load table using spark.read.format("delta").table()
# MAGIC - Table contains sentiment_id (ground truth) and predicted_sentiment_id (model prediction)
# MAGIC - Both are binary: 0=negative, 1=positive/neutral

# COMMAND ----------

# TODO: Load gold table
GOLD_TABLE = "workspace.default.tweets_gold"
SILVER_TABLE = "workspace.default.tweets_silver"

gold_df = spark.read.format("delta").table(GOLD_TABLE)

print(f"Gold row count: {gold_df.count()}")
gold_df.printSchema()
display(gold_df.limit(5))


# COMMAND ----------

eval_df = gold_df.filter(
    (F.col("sentiment_id").isin(0, 1))
    & (F.col("predicted_sentiment_id").isin(0, 1))
)

dropped = gold_df.count() - eval_df.count()
print(f"Rows kept for evaluation: {eval_df.count()}")
print(f"Rows dropped (unmapped labels): {dropped}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Generate Classification Report
# MAGIC
# MAGIC TODO: Convert to pandas and compute classification metrics
# MAGIC 1. Convert gold DataFrame to pandas using .toPandas()
# MAGIC 2. Extract y_true from sentiment_id column
# MAGIC 3. Extract y_pred from predicted_sentiment_id column
# MAGIC 4. Define target_names as ["Negative", "Positive"]
# MAGIC 5. Generate classification_report with output_dict=True
# MAGIC
# MAGIC Reference: sklearn.metrics.classification_report

# COMMAND ----------

# TODO: Generate classification report
eval_pdf = eval_df.select("sentiment_id", "predicted_sentiment_id").toPandas()

y_true = eval_pdf["sentiment_id"].astype(int)
y_pred = eval_pdf["predicted_sentiment_id"].astype(int)

target_names = ["Negative", "Positive"]

# COMMAND ----------

# Printable report 
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

# Dict version (for MLflow logging)
report_dict = classification_report(
    y_true, y_pred, target_names=target_names, output_dict=True
)
report_dict


# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Create Confusion Matrix
# MAGIC
# MAGIC TODO: Visualize model performance with confusion matrix
# MAGIC 1. Generate confusion matrix using sklearn.metrics.confusion_matrix
# MAGIC 2. Create ConfusionMatrixDisplay with target names
# MAGIC 3. Plot and display the matrix
# MAGIC
# MAGIC Confusion Matrix Layout:
# MAGIC                Predicted
# MAGIC              Neg    Pos
# MAGIC Actual  Neg   TN     FP
# MAGIC        Pos   FN     TP

# COMMAND ----------

# TODO: Create and display confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Tweet Sentiment - Confusion Matrix")
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")



# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Log Results to MLflow
# MAGIC
# MAGIC TODO: Track model performance in MLflow experiment
# MAGIC 1. Set MLflow registry to Unity Catalog: mlflow.set_registry_uri("databricks-uc")
# MAGIC 2. Get Delta table version from tweets_silver (for data lineage)
# MAGIC 3. Start MLflow run
# MAGIC 4. Log metrics:
# MAGIC    - accuracy from classification report
# MAGIC 5. Log parameters:
# MAGIC    - model_name: "workspace.default.tweet_sentiment_model"
# MAGIC    - model_version: 1
# MAGIC    - silver_delta_version: from Delta table history
# MAGIC 6. Log artifact:
# MAGIC    - confusion matrix figure as "confusion_matrix.png"
# MAGIC
# MAGIC Reference: Lab 0.5 for MLflow logging patterns

# COMMAND ----------

# TODO: Log metrics and artifacts to MLflow
mlflow.set_registry_uri("databricks-uc")

# Pin the experiment path so this evaluation run is easy to find later
current_user = spark.sql("SELECT current_user()").collect()[0][0]
mlflow_experiment_path = f"/Users/{current_user}/tweet_sentiment_eval"
mlflow.set_experiment(mlflow_experiment_path)
print(f"MLflow experiment: {mlflow_experiment_path}")

# Get the current version of the silver Delta table for lineage
try:
    silver_history = spark.sql(f"DESCRIBE HISTORY {SILVER_TABLE} LIMIT 1")
    silver_delta_version = silver_history.collect()[0]["version"]
except Exception as e:
    print(f"Could not read silver table history: {e}")
    silver_delta_version = -1
print(f"Silver delta version: {silver_delta_version}")


# COMMAND ----------

MODEL_NAME = "workspace.default.small_sentiment_model"
MODEL_VERSION = 1

with mlflow.start_run(run_name="tweet_sentiment_eval") as run:
    # Parameters - data + model lineage
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("model_version", MODEL_VERSION)
    mlflow.log_param("silver_delta_version", silver_delta_version)
    mlflow.log_param("eval_row_count", int(len(eval_pdf)))
    mlflow.log_param("dropped_unmapped_rows", int(dropped))

    # Headline metric
    mlflow.log_metric("accuracy", report_dict["accuracy"])

    # Per-class metrics (useful for spotting class imbalance / bias)
    for cls in target_names:
        mlflow.log_metric(f"precision_{cls.lower()}", report_dict[cls]["precision"])
        mlflow.log_metric(f"recall_{cls.lower()}", report_dict[cls]["recall"])
        mlflow.log_metric(f"f1_{cls.lower()}", report_dict[cls]["f1-score"])

    mlflow.log_metric("macro_f1", report_dict["macro avg"]["f1-score"])
    mlflow.log_metric("weighted_f1", report_dict["weighted avg"]["f1-score"])

    # Confusion matrix counts as metrics (handy for quick UI inspection)
    mlflow.log_metric("true_negative", int(tn))
    mlflow.log_metric("false_positive", int(fp))
    mlflow.log_metric("false_negative", int(fn))
    mlflow.log_metric("true_positive", int(tp))

    # Confusion matrix figure as an artifact
    mlflow.log_figure(fig, "confusion_matrix.png")

    run_id = run.info.run_id

print(f"Logged MLflow run: {run_id}")
print(f"Accuracy: {report_dict['accuracy']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation
# MAGIC
# MAGIC After running this notebook, verify in the MLflow UI:
# MAGIC 1. Navigate to "Experiments" tab
# MAGIC 2. Find experiment for this notebook
# MAGIC 3. Check latest run contains:
# MAGIC    - accuracy metric (e.g., 0.85 = 85% correct)
# MAGIC    - model_name, model_version, silver_delta_version parameters
# MAGIC    - confusion_matrix.png artifact
# MAGIC
# MAGIC ## Interpreting Results
# MAGIC
# MAGIC **Accuracy**:
# MAGIC - High (>80%): Model performing well
# MAGIC - Low (<70%): Consider different model or fine-tuning
# MAGIC
# MAGIC **Confusion Matrix**:
# MAGIC - Diagonal (TN, TP): Correct predictions
# MAGIC - Off-diagonal (FP, FN): Misclassifications
# MAGIC - Imbalanced: May indicate class imbalance or bias
# MAGIC
# MAGIC **Next Steps**:
# MAGIC - If accuracy low: Try different model, improve preprocessing
# MAGIC - If confusion matrix shows bias: Investigate class distribution, confidence thresholds