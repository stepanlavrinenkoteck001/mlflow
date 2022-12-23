# Databricks notebook source
# MAGIC %md # MLOps using MLFlow
# MAGIC  MLflow is an MLOps tool that enables data scientist to quickly productionalization of their Machine Learning projects. To achieve this, MLFlow has four major components which are Tracking, Projects, Models, and Registry. MLflow lets you train, reuse, and deploy models with any library and package them into reproducible steps. MLflow is designed to work with any machine learning library and require minimal changes to integrate into an existing codebase. In this session, we will cover the common pain points of machine learning developers such as tracking experiments, reproducibility, deployment tool and model versioning. Ready to get your hands dirty by doing quick ML project using mlflow and release to production to understand the ML-Ops lifecycle.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### DEMO details
# MAGIC 
# MAGIC ######In this Demo Notebook we are taking a look at:
# MAGIC  * How to set up a ElasticNet Model in Spark.
# MAGIC  * How to create an MLFlow experiment.
# MAGIC  * How to track model params and metrics with MLFlow.
# MAGIC  * How to deploy Model to different environment using MLFlow Model Registry.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src ='/files/MLFlow_Img.png' height=200 margin: 20px>

# COMMAND ----------

# MAGIC %md ### Training a model and adding to the mlFlow registry

# COMMAND ----------

dbutils.widgets.text(name = "model_name", defaultValue = "mlops-demo-wine-model", label = "Model Name")
dbutils.widgets.text(name = "stage", defaultValue = "staging", label = "Stage")
dbutils.widgets.text(name = "git_branch", defaultValue = "main", label = "git_branch")
dbutils.widgets.text(name = "model_type", defaultValue = "sklearn ElasticNet", label = "model_type")
dbutils.widgets.text(name = "created_by", defaultValue = "stepan.lavrinenko@teck.com", label = "created_by")

# COMMAND ----------

model_name=dbutils.widgets.get("model_name")
stage = dbutils.widgets.get("stage")
git_branch = dbutils.widgets.get("git_branch")
model_type = dbutils.widgets.get("model_type")
created_by = dbutils.widgets.get("created_by")

# COMMAND ----------

# MAGIC %md ### Connect to an MLflow tracking server
# MAGIC 
# MAGIC MLflow can collect data about a model training session, such as validation accuracy. It can also save artifacts produced during the training session, such as a PySpark pipeline model.
# MAGIC 
# MAGIC By default, these data and artifacts are stored on the cluster's local filesystem. However, they can also be stored remotely using an [MLflow Tracking Server](https://mlflow.org/docs/latest/tracking.html).

# COMMAND ----------

import mlflow
mlflow.__version__

# Using the hosted mlflow tracking server

# COMMAND ----------

# MAGIC %md ## Training a model

# COMMAND ----------

# MAGIC %md
# MAGIC Navigate to https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ and download both winequality-red.csv  to your local machine. We should already have one at this path here:

# COMMAND ----------

wine_data_path = "/dbfs/FileStore/mlflow_tutorial/winequality_red.csv"

# COMMAND ----------

# MAGIC %md ### In an MLflow run, train and save an ElasticNet model for rating wines
# MAGIC 
# MAGIC Using Scikit-learn's Elastic Net regression module, we will train wine quality dataset. We will use mlflow tracking server to save performance metrics, hyperparameter data, and model artifacts for future reference. mlflow tracking server will persist metrics and artifact, allowing other users to view and download it. For more information about model tracking in MLflow, see the [MLflow tracking reference](https://www.mlflow.org/docs/latest/tracking.html).
# MAGIC 
# MAGIC Later, we will use the saved MLflow model artifacts to deploy the trained model to Azure ML for serving.

# COMMAND ----------

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_model(wine_data_path, model_path, alpha, l1_ratio, git_branch, model_type, created_by):
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    data = pd.read_csv(wine_data_path, sep=None)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]


    # Start a new MLflow training run 
    with mlflow.start_run(tags = {'git_branch':git_branch, 
                                  'model_type' : model_type, 
                                  'created_by' : created_by}):
        # Fit the Scikit-learn ElasticNet model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        # Evaluate the performance of the model using several accuracy metrics
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log model hyperparameters and performance metrics to the MLflow tracking server
        # (or to disk if no)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, model_path)
        
        return mlflow.active_run().info.run_uuid

# COMMAND ----------

alpha_1 = 0.60
l1_ratio_1 = 0.7
model_path = 'model'
run_id1 = train_model(wine_data_path=wine_data_path, 
                      model_path=model_path, 
                      alpha=alpha_1, 
                      l1_ratio=l1_ratio_1, 
                      git_branch = git_branch,
                      model_type = model_type,
                      created_by = created_by)
model_uri = "runs:/"+run_id1+"/model"

# COMMAND ----------


# do a bunch more runs, just so we can compare results
# optimally, you should be using a cross-validator / hyperparaemeter optimizer here
for alpha_1, l1_ratio_1 in zip([0.65,0.66, 0.7, 0.75], [0.7, 0.75, 0.8, 0.85]):
  model_path = 'model'
  run_id1 = train_model(wine_data_path=wine_data_path, 
                        model_path=model_path, 
                        alpha=alpha_1, 
                        l1_ratio=l1_ratio_1, 
                        git_branch = git_branch,
                        model_type = model_type,
                        created_by = created_by)
  model_uri = "runs:/"+run_id1+"/model"
#   print(alpha_l, l1_ratio_l)

# COMMAND ----------

print(model_uri)

# COMMAND ----------

# MAGIC %md ## Register the Model in the Model Registry

# COMMAND ----------

import time
result = mlflow.register_model(
    model_uri,
    model_name
)
time.sleep(10)
version = result.version

# COMMAND ----------

# MAGIC %md ### Transitioning the model to 'Staging"

# COMMAND ----------

import mlflow
client = mlflow.tracking.MlflowClient()

client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="staging")

# COMMAND ----------

# MAGIC %md ### Get the latest version of the model that was put into the current stage

# COMMAND ----------

import mlflow
import mlflow.sklearn

client = mlflow.tracking.MlflowClient()
latest_model = client.get_latest_versions(name = model_name, stages=[stage])
print(latest_model[0])

# COMMAND ----------

model_uri="runs:/{}/model".format(latest_model[0].run_id)
latest_sk_model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------


