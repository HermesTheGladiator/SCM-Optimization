#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:18:54 2019
Random Forest Regression with Pyspark
@author: heerokbanerjee
"""
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import StringIndexer
from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext

#
sc = SparkContext('local')
spark = SparkSession(sc)


#Importing Dataset
dataset = spark.read.format("csv").option("header","true").load("dataset/hpd.csv")
dataset = dataset.withColumn("Order_Demand",dataset["Order_Demand"].cast('double'))
dataset=dataset.select("Product_Code", "Warehouse", "Product_Category" , "Date","Order_Demand")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in ind
CodeIndexer= StringIndexer(inputCol="Product_Code", outputCol="CodeIndex",handleInvalid="skip")
WarehouseIndexer = StringIndexer(inputCol="Warehouse", outputCol="WarehouseIndex",handleInvalid="skip")
CategoryIndexer = StringIndexer(inputCol="Product_Category", outputCol="CategoryIndex",handleInvalid="skip")
DateIndexer= StringIndexer(inputCol="Date", outputCol="DateIndex",handleInvalid="skip")

assembler = VectorAssembler(
    inputCols=["CodeIndex", "WarehouseIndex", "CategoryIndex" , "DateIndex"],
    outputCol="Ghoda", handleInvalid="skip")

DemandImputer=Imputer(inputCols=["Order_Demand"], outputCols=["Order_pure"])

(trainingData, testData) = dataset.randomSplit([0.7, 0.3])
#trainingData.show()
# Train a RandomForest model.
rf = RandomForestRegressor(labelCol="Order_pure", featuresCol="Ghoda", 
                           numTrees=20, maxBins=3000)


# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[CodeIndexer,WarehouseIndexer,CategoryIndexer,
                            DateIndexer,assembler,DemandImputer,rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

predictions.select("prediction").distinct().show()

predictions.distinct().show()

evaluator = RegressionEvaluator(
    labelCol="Order_pure", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

for i in range(1,10):
	rmse=np.sqrt(rmse)

print("Root Mean Squared Error (RMSE)= %g" % rmse)


###plotting graph
eg = predictions.select("prediction","Order_pure","Ghoda").limit(1000)
panda_eg= eg.toPandas()

panda_eg.plot(kind="bar",stacked="true")