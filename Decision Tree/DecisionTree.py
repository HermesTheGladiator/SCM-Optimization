#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:52:23 2019

@author: heerokbanerjee
"""

from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline


sc = SparkContext('local')
spark = SparkSession(sc)



def spark_read(filename):
        file = spark.read.format("csv").option("header", "true").load(filename)
        return file


def convert_to_numeric(data):
    for x in ["WEIGHT (KG)","MEASUREMENT","QUANTITY"]:
        data = data.withColumn(x, data[x].cast('double'))
    return data


dataset=spark_read("dataset/wallmart.csv")
dataset=dataset.select("WEIGHT (KG)","MEASUREMENT","QUANTITY","US PORT"
                       ,"CARRIER CODE")
dataset=convert_to_numeric(dataset)
train_dataset,test_dataset=dataset.randomSplit([0.9,0.1])

### Pipeline Component1
### String Indexer for Column "US PORT"
portIndexer = StringIndexer(
        inputCol="US PORT",
        outputCol="Indexport",handleInvalid="skip")

carrierIndexer = StringIndexer(
        inputCol="CARRIER CODE",
        outputCol="indexcarrier",handleInvalid="skip")


vecAssembler = VectorAssembler(
        inputCols=["WEIGHT (KG)","MEASUREMENT","QUANTITY","Indexport"],
        outputCol="assembled",handleInvalid="skip")

dt = DecisionTreeClassifier(labelCol="indexcarrier", 
                            featuresCol="assembled", maxBins=100)

pipeline = Pipeline(stages=[portIndexer, carrierIndexer, vecAssembler, dt])

#train_dataset.show()
model = pipeline.fit(train_dataset)

predictions = model.transform(test_dataset)

predictions.select("WEIGHT (KG)", "MEASUREMENT", "QUANTITY","US PORT",
                   "prediction").show(5)


### Evaluation of DT Model
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexcarrier", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g" % (accuracy))


###plotting graph
eg = predictions.select("prediction","indexcarrier").limit(5000)
panda_eg= eg.toPandas()

panda_eg.plot(kind='bar',stacked="True",ylim=(0,30))