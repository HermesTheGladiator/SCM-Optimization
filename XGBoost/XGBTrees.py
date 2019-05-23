#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 02:12:41 2019
XGBoosted Classification with Pyspark and xgb lib
@author: heerokbanerjee
"""

import numpy as np

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
#from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext

from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import xgboost as xgb

sc = SparkContext('local')
spark = SparkSession(sc)

fname_train = "dataset/wallmart.csv"


def spark_read(filename):
        file = spark.read.format("csv").option("header", "true").load(filename)
        return file

def convert_to_numeric(data):
    for x in ["WEIGHT (KG)","MEASUREMENT","QUANTITY"]:
        data = data.withColumn(x, data[x].cast('double'))
    return data
    
### Import Training dataset
data = spark_read(fname_train)
data=data.select("ARRIVAL DATE","WEIGHT (KG)","MEASUREMENT","QUANTITY","CARRIER CITY")
(train_data, test_data)=data.randomSplit([0.8,0.2])
train_data=convert_to_numeric(train_data)

### Pipeline Component1
### String Indexer for Column "Timestamp"
###
dateIndexer = StringIndexer(
        inputCol="ARRIVAL DATE",
        outputCol="dateIndex",handleInvalid="skip")
#print(strIndexer.getOutputCol())

#indexer_out.show()

### Pipeline Component2
### String Indexer for Column "Label"
###
carrierIndexer = StringIndexer(
        inputCol="CARRIER CITY",
        outputCol="carrierIndex",handleInvalid="skip")
#print(strIndexer.getOutputCol())
#out2 = labelIndexer.fit(train_data).transform(train_data)

### Pipeline Component2
### VectorAssembler
###
vecAssembler = VectorAssembler(
        inputCols=["WEIGHT (KG)","MEASUREMENT","QUANTITY","dateIndex"],
        outputCol="vecFea",handleInvalid="skip")
#assembler_out = vecAssembler.transform(indexer_out)
#assembler_out.select("vecFea").show(truncate=False)

### Pipeline Component3
### GBT Classifier
#dt_class=DecisionTreeClassifier(labelCol="IndexLabel", featuresCol="vecFea")

### Training- Pipeline Model
### 
pipe=Pipeline(stages=[dateIndexer,carrierIndexer,vecAssembler])
pipe_model=pipe.fit(train_data)

output=pipe_model.transform(train_data)
out_vec=output.select("dateIndex","vecFea").show(10)

num_classes=output.select("carrierIndex").distinct().count()
print(num_classes)
### XGBoostClassifier Model
###  
###
params = {	"objective":"reg:linear",
			'colsample_bytree': 0.6,
			'learning_rate': 0.38,
            'max_depth': 40,
            'alpha': 8,
            'n_estimators':50}

features_x=np.array(output.select("vecFea").collect())
labels_y=np.array(output.select("QUANTITY").collect())
print(max(labels_y))
features_x=np.squeeze(features_x,axis=1)
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(features_x,labels_y,test_size=0.51,random_state=123)


xgb_train = xgb.DMatrix(X_train, label=Y_train)
xgb_test = xgb.DMatrix(X_test, label=Y_test)

#xgbmodel=XGBClassifier()

xg_reg = xgb.train(params=params, dtrain=xgb_train, num_boost_round=1000)
print(xg_reg)

### Testing Pipeline + XGBoostClassifier
###
#test_output=pipe_model.transform(test_data)

xgb_output=xg_reg.predict(xgb_test)
print(xgb_output)

#predictions = np.asarray([np.argmax(line) for line in xgb_output])
#print(predictions)

### Determining Accuracy Score
###
mse = mean_squared_error(Y_test, xgb_output)
for i in range(1,5):
	mse=np.sqrt(mse)

print("MSE: ",mse)





