import configparser
from datetime import datetime
import time
import os
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, col, to_date
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql import types as t
from pyspark.sql.types     import IntegerType, TimestampType, DoubleType, StructType, StructField
from pyspark.sql.types     import *
from pyspark.sql.functions import to_timestamp
import boto3
from botocore.client import Config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import csv
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#!pip install jupyters3
import jupyters3

"""^^^Install all tools and import all required libraries^^^"""



config = configparser.ConfigParser()
config.read_file(open('dl.cfg'))

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']

"""^^^Congifure AWS access^^^"""



spark = SparkSession.builder\
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0")\
    .enableHiveSupport().getOrCreate()

print('Creating Spark session: COMPLETE')

"""^^^Creating Spark session for data processing, if it does not currently exist^^^"""



s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)

"""^^^ Utilize boto3 library to verify connection by printing out S3 buckets^^^"""



LTC_csv=spark.read.csv('s3a://dend-capstone-crypto-bucket/ltcusd.csv', header=True)
ETH_json=spark.read.json('s3a://capston-bucket2/ethusd.json')
LTC_csv.printSchema()
ETH_json.printSchema()
"""^^^Read in S3 data using Spark and print schema^^^"""

print(f'LTC_csv dataframe shape:',(LTC_csv.count(), len(LTC_csv.columns)))
LTC_csv.select([count(when(isnan(c), c)).alias(c) for c in LTC_csv.columns]).show()
"""^^^Data Quality check #1--Check shape of dataframe and if any columns contain NaN values^^^"""


def process_ltc(spark, input_data, output_data):
    print('Processing xrp data from S3 bucket...')
    """
    This function will:  Extract and process data to 
    create (df)
    """
    
    
    ltc_data = input_data
    df = LTC_csv
    print('Reading ltc data from S3 bucket: COMPLETE')
    
    """^^^Read in data from S3 bucket and assign it as df^^^"""

future_pred=30 ##References how far out the prediction model will predict
#coin_to_predict= "ltcusd"    

main_df=pd.DataFrame()
df=LTC_csv.toPandas()
df.set_index("time", inplace=True)
    
if len(main_df)== 0:
    main_df=df
else:
    main_df = main_df.join(df)
        
"""^^^Convert Spark dataframe to pandas dataframe & set index as 'time'^^^"""



main_df.fillna(method='bfill', inplace=True)
main_df.dropna(inplace=True)
"""^^^Back fill close data where values were NaN^^^"""



main_df['LTC_future']= main_df["close"].shift(-future_pred)
main_df.dropna(inplace=True)
ltc_df=main_df[['close', 'LTC_future']]
ltc_df.head()
"""^^^Create new df as ltc_df with only 'close' and 'LTC_future' columns extracted from main_df^^^"""



"""___Begin building model for LTC future prediction___"""

X= np.array(ltc_df.drop(['LTC_future'], 1))[:-future_pred]
print('x:', X)
Y=np.array(ltc_df['LTC_future'])[:-future_pred]
print('y:',Y)

#Split data 75% training
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
"""^^^Split training and test data with 75/25 split^^^"""

# Decision tree regressor model
tree= DecisionTreeRegressor().fit(x_train, y_train)
#Linear Regression model
lr= LinearRegression().fit(x_train, y_train)
"""^^^Create Decision Tree and Linear Regresion Models^^^"""

#Get last rows of future
x_future= ltc_df.drop(['LTC_future'], 1)[:-future_pred]
x_future= x_future.tail(future_pred)
x_future= np.array(x_future)
x_future


#Show model tree prediction
tree_pred= tree.predict(x_future)
print(f'Decission Tree: {tree_pred}')
print()
#Show Linear model pred
lr_pred= lr.predict(x_future)
print(f'Linear Regression: {lr_pred}')


predictions= tree_pred
valid_tree= ltc_df[X.shape[0]:]
valid_tree['Predictions']= predictions
valid_tree= valid_tree.apply(pd.to_numeric)
valid_tree['Error_%']=(valid_tree['LTC_future']-valid_tree['Predictions'])/valid_tree['LTC_future']*(100)
valid_tree['LTC_future'].dtype

"""^^^Create new column 'Error_%' to view the percentage of error between the predictions and actual future value.
[((future value) - (prediction value) / (future value))* 100] ^^^"""

valid_tree.reset_index(inplace=True) 
valid_tree.head()
"""^^^Reset index for valid_tree dataframe^^^"""

predictions= lr_pred
valid_lr= ltc_df[X.shape[0]:]
valid_lr['Predictions']= predictions
valid_lr= valid_lr.apply(pd.to_numeric)
valid_lr['Error_%']=(valid_lr['LTC_future']-valid_lr['Predictions'])/valid_lr['LTC_future']*(100)
valid_lr.reset_index(inplace=True)
valid_lr.head()
"""^^^Create new column 'Error_%' to view the percentage of error between the predictions and actual future value.
[((future value) - (prediction value) / (future value))* 100] ^^^"""
"""^^^Reset index for valid_tree dataframe^^^"""


valid_tree['time'] = pd.to_datetime(valid_tree['time'],unit='ms')
valid_lr['time'] = pd.to_datetime(valid_lr['time'],unit='ms')
"""^^^Convert the 'time' column from epoch seconds to pandas datetime^^^"""


valid_tree_spark = spark.createDataFrame(valid_tree)
valid_tree_spark=valid_tree_spark.withColumn('time', col('time').cast('Timestamp'))
valid_tree_spark=valid_tree_spark.withColumn('close', col('close').cast('Float'))
valid_tree_spark=valid_tree_spark.withColumn('LTC_future', col('LTC_future').cast('Float'))
valid_tree_spark=valid_tree_spark.withColumn('Predictions', col('Predictions').cast('Float'))
valid_tree_spark=valid_tree_spark.withColumn('Error_%', col('Error_%').cast('Float'))
valid_tree_spark.printSchema()
valid_tree_spark.show()
"""^^^Convert data back to Spark dataframes and cast the appropriate data types and print schema to verify^^^"""



valid_lr_spark = spark.createDataFrame(valid_lr)
valid_lr_spark=valid_lr_spark.withColumn('time', col('time').cast('Timestamp'))
valid_lr_spark=valid_lr_spark.withColumn('close', col('close').cast('Float'))
valid_lr_spark=valid_lr_spark.withColumn('LTC_future', col('LTC_future').cast('Float'))
valid_lr_spark=valid_lr_spark.withColumn('Predictions', col('Predictions').cast('Float'))
valid_lr_spark=valid_lr_spark.withColumn('Error_%', col('Error_%').cast('Float'))
valid_lr_spark.printSchema()
valid_lr_spark.show()
"""^^^Convert data back to Spark dataframes and cast the appropriate data types and print schema to verify^^^"""

print(f'valid_tree_spark dataframe shape:',(valid_tree_spark.count(), len(valid_tree_spark.columns)))
#valid_tree_spark.select([count(when(isnan(c), c)).alias(c) for c in valid_tree_spark.columns]).show()

print(f'valid_lr_spark dataframe shape:',(valid_lr_spark.count(), len(valid_lr_spark.columns)))
#valid_lr_spark.select([count(when(isnan(c), c)).alias(c) for c in valid_lr_spark.columns]).show()

"""^^^Data Quality check #2-- Checking shape of dataframes^^^"""


valid_tree_spark.write.partitionBy('time').parquet("s3a://dend-capstone-output/"+ 'valid_tree_spark')
valid_lr_spark.write.partitionBy('time').parquet("s3a://dend-capstone-output/"+ 'valid_lr_spark')
print('Write parquet files to S3 files & partition by time: COMPLETE')
"""^^^Write decision tree and linear regression tables to S3 bucket as parquet files partitioned on 'time'^^^"""


    
def main():
    spark
    input_data = "s3a://dend-capstone-crypto-bucket/"
    output_data = "s3a://dend-capstone-output/"
    process_ltc
    #write_parq()
    
    #process_xrp(spark, input_data, output_data)
    print('ETL PROCESSING COMPLETE!')
    
    
    
if __name__=="__main__":
    main()