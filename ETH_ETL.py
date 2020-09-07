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


"""___Convert ethusd.json file data from json format to csv___"""

with open('ethusd.json', encoding='utf-8-sig') as ETH_json_df:
    ETH_df=json.load(ETH_json_df)
ETH_csv=ETH_df["results"]

# now we will open a file for writing 
data_file = open('ethusd.csv', 'w') 
  
# create the csv writer object 
csv_writer = csv.writer(data_file) 
  
# Counter variable used for writing  
# headers to the CSV file 
count = 0

for record in ETH_csv: 
    if count == 0: 
  
        # Writing headers of CSV file 
        header = record.keys() 
        csv_writer.writerow(header) 
        count += 1
  
    # Writing data of CSV file 
    csv_writer.writerow(record.values()) 
data_file.close() 

print('Writing json data to csv file: COMPLETE')
ETH_df=pd.read_csv('ethusd.csv')
ETH_df.head()
"""^^^Convert ethusd.json file data from json format to csv^^^"""


def process_eth(spark, input_data, output_data):
    print('Processing xrp data from S3 bucket...')
    """
    This function will:  Extract and process data to 
    create (df)
    """
    
    
    eth_data = input_data
    df = ETH_df
    print('Reading ltc data from S3 bucket: COMPLETE')
    
    """^^^Read in data from S3 bucket and assign it as df^^^"""

future_pred=30 ##References how far out the prediction model will predict
#coin_to_predict= "ethusd"    

main_df=pd.DataFrame()
df=ETH_df
df.set_index("time", inplace=True)
    
if len(main_df)== 0:
    main_df=df
else:
    main_df = main_df.join(df)
        
"""^^^Convert Spark dataframe to pandas dataframe & set index as 'time'^^^"""



main_df.fillna(method='bfill', inplace=True)
main_df.dropna(inplace=True)
"""^^^Back fill close data where values were NaN^^^"""



main_df['ETH_future']= main_df["close"].shift(-future_pred)
main_df.dropna(inplace=True)
eth_df=main_df[['close', 'ETH_future']]
eth_df.head()
"""^^^Create new df as ltc_df with only 'close' and 'ETH_future' columns extracted from main_df^^^"""



"""___Begin building model for ETH future prediction___"""

X= np.array(eth_df.drop(['ETH_future'], 1))[:-future_pred]
print('x:', X)
Y=np.array(eth_df['ETH_future'])[:-future_pred]
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
x_future= eth_df.drop(['ETH_future'], 1)[:-future_pred]
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
ETH_tree= eth_df[X.shape[0]:]
ETH_tree['Predictions']= predictions
ETH_tree= ETH_tree.apply(pd.to_numeric)
ETH_tree['Error_%']=(ETH_tree['ETH_future']-ETH_tree['Predictions'])/ETH_tree['ETH_future']*(100)
ETH_tree['ETH_future'].dtype

"""^^^Create new column 'Error_%' to view the percentage of error between the predictions and actual future value.
[((future value) - (prediction value) / (future value))* 100] ^^^"""

ETH_tree.reset_index(inplace=True) 
ETH_tree.head()
"""^^^Reset index for valid_tree dataframe^^^"""

predictions= lr_pred
ETH_lr= eth_df[X.shape[0]:]
ETH_lr['Predictions']= predictions
ETH_lr= ETH_lr.apply(pd.to_numeric)
ETH_lr['Error_%']=(ETH_lr['ETH_future']-ETH_lr['Predictions'])/ETH_lr['ETH_future']*(100)
ETH_lr.reset_index(inplace=True)
ETH_lr.head()
"""^^^Create new column 'Error_%' to view the percentage of error between the predictions and actual future value.
[((future value) - (prediction value) / (future value))* 100] ^^^"""
"""^^^Reset index for valid_tree dataframe^^^"""


ETH_tree['time'] = pd.to_datetime(ETH_tree['time'],unit='ms')
ETH_lr['time'] = pd.to_datetime(ETH_lr['time'],unit='ms')
"""^^^Convert the 'time' column from epoch seconds to pandas datetime^^^"""


ETH_tree_spark = spark.createDataFrame(ETH_tree)
ETH_tree_spark=ETH_tree_spark.withColumn('time', col('time').cast('Timestamp'))
ETH_tree_spark=ETH_tree_spark.withColumn('close', col('close').cast('Float'))
ETH_tree_spark=ETH_tree_spark.withColumn('ETH_future', col('ETH_future').cast('Float'))
ETH_tree_spark=ETH_tree_spark.withColumn('Predictions', col('Predictions').cast('Float'))
ETH_tree_spark=ETH_tree_spark.withColumn('Error_%', col('Error_%').cast('Float'))
ETH_tree_spark.printSchema()
ETH_tree_spark.show()
"""^^^Convert data back to Spark dataframes and cast the appropriate data types and print schema to verify^^^"""



ETH_lr_spark = spark.createDataFrame(ETH_lr)
ETH_lr_spark=ETH_lr_spark.withColumn('time', col('time').cast('Timestamp'))
ETH_lr_spark=ETH_lr_spark.withColumn('close', col('close').cast('Float'))
ETH_lr_spark=ETH_lr_spark.withColumn('ETH_future', col('ETH_future').cast('Float'))
ETH_lr_spark=ETH_lr_spark.withColumn('Predictions', col('Predictions').cast('Float'))
ETH_lr_spark=ETH_lr_spark.withColumn('Error_%', col('Error_%').cast('Float'))
ETH_lr_spark.printSchema()
ETH_lr_spark.show()
"""^^^Convert data back to Spark dataframes and cast the appropriate data types and print schema to verify^^^"""



ETH_tree_spark.write.partitionBy('time').parquet("s3a://dend-capstone-output/"+ 'ETH_tree_spark')
ETH_lr_spark.write.partitionBy('time').parquet("s3a://dend-capstone-output/"+ 'ETH_lr_spark')
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