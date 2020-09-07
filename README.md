<h2>Introduction</h2>

In this Capstone project, I wanted to perform an ETL process on a dataset of my choosing and I chose to work with some historical cryptocurrency data for Litecoin(LTC) and Ethereum(ETH). I wanted to create a ETL pipeline that would also include some basic machine learning implementations using linear regression and decision tree models to predict future closing prices.
The pipeline will extract csv and json data from an S3 bucket and will perform data cleaning tasks and implement the ML models and save the results into a new table. Then the tables are written to another S3 bucket in parquet format, for analytical use.


<h2>Data Source</h2>
https://www.kaggle.com/tencars/392-crypto-currency-pairs-at-minute-resolution


<h2>Tools</h2>
AWS S3, PySpark, AWS EMR, Jupyter Notebook

<h2> Data</h2>

<h4>LTC Dataset</h4>
LTC data shape (1663435, 6)

columns= 'time','open','close','high','low','volume'

values= '1368976980000','3.1491','3.1491','3.1491','3.1491','10.000000'

Definitions= 
'time'= Timestamp of trades made, represented as Epoch
'open'= cryptocurrency price at day open
'high'= price max for the day or highest value of currency for the day
'low'= price min for the day or lowest value of currency for the day
'volume'= Size or volume of trades per timestamp


<h4>ETH Dataset</h4>
columns= 'time','open','close','high','low','volume'

values= {'time': 1595809620000, 'open': 312.89333, 'close':...}

Definitions= 
'time'= Timestamp of trades made, represented as Epoch
'open'= cryptocurrency price at day open
'high'= price max for the day or highest value of currency for the day
'low'= price min for the day or lowest value of currency for the day
'volume'= Size or volume of trades per timestamp


<h2>Implementation</h2>

1. Create AWS account and obtain credentials for dl.cfg file
2. Open terminal and execute LTC_ETL.py for Litecoin(LTC) or ETH_ETL.py for Ethereum(ETH)
3. Alternatively you can run through the Jupyter notebook

<h2>Future Scenarios and Update</h2>

<h4>Data Updates</h4>
For this project I used a batch approach and in the future, this data would ideally be set up for streaming and real time live predictions of cryptocurrency prices.

<h4>Data Size</h4>
If the data size was increased by a 100x, ideally we would make some changes to architecture and would have different tools and options available but in regards to scaling up I think AWS EMR is a fantastic option and really allows for more customization for you Spark clusters. However if we were setting up live data streaming, I believe looking at Redshift would make sense, as you can have a cluster running 24/7, where as with EMR is ideal if you dont need to run your cluster 24/7.

<h4>Daily Run</h4>
For scheduling a daily run of the ETL pipeline, using Airflow would be a must. Would need to set up DAG's and adjust python scripts to implement running the pipeline daily. Would not suggest running through a notebook environment for actual implementation.

<h4>Access</h4>
If the database needed to be accessed by 100+ people, it would not really pose a major issue at all. The AWS access protocols would need to be followd for the larger user scale. S3 access, IAM Users and cluster access would need to be assessed and making sure the appropriate user has the appropriate access.