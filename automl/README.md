* Step 1: Data set needs to be in  US multi-region or  European multi-region.
* Step 2: In AutoML table Transfer your dataset to AutoML DATASET (European
 Union or Global) 
 from BigQuery, Cloud storage, or your computer. Data Example is sap-ds-demo:sap_multi_region.churn_indicator_with_label
Minimum row 10000 rows (limitation)
* Step 3: Set target,  set parameters and  start training
* Step 4: deploy the model 
    * Batch Prediction (Input/output BQ or GCS ), this doesn't need deployment. 
    * Online Prediction, this service require deployment. Online prediction deploys your model so you can send real-time REST requests to it. Online prediction is useful for time-sensitive predictions. After model deployment you can use API endpoint or consul for prediction. for deployment go to AutoML>Models>TEST&USE>ONLINE PREDICTION

* Step 5: Test both batch and online deployed models
    * Batch prediction (Input/output BQ or GCS ): 
    
        There are number of ways to run it:
        * Consul: Got to AutoML>Models>TEST&USE>BATCH PREDICTION, It is a two step process: define batch input and output location. Example sap_multi_region.churn_batch as input and project sap-ds-demo
        * API call: Please see automl_bach_request.py
    * Online prediction, please note this service require deployment. After model deployment you can use API endpoint or consul for prediction. 
        * Consul: Got to AutoML>Models>TEST&USE>ONLINE PREDICTION
        * API call: please see automl_online_request.py


