from google.cloud import automl_v1beta1 as automl
from google.oauth2 import service_account

project_id = 'sap-ds-demo'
compute_region = 'eu'
model_display_name = 'untitled_16014882_20200930065900'
bq_input_uri = 'bq://sap-ds-demo.sap_multi_region.churn_batch'
bq_output_uri = 'bq://sap-ds-demo'
params = {'api_endpoint': 'eu-automl.googleapis.com:443'}



client = automl.TablesClient(
    credentials=service_account.Credentials.from_service_account_file(
        '/Users/napt/.ssh/sap-ds-demo-8970dec417c1.json'), 
        project='sap-ds-demo', region='eu', client_options=params)

# Query model
response = client.batch_predict(bigquery_input_uri=bq_input_uri,
                                bigquery_output_uri=bq_output_uri,
                                model_display_name=model_display_name)


print("Making batch prediction... ")

response.result()
# AutoML puts predictions in a newly generated dataset with a name by a mask "prediction_" + model_id + "_" + timestamp
# here's how to get the dataset name:

dataset_name = response.metadata.batch_predict_details.output_info.bigquery_output_dataset

print("Batch prediction complete.\nResults are in '{}' dataset.\n{}".format(
    dataset_name, response.metadata))