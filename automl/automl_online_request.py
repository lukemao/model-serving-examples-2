from google.cloud import automl_v1beta1 as automl
from google.oauth2 import service_account

project_id = 'sap-ds-demo'
compute_region = 'eu'
model_display_name = 'untitled_16014882_20200930065900'
bq_input_uri = 'bq://sap-ds-demo.sap_multi_region.churn_batch'
bq_output_uri = 'bq://sap-ds-demo'
params = {'api_endpoint': 'eu-automl.googleapis.com:443'}
feature_importance = True

import pandas as pd
df = pd.read_csv('data.csv')
inputs = df.to_dict('records')

client = automl.TablesClient(
    credentials=service_account.Credentials.from_service_account_file(
        '/Users/napt/.ssh/sap-ds-demo-8970dec417c1.json'), 
        project='sap-ds-demo', region='eu', client_options=params)

if feature_importance:
    response = client.predict(
        model_display_name=model_display_name,
        inputs=inputs[0],
        feature_importance=True,
    )
else:
    response = client.predict(
        model_display_name=model_display_name, inputs=inputs
    )

print("Prediction results:")
for result in response.payload:
    print(
        "Predicted class name: {}".format(result.tables.value)
    )
    print("Predicted class score: {}".format(result.tables.score))

    if feature_importance:
        # get features of top importance
        feat_list = [
            (column.feature_importance, column.column_display_name)
            for column in result.tables.tables_model_column_info
        ]
        feat_list.sort(reverse=True)
        if len(feat_list) < 10:
            feat_to_show = len(feat_list)
        else:
            feat_to_show = 10

        print("Features of top importance:")
        for feat in feat_list[:feat_to_show]:
            print(feat)