* Step1: Train a model using Query 1 in training.sql

* Step2: Evaluate a model using Query 2 in training.sql

* Step3: Extract to GCS

* Step4: If the extracted model is'nt in EU or US Multi region, transfer it
into EU/Us multi-region

* Step5: Deploy the model in ai-platform

* Srep6: Test the model, example is, create your variables 
    * MODEL_NAME="bq_sap_churn_logistic"
    * INPUT_DATA_FILE="[INPUT-JSON]"
    * VERSION_NAME="v1"
    * REGION="europe-west4"

Then run: 
gcloud ml-engine predict --model $MODEL_NAME --version $VERSION_NAME --json-instances $INPUT_DATA_FILE --region $REGION