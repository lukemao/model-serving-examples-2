# model-serving-example using ai-platform custom model
## Model deployment
### Step 1: Build Predictor code
* Here you need to create a predictor class with prediction function. An
 example is predictor.py in tf_code folder.
* You need to create setup.py file. An example is available in tf_code folder.
* build the distribution code by running:
```commandline
setup.py sdist --formats=gztar
```

### Step 2: Create ai-platform model
Example of command
```commandline
gcloud beta ai-platform models create email_ai --enable-logging --enable-console-logging
```
Please note that you can specify the region in above command and use the
 default region. 
 
### Step 3: move required files into GCS.
Copy both python gzipped distribution predictor package and model file into
 a GCS folder. 
 
### Step 4: Create a model version.
Example of command:
```commandline
gcloud beta ai-platform versions create v1 --model email_ai --origin gs://landing-data-models/models/ --python-version 3.7 --runtime-version 1.15 --machine-type mls1-c4-m4 --package-uris "gs://landing-data-models/models/my_custom_code-0.1.tar.gz"  --prediction-class predictor.MyPredictor
```

## Model prediction

There are number of ways to do the prediction:

* Build a prediction pieline using Airflow composer
* Use aiplatform consul. Go to ai-platform > models > version > TEST&USE, then if you have sample data you can perform the prediction. An example of sample data is sample_data_consul.json.
* Use gcloud commands:
```
MODEL_NAME="churn_random_forest"
VERSION_NAME="v1"
INPUT_DATA_FILE="data_churn_small.json"
gcloud ml-engine predict --model $MODEL_NAME --version $VERSION_NAME --json-instances $INPUT_DATA_FILE
```
